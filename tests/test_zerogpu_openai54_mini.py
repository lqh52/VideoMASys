"""Pinned non-reasoning GPT-5.4 mini contract; mock HTTP only, never paid calls."""

import base64
import copy
import io
import json
import math

import httpx
import pytest
from PIL import Image

from videomasys.zerogpu.core import Caption, Observation, QAPlan
from videomasys.zerogpu.openai import (
    GPT41_MINI_MODEL,
    GPT54_MINI_MODEL,
    OPENAI_MODEL,
    OPENAI_SYSTEM_PROMPT,
    OPENAI_URL,
    OpenAI41MiniVLM,
    OpenAI54MiniVLM,
    OpenAIVLM,
    strict_json_schema,
)
from videomasys.zerogpu.remote import HostedVLMError

TOKEN = "sk-fixture-only-never-log"


def completion(model=GPT54_MINI_MODEL, text="Exact answer"):
    return httpx.Response(
        200,
        json={
            "model": model,
            "choices": [
                {
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 2,
                "total_tokens": 22,
                "prompt_tokens_details": {"cached_tokens": 10},
                "completion_tokens_details": {"reasoning_tokens": 0},
            },
        },
    )


@pytest.fixture
def make_adapter():
    resources = []

    def build(handler, **kwargs):
        client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)
        adapter = OpenAI54MiniVLM(TOKEN, client=client, retry_backoff_seconds=0, **kwargs)
        resources.append((adapter, client))
        return adapter

    yield build
    for adapter, client in resources:
        adapter.close()
        client.close()


def test_pinned_non_reasoning_payload_and_plaintext_remain_explicit(make_adapter, monkeypatch):
    for name in ("OPENAI_BASE_URL", "HF_ENDPOINT", "HTTPS_PROXY"):
        monkeypatch.setenv(name, "https://attacker.invalid")
    monkeypatch.setenv("VLM_MODEL", "gpt-5.4")
    requests, admissions = [], []
    adapter = make_adapter(
        lambda request: requests.append(request) or completion(),
        before_request=lambda: admissions.append(1),
    )
    schema = Caption.model_json_schema()
    assert adapter.generate_structured("Caption frames", [], 500, schema) == "Exact answer"
    assert adapter("Choose exact option", [], 120) == "Exact answer"
    strict, plain = [json.loads(request.content) for request in requests]
    for body, tokens in ((strict, 500), (plain, 120)):
        assert body["model"] == GPT54_MINI_MODEL
        assert body["reasoning_effort"] == "none"
        assert body["temperature"] == 0
        assert body["max_completion_tokens"] == tokens
        assert body["store"] is False and body["stream"] is False
        assert body["service_tier"] == "default"
        assert not {"max_tokens", "repetition_penalty", "tools"} & body.keys()
        assert body["messages"][0] == {"role": "system", "content": OPENAI_SYSTEM_PROMPT}
    assert strict["response_format"]["json_schema"] == {
        "name": "video_evidence",
        "strict": True,
        "schema": strict_json_schema(schema),
    }
    assert "response_format" not in plain
    assert len(admissions) == 2
    assert all(str(request.url) == OPENAI_URL for request in requests)
    assert all(request.headers["Authorization"] == "Bearer " + TOKEN for request in requests)
    assert adapter.calls[-1]["generation_parameters"] == {
        "reasoning_effort": "none",
        "max_completion_tokens": 120,
        "store": False,
        "service_tier": "default",
        "stream": False,
        "temperature": 0,
    }
    assert adapter.calls[0]["requested_provider"] == "OpenAI"
    assert adapter.calls[0]["attempts"][0]["usage"] == {
        "prompt_tokens": 20,
        "completion_tokens": 2,
        "total_tokens": 22,
        "prompt_tokens_details": {"cached_tokens": 10},
        "completion_tokens_details": {"reasoning_tokens": 0},
    }
    assert TOKEN not in json.dumps(adapter.calls)


@pytest.mark.parametrize(
    "model",
    [
        OPENAI_MODEL,
        GPT41_MINI_MODEL,
        "gpt-5.4-mini",
        "gpt-5.4-mini-2026-03-18",
        GPT54_MINI_MODEL + "-extra",
        "gpt-5.4",
        "Qwen/Qwen3-VL-30B-A3B-Instruct",
    ],
)
def test_config_and_reported_model_require_exact_pinned_snapshot(make_adapter, model):
    with pytest.raises(HostedVLMError) as failure:
        OpenAI54MiniVLM(TOKEN, model=model)
    assert failure.value.code == "request_rejected"
    requests = []
    adapter = make_adapter(lambda request: requests.append(request) or completion(model))
    with pytest.raises(HostedVLMError) as failure:
        adapter("Exact option", [], 100)
    assert failure.value.code == "model_mismatch" and len(requests) == 1
    assert json.loads(requests[0].content)["model"] == GPT54_MINI_MODEL


@pytest.mark.parametrize(
    "adapter_class,model,reasoning",
    [(OpenAIVLM, OPENAI_MODEL, "none"), (OpenAI41MiniVLM, GPT41_MINI_MODEL, None)],
)
def test_existing_adapters_keep_their_defaults_and_reject_new_model(
    adapter_class, model, reasoning
):
    with pytest.raises(HostedVLMError):
        adapter_class(TOKEN, model=GPT54_MINI_MODEL)
    with httpx.Client(transport=httpx.MockTransport(lambda request: completion(model))) as client:
        adapter = adapter_class(TOKEN, client=client)
        assert adapter.model == model
        parameters = adapter._generation_parameters(100)
        assert parameters.get("reasoning_effort") == reasoning
        assert ("temperature" in parameters) is (adapter_class is OpenAI41MiniVLM)
        adapter.close()


@pytest.mark.parametrize("schema_class", [Caption, Observation, QAPlan])
def test_strict_schema_normalization_is_inherited_and_does_not_mutate_contract(
    make_adapter, schema_class
):
    requests = []
    schema = schema_class.model_json_schema()
    original = copy.deepcopy(schema)
    adapter = make_adapter(lambda request: requests.append(request) or completion(text="{}"))
    adapter.generate_structured("Schema fixture", [], 800, schema)
    sent = json.loads(requests[0].content)["response_format"]["json_schema"]["schema"]
    assert sent == strict_json_schema(original) and schema == original
    assert sent["additionalProperties"] is False
    assert sent["required"] == list(sent["properties"])


@pytest.mark.parametrize(
    "count,expected,image_tokens",
    [(1, (640, 480), 360), (6, (640, 480), 360), (18, (512, 384), 231)],
)
def test_bounded_high_detail_images_fit_the_existing_admission_allowance(
    make_adapter, tmp_path, count, expected, image_tokens
):
    frame = tmp_path / "frame.png"
    Image.new("RGB", (1280, 960), "red").save(frame)
    requests = []
    adapter = make_adapter(lambda request: requests.append(request) or completion())
    prompt = "Inspect the frames"
    adapter(prompt, [frame] * count, 800)
    content = json.loads(requests[0].content)["messages"][1]["content"]
    for item in content[:-1]:
        assert item["image_url"]["detail"] == "high"
        prefix, data = item["image_url"]["url"].split(",", 1)
        assert prefix == "data:image/jpeg;base64"
        with Image.open(io.BytesIO(base64.b64decode(data))) as image:
            assert image.size == expected
            # Official image guide: 32px patches, 1.2x multiplier, no upscaling.
            patches = math.ceil(image.width / 32) * math.ceil(image.height / 32)
            assert math.ceil(patches * 1.2) == image_tokens < 600
    bound = adapter.calls[0]["context_token_bound"]
    assert bound == len((OPENAI_SYSTEM_PROMPT + prompt).encode()) + 256 + 600 * count + 800
    assert bound <= 32_768


@pytest.mark.parametrize(
    "code",
    ["insufficient_quota", "credit_balance_exhausted", "project_spend_limit_exceeded"],
)
def test_billing_stops_without_model_fallback_and_redacts_credentials(make_adapter, code):
    requests, admissions = [], []
    adapter = make_adapter(
        lambda request: (
            requests.append(request)
            or httpx.Response(429, json={"error": {"code": code, "message": TOKEN}})
        ),
        before_request=lambda: admissions.append(1),
    )
    with pytest.raises(HostedVLMError) as failure:
        adapter("caption", [], 100)
    assert failure.value.code == "credits" and len(requests) == len(admissions) == 1
    assert TOKEN not in str(failure.value) + json.dumps(adapter.calls)
    assert json.loads(requests[0].content)["model"] == GPT54_MINI_MODEL


@pytest.mark.parametrize("status", [307, 429, 503])
def test_redirect_block_and_per_attempt_admission_remain_bounded(make_adapter, status):
    requests, admissions = [], []
    adapter = make_adapter(
        lambda request: (
            requests.append(request)
            or httpx.Response(status, headers={"Location": "https://attacker.invalid"})
        ),
        before_request=lambda: admissions.append(1),
    )
    with pytest.raises(HostedVLMError):
        adapter.generate_structured("caption", [], 100, Caption.model_json_schema())
    assert len(requests) == len(admissions) == (1 if status == 307 else 2)
    assert all(str(request.url) == OPENAI_URL for request in requests)
    assert all(json.loads(request.content)["model"] == GPT54_MINI_MODEL for request in requests)


def test_request_limits_and_closed_session_fail_before_admission(make_adapter):
    requests, admissions = [], []
    adapter = make_adapter(
        lambda request: requests.append(request) or completion(),
        before_request=lambda: admissions.append(1),
    )
    for prompt, paths, tokens, schema in (
        ("caption", [], 500, {"type": "object"}),
        ("caption", [], 1601, Caption.model_json_schema()),
        ("x" * 32_001, [], 100, Caption.model_json_schema()),
        ("caption", ["missing"] * 19, 100, Caption.model_json_schema()),
    ):
        with pytest.raises(HostedVLMError) as failure:
            adapter.generate_structured(prompt, paths, tokens, schema)
        assert failure.value.code == "invalid_request"
    adapter.close()
    with pytest.raises(HostedVLMError) as failure:
        adapter("caption", [], 100)
    assert failure.value.code == "closed"
    assert not requests and not admissions
    with pytest.raises(HostedVLMError):
        OpenAI54MiniVLM(TOKEN, timeout_seconds=61)
