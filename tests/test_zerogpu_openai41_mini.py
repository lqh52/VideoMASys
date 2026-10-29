"""Pinned comparator only; mock HTTP, no inference or automatic fallback."""

import base64
import io
import json

import httpx
import pytest
from PIL import Image

from videomasys.zerogpu.core import Caption
from videomasys.zerogpu.openai import (
    GPT41_MINI_MODEL,
    OPENAI_MODEL,
    OPENAI_SYSTEM_PROMPT,
    OPENAI_URL,
    OpenAI41MiniVLM,
    OpenAIVLM,
    strict_json_schema,
)
from videomasys.zerogpu.remote import HostedVLMError

TOKEN = "sk-fixture-only-never-log"


def completion(model=GPT41_MINI_MODEL):
    return httpx.Response(
        200,
        json={
            "model": model,
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Exact answer"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 2,
                "total_tokens": 22,
                "prompt_tokens_details": {"cached_tokens": 10},
            },
        },
    )


@pytest.fixture
def make_adapter():
    resources = []

    def build(handler, **kwargs):
        client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)
        adapter = OpenAI41MiniVLM(TOKEN, client=client, retry_backoff_seconds=0, **kwargs)
        resources.append((adapter, client))
        return adapter

    yield build
    for adapter, client in resources:
        adapter.close()
        client.close()


def test_pinned_comparator_plain_and_strict_payloads_remain_separate(make_adapter, monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://attacker.invalid")
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
        assert body["model"] == GPT41_MINI_MODEL
        assert body["temperature"] == 0
        assert body["max_completion_tokens"] == tokens
        assert body["store"] is False and body["stream"] is False
        assert body["service_tier"] == "default"
        assert not {"reasoning_effort", "max_tokens", "repetition_penalty", "tools"} & body.keys()
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
        "temperature": 0,
        "max_completion_tokens": 120,
        "store": False,
        "service_tier": "default",
        "stream": False,
    }
    assert adapter.calls[0]["attempts"][0]["usage"]["prompt_tokens_details"] == {
        "cached_tokens": 10,
    }
    assert TOKEN not in json.dumps(adapter.calls)


@pytest.mark.parametrize(
    "model",
    [
        OPENAI_MODEL,
        "gpt-4.1-mini",
        "gpt-4.1-mini-2025-04-15",
        GPT41_MINI_MODEL + "-extra",
        "gpt-4.1-2025-04-14",
    ],
)
def test_config_and_reported_model_require_exact_pinned_snapshot(make_adapter, model):
    with pytest.raises(HostedVLMError) as failure:
        OpenAI41MiniVLM(TOKEN, model=model)
    assert failure.value.code == "request_rejected"
    requests = []
    adapter = make_adapter(lambda request: requests.append(request) or completion(model))
    with pytest.raises(HostedVLMError) as failure:
        adapter("Exact option", [], 100)
    assert failure.value.code == "model_mismatch" and len(requests) == 1
    assert json.loads(requests[0].content)["model"] == GPT41_MINI_MODEL


def test_luna_constructor_still_disallows_comparator_and_keeps_original_default():
    with pytest.raises(HostedVLMError):
        OpenAIVLM(TOKEN, model=GPT41_MINI_MODEL)
    with httpx.Client(
        transport=httpx.MockTransport(lambda request: completion(OPENAI_MODEL))
    ) as client:
        adapter = OpenAIVLM(TOKEN, client=client)
        assert adapter.model == OPENAI_MODEL
        assert adapter._generation_parameters(100)["reasoning_effort"] == "none"
        assert "temperature" not in adapter._generation_parameters(100)
        adapter.close()


@pytest.mark.parametrize("count,expected", [(1, (640, 480)), (18, (512, 384))])
def test_comparator_keeps_bounded_high_detail_images(make_adapter, tmp_path, count, expected):
    frame = tmp_path / "frame.png"
    Image.new("RGB", (1280, 960), "red").save(frame)
    requests = []
    adapter = make_adapter(lambda request: requests.append(request) or completion())
    adapter("Inspect the frames", [frame] * count, 800)
    content = json.loads(requests[0].content)["messages"][1]["content"]
    for item in content[:-1]:
        assert item["image_url"]["detail"] == "high"
        prefix, data = item["image_url"]["url"].split(",", 1)
        assert prefix == "data:image/jpeg;base64"
        with Image.open(io.BytesIO(base64.b64decode(data))) as image:
            assert image.size == expected
    assert adapter.calls[0]["context_token_bound"] < 32_768


def test_comparator_inherits_schema_limits_deadline_and_credit_stop(make_adapter):
    requests, admissions = [], []
    adapter = make_adapter(
        lambda request: (
            requests.append(request)
            or httpx.Response(429, json={"error": {"type": "insufficient_quota", "message": TOKEN}})
        ),
        before_request=lambda: admissions.append(1),
    )
    with pytest.raises(HostedVLMError) as failure:
        adapter.generate_structured("caption", [], 500, {"type": "object"})
    assert failure.value.code == "invalid_request" and not admissions
    with pytest.raises(HostedVLMError) as failure:
        adapter("caption", [], 1601)
    assert failure.value.code == "invalid_request" and not admissions
    with pytest.raises(HostedVLMError) as failure:
        adapter("caption", [], 100)
    assert failure.value.code == "credits" and len(requests) == len(admissions) == 1
    with pytest.raises(HostedVLMError):
        OpenAI41MiniVLM(TOKEN, timeout_seconds=61)


def test_structured_allowance_preserves_request_and_plain_call_limit(make_adapter, tmp_path):
    frame = tmp_path / "frame.jpg"
    Image.new("RGB", (20, 20), "blue").save(frame)
    prompt, paths = "x" * 24_000, [frame] * 18
    schema = Caption.model_json_schema()
    requests, admissions = [], []
    adapter = make_adapter(
        lambda request: requests.append(request) or completion(),
        before_request=lambda: admissions.append(1),
    )
    budget = adapter.request_budget(prompt, len(paths), 1400, schema)
    assert 32_768 < budget["context_bound"] < budget["context_limit"] == 65_536
    assert not requests and not admissions
    adapter.generate_structured(prompt, paths, 1400, schema)
    body = json.loads(requests[0].content)
    assert body["messages"][1]["content"][-1]["text"] == prompt
    assert len(body["messages"][1]["content"]) == 19
    assert body["response_format"]["json_schema"]["schema"] == strict_json_schema(schema)
    assert body["max_completion_tokens"] == 1400
    assert adapter.calls[-1]["context_token_limit"] == 65_536
    with pytest.raises(HostedVLMError) as failure:
        adapter(prompt, paths, 1400)
    assert failure.value.code == "local_request_budget"
    assert adapter.calls[-1]["request_budget"]["context_limit"] == 32_768
    assert len(requests) == len(admissions) == 1


def test_structured_allowance_remains_bounded_with_safe_pretransport_diagnostics(make_adapter):
    requests, admissions = [], []
    adapter = make_adapter(
        lambda request: requests.append(request) or completion(),
        before_request=lambda: admissions.append(1),
    )
    schema = Caption.model_json_schema()
    schema["description"] = "x" * 60_000
    with pytest.raises(HostedVLMError) as failure:
        adapter.generate_structured("x" * 6000, [], 1400, schema)
    assert failure.value.code == "local_request_budget"
    record = adapter.calls[-1]
    assert record["request_budget"]["context_bound"] > 65_536
    assert record["request_budget"]["context_limit"] == 65_536
    assert all(type(value) is int for value in record["request_budget"].values())
    assert not requests and not admissions and not record["attempts"]
    assert TOKEN not in json.dumps(record) and "x" * 20 not in json.dumps(record)


@pytest.mark.parametrize("status", [307, 503])
def test_comparator_redirect_block_and_transient_attempt_ceiling(make_adapter, status):
    requests, admissions = [], []
    adapter = make_adapter(
        lambda request: (
            requests.append(request)
            or httpx.Response(status, headers={"Location": "https://attacker.invalid"})
        ),
        before_request=lambda: admissions.append(1),
    )
    with pytest.raises(HostedVLMError):
        adapter("caption", [], 100)
    assert len(requests) == len(admissions) == (1 if status == 307 else 2)
    assert all(str(request.url) == OPENAI_URL for request in requests)
