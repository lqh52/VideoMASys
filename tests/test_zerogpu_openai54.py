"""Full GPT-5.4 snapshot contract; mock HTTP only, never paid inference."""

import base64
import io
import json
import math
from copy import deepcopy

import httpx
import pytest
from PIL import Image

from videomasys.domain.visual_evidence import StructuredAnswer
from videomasys.zerogpu.openai import (
    GPT41_MINI_MODEL,
    GPT54_MINI_MODEL,
    GPT54_MODEL,
    OPENAI_MODEL,
    OPENAI_URL,
    OpenAI41MiniVLM,
    OpenAI54MiniVLM,
    OpenAI54VLM,
    OpenAIVLM,
    strict_json_schema,
)
from videomasys.zerogpu.remote import HostedVLMError

TOKEN = "sk-test-only-not-a-credential"


def completion(model=GPT54_MODEL):
    return httpx.Response(200, json={
        "model": model,
        "choices": [{"message": {"role": "assistant", "content": "{}"},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 30, "completion_tokens": 2, "total_tokens": 32,
                  "completion_tokens_details": {"reasoning_tokens": 0}},
    })


@pytest.fixture
def make_adapter():
    resources = []

    def build(handler, **kwargs):
        client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)
        adapter = OpenAI54VLM(TOKEN, client=client, retry_backoff_seconds=0, **kwargs)
        resources.append((adapter, client))
        return adapter

    yield build
    for adapter, client in resources:
        adapter.close()
        client.close()


def test_full_snapshot_none_and_strict_evidence_schema_are_preserved(make_adapter, monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://attacker.invalid")
    monkeypatch.setenv("VLM_MODEL", GPT54_MINI_MODEL)
    requests = []
    adapter = make_adapter(lambda request: requests.append(request) or completion())
    schema = StructuredAnswer.model_json_schema()
    original = deepcopy(schema)
    assert adapter.generate_structured("Preserve evidence", [], 1400, schema) == "{}"
    body = json.loads(requests[0].content)
    assert str(requests[0].url) == OPENAI_URL
    assert body["model"] == GPT54_MODEL == "gpt-5.4-2026-03-05"
    assert body["reasoning_effort"] == "none" and body["temperature"] == 0
    assert body["max_completion_tokens"] == 1400
    assert body["store"] is False and body["stream"] is False
    assert body["service_tier"] == "default"
    assert not {"tools", "max_tokens", "repetition_penalty"} & body.keys()
    sent = body["response_format"]["json_schema"]
    assert sent["strict"] is True and sent["schema"] == strict_json_schema(original)
    assert sent["schema"]["properties"]["claims"]["maxItems"] == 5
    assert sent["schema"]["properties"]["answer_claim_ids"]["maxItems"] == 3
    assert schema == original
    record = adapter.calls[0]
    assert record["context_token_limit"] == 65_536
    assert record["attempts"][0]["usage"]["completion_tokens_details"]["reasoning_tokens"] == 0
    assert TOKEN not in json.dumps(adapter.calls)


@pytest.mark.parametrize("model", ["gpt-5.4", GPT54_MINI_MODEL, GPT54_MODEL + "-extra"])
def test_full_snapshot_rejects_config_or_reported_model_drift(make_adapter, model):
    with pytest.raises(HostedVLMError) as failure:
        OpenAI54VLM(TOKEN, model=model)
    assert failure.value.code == "request_rejected"
    requests = []
    adapter = make_adapter(lambda request: requests.append(request) or completion(model))
    with pytest.raises(HostedVLMError) as failure:
        adapter("Exact choice", [], 100)
    assert failure.value.code == "model_mismatch" and len(requests) == 1


@pytest.mark.parametrize("adapter_class,model", [
    (OpenAIVLM, OPENAI_MODEL), (OpenAI41MiniVLM, GPT41_MINI_MODEL),
    (OpenAI54MiniVLM, GPT54_MINI_MODEL),
])
def test_existing_adapters_keep_defaults_and_cannot_silently_use_full_54(adapter_class, model):
    with pytest.raises(HostedVLMError):
        adapter_class(TOKEN, model=GPT54_MODEL)
    with httpx.Client(transport=httpx.MockTransport(lambda request: completion(model))) as client:
        adapter = adapter_class(TOKEN, client=client)
        assert adapter.model == model
        adapter.close()


@pytest.mark.parametrize("count,bounds,expected", [
    (1, None, (640, 480)), (18, (512, 384), (512, 384)),
])
def test_image_evidence_and_schema_budget_are_bounded_without_dropping_frames(
    make_adapter, tmp_path, count, bounds, expected,
):
    frame = tmp_path / "frame.png"
    Image.new("RGB", (1280, 960), "blue").save(frame)
    requests, admissions = [], []
    adapter = make_adapter(
        lambda request: requests.append(request) or completion(), image_bounds=bounds,
        before_request=lambda: admissions.append(1),
    )
    prompt = "x" * 24_000 if count == 18 else "Inspect these frames"
    paths, schema = [frame] * count, StructuredAnswer.model_json_schema()
    adapter.generate_structured(prompt, paths, 1400, schema)
    content = json.loads(requests[0].content)["messages"][1]["content"]
    assert len(content) == count + 1 and content[-1]["text"] == prompt
    for item in content[:-1]:
        assert item["image_url"]["detail"] == "high"
        data = item["image_url"]["url"].removeprefix("data:image/jpeg;base64,")
        with Image.open(io.BytesIO(base64.b64decode(data))) as encoded:
            assert encoded.size == expected
            # Official 32px patches at 1.2x fit the unchanged 600/frame app allowance.
            patches = math.ceil(encoded.width / 32) * math.ceil(encoded.height / 32)
            assert math.ceil(patches * 1.2) <= 360 < 600
    if count == 18:
        call = adapter.calls[-1]
        assert 32_768 < call["context_token_bound"] < call["context_token_limit"] == 65_536
        with pytest.raises(HostedVLMError) as failure:
            adapter(prompt, paths, 1400)
        assert failure.value.code == "local_request_budget"
        assert adapter.calls[-1]["request_budget"]["context_limit"] == 32_768
    assert len(requests) == len(admissions) == 1


def test_bad_schema_evidence_output_and_context_fail_before_admission(make_adapter):
    requests, admissions = [], []
    adapter = make_adapter(
        lambda request: requests.append(request) or completion(),
        before_request=lambda: admissions.append(1),
    )
    schema = StructuredAnswer.model_json_schema()
    for prompt, paths, tokens, contract in (
        ("Inspect", [], 100, {"type": "object"}),
        ("Inspect", [], 1601, schema),
        ("Inspect", ["missing"] * 19, 100, schema),
        ("Inspect", ["missing"], 100, schema),
        ("x" * 32_001, [], 100, schema),
    ):
        with pytest.raises(HostedVLMError) as failure:
            adapter.generate_structured(prompt, paths, tokens, contract)
        assert failure.value.code == "invalid_request"
    large = {"type": "object", "properties": {}, "description": "x" * 60_000}
    with pytest.raises(HostedVLMError) as failure:
        adapter.generate_structured("x" * 6000, [], 1400, large)
    assert failure.value.code == "local_request_budget"
    assert adapter.calls[-1]["request_budget"]["context_bound"] > 65_536
    assert not requests and not admissions


@pytest.mark.parametrize("status,attempts", [(307, 1), (429, 1), (503, 2)])
def test_transport_guards_keep_endpoint_model_attempt_bound_and_credit_stop(
    make_adapter, status, attempts,
):
    requests, admissions = [], []
    adapter = make_adapter(
        lambda request: requests.append(request) or httpx.Response(
            status, headers={"Location": "https://attacker.invalid"},
            json={"error": {"code": "insufficient_quota", "message": TOKEN}},
        ),
        before_request=lambda: admissions.append(1),
    )
    with pytest.raises(HostedVLMError) as failure:
        adapter("Exact choice", [], 100)
    if status == 429:
        assert failure.value.code == "credits"
    assert len(requests) == len(admissions) == attempts
    assert all(str(request.url) == OPENAI_URL for request in requests)
    assert all(json.loads(request.content)["model"] == GPT54_MODEL for request in requests)
    assert TOKEN not in str(failure.value) + json.dumps(adapter.calls)
