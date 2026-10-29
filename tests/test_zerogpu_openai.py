"""Direct OpenAI transport/schema contract tests. All HTTP uses MockTransport."""

import base64
import copy
import io
import json
import threading
import time
from typing import Literal

import httpx
import pytest
from PIL import Image
from pydantic import Field, create_model

from videomasys.domain.schemas import ToolProgram
from videomasys.zerogpu.core import (
    Answer,
    AnswerText,
    Caption,
    ChoiceComparison,
    EvidenceAnswer,
    Inspection,
    Observation,
    QAPlan,
    Verdict,
)
from videomasys.zerogpu.openai import (
    OPENAI_MODEL,
    OPENAI_SYSTEM_PROMPT,
    OPENAI_URL,
    OpenAIVLM,
    strict_json_schema,
)
from videomasys.zerogpu.remote import DEFAULT_MODEL, HostedVLMError

TOKEN = "sk-proj-fixture-secret-never-log"
PROMPT = "private visual question"


def completion(text="The actual answer", *, model=OPENAI_MODEL, **fields):
    return httpx.Response(
        200,
        json={
            "model": model,
            "choices": [
                {"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
            ],
            **fields,
        },
    )


@pytest.fixture
def make_adapter():
    resources = []

    def build(handler, **kwargs):
        client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)
        adapter = OpenAIVLM(TOKEN, client=client, retry_backoff_seconds=0, **kwargs)
        resources.append((adapter, client))
        return adapter

    yield build
    for adapter, client in resources:
        adapter.close()
        client.close()


def test_plain_generation_uses_fixed_openai_route_and_exact_parameters(make_adapter, monkeypatch):
    for key in ("OPENAI_BASE_URL", "HF_ENDPOINT", "HTTPS_PROXY"):
        monkeypatch.setenv(key, "https://attacker.invalid")
    requests, admissions = [], []
    adapter = make_adapter(
        lambda request: requests.append(request) or completion("  Exact choice\n"),
        before_request=lambda: admissions.append(1),
    )
    assert adapter(PROMPT, [], 120) == "  Exact choice\n"
    request = requests[0]
    assert str(request.url) == OPENAI_URL
    assert request.headers["Authorization"] == "Bearer " + TOKEN
    assert json.loads(request.content) == {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": OPENAI_SYSTEM_PROMPT},
            {"role": "user", "content": [{"type": "text", "text": PROMPT}]},
        ],
        "reasoning_effort": "none",
        "max_completion_tokens": 120,
        "store": False,
        "service_tier": "default",
        "stream": False,
    }
    record = adapter.calls[0]
    assert record["requested_provider"] == "OpenAI" and len(admissions) == 1
    assert record["generation_parameters"] == {
        "reasoning_effort": "none",
        "max_completion_tokens": 120,
        "store": False,
        "service_tier": "default",
        "stream": False,
    }
    assert TOKEN not in json.dumps(record) and PROMPT not in json.dumps(record)
    assert "Return only the requested JSON" not in OPENAI_SYSTEM_PROMPT


@pytest.mark.parametrize("count,size", [(1, (640, 480)), (6, (640, 480)), (18, (512, 384))])
def test_images_keep_bounded_jpeg_resizing_and_explicit_high_detail(
    make_adapter, tmp_path, count, size
):
    frame = tmp_path / "frame.png"
    Image.new("RGB", (1280, 960), "red").save(frame)
    requests = []
    adapter = make_adapter(lambda request: requests.append(request) or completion())
    adapter.generate("caption " * 700, [frame] * count, 800)
    content = json.loads(requests[0].content)["messages"][1]["content"]
    assert content[-1]["type"] == "text" and len(content) == count + 1
    for item in content[:-1]:
        assert item["image_url"]["detail"] == "high"
        prefix, data = item["image_url"]["url"].split(",", 1)
        assert prefix == "data:image/jpeg;base64"
        with Image.open(io.BytesIO(base64.b64decode(data))) as image:
            assert image.size == size
    assert adapter.calls[0]["image_detail"] == "high"
    assert str(frame) not in requests[0].content.decode()


def test_structured_format_is_explicit_per_call_and_plaintext_never_inherits_it(make_adapter):
    schema = Caption.model_json_schema()
    original = copy.deepcopy(schema)
    requests = []
    adapter = make_adapter(
        lambda request: requests.append(request) or completion('{"caption":"Real"}')
    )
    assert adapter.generate_structured(PROMPT, [], 500, schema) == '{"caption":"Real"}'
    adapter.generate("Choose one exact option", [], 100)
    body, plain = [json.loads(request.content) for request in requests]
    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "video_evidence",
            "strict": True,
            "schema": strict_json_schema(schema),
        },
    }
    assert "response_format" not in plain
    assert schema == original
    assert adapter.calls[0]["response_format"] == "json_schema_strict"
    assert "response_format" not in adapter.calls[1]
    assert "schema" not in json.dumps(adapter.calls[0]["generation_parameters"])


def schema_models():
    comparison = create_model(
        "FixtureComparison", __base__=ChoiceComparison, choice=(Literal["One", "Two"], ...)
    )
    grounded = create_model(
        "FixtureGrounded",
        __base__=AnswerText,
        selected_choice=(Literal["One"], Field(description="Exact candidate")),
        option_comparisons=(list[comparison], Field(min_length=2, max_length=2)),
        supporting_scenes=(list[Literal[0, 3, 5]], Field(min_length=1, max_length=3)),
    )
    opened = create_model(
        "FixtureOpen",
        __base__=EvidenceAnswer,
        supporting_scenes=(list[Literal[0, 3, 5]], Field(min_length=1, max_length=3)),
    )
    observation = create_model("FixtureObservation", __base__=Observation, scene=(Literal[3], ...))
    return [
        QAPlan,
        ToolProgram,
        Caption,
        Observation,
        Inspection,
        AnswerText,
        EvidenceAnswer,
        Answer,
        Verdict,
        ChoiceComparison,
        comparison,
        grounded,
        opened,
        observation,
    ]


def assert_constraints_preserved(original, normalized):
    assert "default" not in normalized
    for key, value in original.items():
        if key in {"default", "required", "additionalProperties"}:
            continue
        if key in {"properties", "$defs", "definitions"}:
            assert list(value) == list(normalized[key])
            for name in value:
                assert_constraints_preserved(value[name], normalized[key][name])
        elif key == "items":
            assert_constraints_preserved(value, normalized[key])
        elif key == "anyOf":
            for old, new in zip(value, normalized[key], strict=True):
                assert_constraints_preserved(old, new)
        else:
            assert normalized[key] == value
    if original.get("type") == "object":
        assert normalized["additionalProperties"] is False
        assert normalized["required"] == list(normalized["properties"])


@pytest.mark.parametrize("model", schema_models(), ids=lambda model: model.__name__)
def test_all_active_static_and_dynamic_schemas_preserve_constraints(model, make_adapter):
    schema = model.model_json_schema()
    original = copy.deepcopy(schema)
    normalized = strict_json_schema(schema)
    assert_constraints_preserved(schema, normalized)
    assert original == schema
    requests = []
    adapter = make_adapter(lambda request: requests.append(request) or completion("{}"))
    adapter.generate_structured(PROMPT, [], 800, schema)
    assert json.loads(requests[0].content)["response_format"]["json_schema"]["schema"] == normalized


def test_nullable_default_refs_recursive_schema_and_property_named_default_are_preserved():
    schema = {
        "type": "object",
        "properties": {
            "default": {"type": "string", "default": "not a guaranteed value"},
            "nullable": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
            "child": {"anyOf": [{"$ref": "#"}, {"type": "null"}]},
        },
    }
    normalized = strict_json_schema(schema)
    assert normalized["required"] == ["default", "nullable", "child"]
    assert normalized["properties"]["default"] == {"type": "string"}
    assert normalized["properties"]["nullable"]["anyOf"][-1] == {"type": "null"}
    assert normalized["properties"]["child"]["anyOf"][0] == {"$ref": "#"}


@pytest.mark.parametrize(
    "schema",
    [
        None,
        [],
        {},
        {"type": "string"},
        {"type": "object"},
        {"type": "object", "additionalProperties": True},
        {"type": "object", "additionalProperties": {"type": "string"}},
        {"type": "object", "properties": {"bad": {}}},
        {"type": "object", "properties": {"bad": {"type": "object"}}},
        {"type": "object", "properties": {"bad": {"type": "array", "items": {}}}},
        {"type": "object", "properties": {"bad": {"$ref": "https://attacker.invalid/schema"}}},
        {"type": "object", "properties": {"bad": {"$ref": "#/$defs/absent"}}},
        {"type": "object", "properties": {}, "anyOf": [{"type": "object"}]},
        {"type": "object", "properties": {}, "allOf": []},
        {"type": "object", "properties": {}, "patternProperties": {}},
        {"type": "object", "properties": {}, "description": "x" * 64_001},
    ],
)
def test_unsupported_or_free_form_schema_fails_before_admission(make_adapter, schema):
    requests, admissions = [], []
    adapter = make_adapter(
        lambda request: requests.append(request) or completion(),
        before_request=lambda: admissions.append(1),
    )
    with pytest.raises(HostedVLMError) as failure:
        adapter.generate_structured(PROMPT, [], 500, schema)
    assert failure.value.code == "invalid_request"
    assert not requests and not admissions


@pytest.mark.parametrize("model", [DEFAULT_MODEL, "gpt-5.6-sol", OPENAI_MODEL + "-other", ""])
def test_only_requested_luna_alias_is_configurable(model):
    with pytest.raises(HostedVLMError, match="configured model"):
        OpenAIVLM(TOKEN, model=model)


@pytest.mark.parametrize(
    "model,accepted",
    [
        (OPENAI_MODEL, True),
        (OPENAI_MODEL + "-2026-09-07", True),  # Synthetic response shape, not a claimed release.
        (OPENAI_MODEL + "-2026-02-31", False),
        (OPENAI_MODEL + "-2026-9-7", False),
        (OPENAI_MODEL + "-2026-09-07-extra", False),
        (OPENAI_MODEL + "-costlier", False),
        ("gpt-5.6-sol", False),
        (None, False),
    ],
)
def test_response_model_is_exact_alias_or_valid_same_model_date(make_adapter, model, accepted):
    requests = []
    adapter = make_adapter(lambda request: requests.append(request) or completion(model=model))
    if accepted:
        assert adapter(PROMPT, [], 100) == "The actual answer"
    else:
        with pytest.raises(HostedVLMError) as failure:
            adapter(PROMPT, [], 100)
        assert failure.value.code == "model_mismatch"
    assert len(requests) == 1


def test_usage_details_allowlist_caches_reasoning_and_rejects_arbitrary_values(make_adapter):
    adapter = make_adapter(
        lambda request: completion(
            usage={
                "prompt_tokens": 100,
                "completion_tokens": 10,
                "total_tokens": 110,
                "prompt_tokens_details": {
                    "cached_tokens": 20,
                    "cache_write_tokens": 40,
                    "image_tokens": 30,
                    "text_tokens": 70,
                    "secret": TOKEN,
                },
                "completion_tokens_details": {
                    "reasoning_tokens": 0,
                    "text_tokens": 10,
                    "secret": TOKEN,
                },
                "secret": TOKEN,
            }
        )
    )
    adapter(PROMPT, [], 100)
    usage = adapter.calls[0]["attempts"][0]["usage"]
    assert usage["prompt_tokens_details"] == {
        "cached_tokens": 20,
        "cache_write_tokens": 40,
        "image_tokens": 30,
        "text_tokens": 70,
    }
    assert usage["completion_tokens_details"] == {"reasoning_tokens": 0, "text_tokens": 10}
    assert TOKEN not in json.dumps(adapter.calls)
    assert "OpenAI-reported" in adapter.calls[0]["attempts"][0]["usage_note"]


@pytest.mark.parametrize("status", [301, 302, 307, 308, 401, 403, 402, 400, 500])
def test_errors_and_redirects_never_forward_or_disclose_credentials(make_adapter, status):
    requests = []
    adapter = make_adapter(
        lambda request: (
            requests.append(request)
            or httpx.Response(
                status, headers={"Location": "https://attacker.invalid"}, text=TOKEN + PROMPT
            )
        )
    )
    with pytest.raises(HostedVLMError) as failure:
        adapter(PROMPT, [], 100)
    assert len(requests) == 1 and str(requests[0].url) == OPENAI_URL
    assert TOKEN not in str(failure.value) + json.dumps(adapter.calls)
    assert PROMPT not in str(failure.value) + json.dumps(adapter.calls)


@pytest.mark.parametrize("first", [429, 502, 503, 504, "empty", "connect"])
def test_allowed_retry_has_same_payload_and_fresh_admission(make_adapter, first):
    requests, admissions = [], []

    def handler(request):
        requests.append(request)
        if len(requests) == 1:
            if first == "connect":
                raise httpx.ConnectError(TOKEN)
            return completion("") if first == "empty" else httpx.Response(first)
        return completion()

    adapter = make_adapter(handler, before_request=lambda: admissions.append(1))
    assert adapter.generate_structured(PROMPT, [], 100, Caption.model_json_schema())
    assert len(requests) == len(admissions) == 2
    assert requests[0].content == requests[1].content


def test_two_attempt_ceiling_and_guard_denial_apply_to_structured_calls(make_adapter):
    requests = []
    adapter = make_adapter(lambda request: requests.append(request) or completion(""))
    with pytest.raises(HostedVLMError):
        adapter.generate_structured(PROMPT, [], 100, Caption.model_json_schema())
    assert len(requests) == 2
    admissions = []

    def admit():
        admissions.append(1)
        if len(admissions) == 2:
            raise RuntimeError(TOKEN)

    requests.clear()
    adapter = make_adapter(
        lambda request: requests.append(request) or httpx.Response(503), before_request=admit
    )
    with pytest.raises(HostedVLMError) as failure:
        adapter.generate_structured(PROMPT, [], 100, Caption.model_json_schema())
    assert failure.value.code == "admission_denied" and len(requests) == 1


@pytest.mark.parametrize(
    "error", [httpx.ReadTimeout, httpx.WriteTimeout, httpx.RemoteProtocolError]
)
def test_ambiguous_failures_are_not_retried(make_adapter, error):
    requests = []

    def handler(request):
        requests.append(request)
        raise error(TOKEN)

    adapter = make_adapter(handler)
    with pytest.raises(HostedVLMError):
        adapter(PROMPT, [], 100)
    assert len(requests) == 1


def test_hard_deadline_bounds_caller_and_keeps_timed_out_worker_busy(make_adapter):
    entered, release = threading.Event(), threading.Event()

    def handler(request):
        entered.set()
        release.wait(2)
        return completion()

    adapter = make_adapter(handler, timeout_seconds=0.03)
    started = time.monotonic()
    try:
        with pytest.raises(HostedVLMError) as failure:
            adapter.generate_structured(PROMPT, [], 100, Caption.model_json_schema())
        assert entered.is_set() and failure.value.code == "timeout"
        assert time.monotonic() - started < 0.5
        with pytest.raises(HostedVLMError) as failure:
            adapter(PROMPT, [], 100)
        assert failure.value.code == "busy"
        assert len(adapter.calls[0]["attempts"]) == 1
    finally:
        release.set()
        assert adapter._worker_slot.acquire(timeout=1)
        adapter._worker_slot.release()


@pytest.mark.parametrize("timeout", [61, 120, True, 0, float("inf")])
def test_openai_cannot_raise_wall_clock_cap_above_sixty_seconds(timeout):
    with pytest.raises(HostedVLMError):
        OpenAIVLM(TOKEN, timeout_seconds=timeout)


@pytest.mark.parametrize(
    "prompt,paths,tokens",
    [
        ("x" * 32_001, [], 100),
        (PROMPT, [], 1601),
        (PROMPT, ["missing"] * 19, 100),
        (PROMPT, ["missing"], 100),
        (PROMPT, [], True),
    ],
)
def test_request_limits_apply_before_admission(make_adapter, prompt, paths, tokens):
    requests, admissions = [], []
    adapter = make_adapter(
        lambda request: requests.append(request) or completion(),
        before_request=lambda: admissions.append(1),
    )
    with pytest.raises(HostedVLMError):
        adapter.generate_structured(prompt, paths, tokens, Caption.model_json_schema())
    assert not requests and not admissions


@pytest.mark.parametrize(
    "error",
    [
        {"type": "insufficient_quota"},
        {"code": "insufficient_quota"},
        {"code": "credit_balance_exhausted"},
        {"code": "organization_spend_limit_exceeded"},
        {"code": "project_spend_limit_exceeded"},
        {"code": "organization_usage_limit_exceeded"},
    ],
)
def test_billing_quota_errors_do_not_retry_or_misreport_rate_limits(make_adapter, error):
    requests, admissions = [], []
    adapter = make_adapter(
        lambda request: (
            requests.append(request)
            or httpx.Response(429, json={"error": {**error, "message": TOKEN + PROMPT}})
        ),
        before_request=lambda: admissions.append(1),
    )
    with pytest.raises(HostedVLMError) as failure:
        adapter(PROMPT, [], 100)
    assert failure.value.code == "credits" and len(requests) == len(admissions) == 1
    assert TOKEN not in json.dumps(adapter.calls) + str(failure.value)
    assert PROMPT not in json.dumps(adapter.calls) + str(failure.value)
    attempt = adapter.calls[0]["attempts"][0]
    assert attempt["error_response_issue"] == "none"
    assert "retry_delay_seconds" not in attempt


@pytest.mark.parametrize(
    "code,param",
    [
        ("invalid_json_schema", "response_format.json_schema.schema"),
        ("unsupported_parameter", "reasoning_effort"),
        ("model_not_found", "model"),
    ],
)
def test_request_errors_preserve_only_fixed_diagnostic_fields(make_adapter, caplog, code, param):
    adapter = make_adapter(
        lambda request: httpx.Response(
            400,
            json={
                "error": {
                    "code": code,
                    "type": "invalid_request_error",
                    "param": param,
                    "message": TOKEN + PROMPT,
                    "private": TOKEN,
                }
            },
        )
    )
    with pytest.raises(HostedVLMError) as failure:
        adapter(PROMPT, [], 100)
    attempt = adapter.calls[0]["attempts"][0]
    assert failure.value.code == "request_rejected"
    assert attempt["provider_error_code"] == code
    assert attempt["provider_error_type"] == "invalid_request_error"
    assert attempt["provider_error_param"] == param
    rendered = json.dumps(adapter.calls) + str(failure.value) + caplog.text
    assert TOKEN not in rendered and PROMPT not in rendered


def test_unknown_error_fields_are_redacted_without_echo_or_retry(make_adapter, caplog):
    adapter = make_adapter(
        lambda request: httpx.Response(
            400, json={"error": {"code": TOKEN, "type": PROMPT, "param": TOKEN, "message": PROMPT}}
        )
    )
    with pytest.raises(HostedVLMError):
        adapter(PROMPT, [], 100)
    attempt = adapter.calls[0]["attempts"][0]
    assert [attempt["provider_error_" + field] for field in ("code", "type", "param")] == [
        "other",
        "other",
        "other",
    ]
    assert TOKEN not in json.dumps(adapter.calls) + caplog.text
    assert PROMPT not in json.dumps(adapter.calls) + caplog.text


@pytest.mark.parametrize(
    "body,issue",
    [
        (b"not json", "invalid_json"),
        (b"[]", "invalid_envelope"),
        (b'{"error":null}', "invalid_envelope"),
        (b"x" * 16_385, "body_too_large"),
    ],
)
def test_malformed_error_body_is_bounded_and_keeps_status_classification(make_adapter, body, issue):
    requests = []
    adapter = make_adapter(
        lambda request: requests.append(request) or httpx.Response(400, content=body)
    )
    with pytest.raises(HostedVLMError) as failure:
        adapter(PROMPT, [], 100)
    assert failure.value.code == "request_rejected" and len(requests) == 1
    assert adapter.calls[0]["attempts"][0]["error_response_issue"] == issue


def test_ordinary_documented_rate_limit_still_retries_once(make_adapter):
    requests = []

    def handler(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(429, json={"error": {"code": "rate_limit_exceeded"}})
        return completion()

    adapter = make_adapter(handler)
    assert adapter(PROMPT, [], 100) == "The actual answer" and len(requests) == 2


def test_usage_details_reject_booleans_negative_and_nonfinite_values(make_adapter):
    adapter = make_adapter(
        lambda request: completion(
            usage={
                "prompt_tokens_details": {
                    "cached_tokens": True,
                    "cache_write_tokens": -1,
                    "image_tokens": 1_000_000_001,
                    "text_tokens": "100",
                },
                "completion_tokens_details": {"reasoning_tokens": TOKEN},
            }
        )
    )
    adapter(PROMPT, [], 100)
    usage = adapter.calls[0]["attempts"][0]["usage"]
    assert usage["prompt_tokens_details"] == {} and usage["completion_tokens_details"] == {}


def test_schema_request_budget_includes_schema_size_before_admission(make_adapter):
    requests, admissions = [], []
    schema = Caption.model_json_schema()
    schema["description"] = "x" * 30_000
    adapter = make_adapter(
        lambda request: requests.append(request) or completion(),
        before_request=lambda: admissions.append(1),
    )
    with pytest.raises(HostedVLMError):
        adapter.generate_structured("x" * 4000, [], 800, schema)
    assert not requests and not admissions
