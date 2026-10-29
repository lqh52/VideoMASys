"""Mock-transport security and accounting tests; never invoke an inference provider."""

import base64
import contextvars
import io
import json
import threading
import time
import traceback

import httpx
import pytest
from PIL import Image

from videomasys.zerogpu import remote
from videomasys.zerogpu.remote import DEFAULT_MODEL, ROUTER_URL, HostedVLM, HostedVLMError

TOKEN = "hf_test_credential_do_not_log"
PROMPT = "private prompt do not include in diagnostic errors"


def completion(text="Unwrap it", **fields):
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
            "model": DEFAULT_MODEL.split(":")[0],
            **fields,
        },
    )


@pytest.fixture
def make_adapter():
    resources = []

    def build(handler, **kwargs):
        client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)
        adapter = HostedVLM(TOKEN, client=client, retry_backoff_seconds=0, **kwargs)
        resources.append((adapter, client))
        return adapter

    yield build
    for adapter, client in resources:
        adapter.close()
        client.close()


def test_fixed_endpoint_auth_greedy_plaintext_and_provider_usage(make_adapter, monkeypatch):
    monkeypatch.setenv("HF_ENDPOINT", "https://untrusted.example")
    requests, admissions = [], []

    def handler(request):
        requests.append(request)
        return completion(
            "  Unwrap it\n",
            id="chatcmpl-fixture",
            provider="featherless-ai",
            usage={"prompt_tokens": 43, "completion_tokens": 3, "total_tokens": 46},
        )

    adapter = make_adapter(handler, before_request=lambda: admissions.append(1))
    assert adapter(PROMPT, [], 120) == "  Unwrap it\n"
    request = requests[0]
    assert str(request.url) == ROUTER_URL
    assert request.headers["authorization"] == "Bearer " + TOKEN
    body = json.loads(request.content)
    assert body["model"] == DEFAULT_MODEL
    assert body["temperature"] == 0 and body["stream"] is False and body["max_tokens"] == 120
    assert body["repetition_penalty"] == 1.08
    assert "response_format" not in body and "tools" not in body
    assert body["messages"] == [
        {"role": "system", "content": remote.SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "text", "text": PROMPT}]},
    ]
    assert len(admissions) == 1
    attempt = adapter.calls[0]["attempts"][0]
    assert attempt["usage"] == {"prompt_tokens": 43, "completion_tokens": 3, "total_tokens": 46}
    assert "may omit image tokens" in attempt["usage_note"]
    assert attempt["reported_provider"] == "featherless-ai"
    assert attempt["reported_model"] == DEFAULT_MODEL.split(":")[0]
    assert attempt["http_status"] == 200 and attempt["latency_ms"] >= 0
    assert TOKEN not in json.dumps(adapter.calls) and PROMPT not in json.dumps(adapter.calls)
    assert adapter.calls[0]["generation_parameters"] == {
        "temperature": 0,
        "repetition_penalty": 1.08,
        "max_tokens": 120,
        "stream": False,
    }


@pytest.mark.parametrize("count,expected", [(1, (640, 480)), (6, (640, 480)), (18, (512, 384))])
def test_local_images_are_validated_downscaled_and_precede_text(
    make_adapter, tmp_path, count, expected
):
    image_path = tmp_path / "frame.png"
    Image.new("RGB", (1280, 960), "red").save(image_path)
    requests = []
    adapter = make_adapter(lambda request: requests.append(request) or completion())
    assert adapter("A legitimate image prompt " * 150, [image_path] * count, 800) == "Unwrap it"
    body = json.loads(requests[0].content)
    content = body["messages"][1]["content"]
    assert len(content) == count + 1 and content[-1]["type"] == "text"
    for item in content[:-1]:
        assert item["type"] == "image_url"
        prefix, encoded = item["image_url"]["url"].split(",", 1)
        assert prefix == "data:image/jpeg;base64"
        with Image.open(io.BytesIO(base64.b64decode(encoded))) as decoded:
            assert decoded.size == expected and decoded.mode == "RGB"
    assert str(image_path) not in requests[0].content.decode()
    assert adapter.calls[0]["context_token_bound"] <= remote.MAX_CONTEXT_BOUND
    assert adapter.calls[0]["image_encoding"] == "jpeg_quality90"


@pytest.mark.parametrize("status", [301, 302, 307, 308])
def test_redirects_never_forward_bearer_credentials(make_adapter, status):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(status, headers={"Location": "https://attacker.example/" + TOKEN})

    adapter = make_adapter(handler)
    with pytest.raises(HostedVLMError) as failure:
        adapter(PROMPT, [], 100)
    assert failure.value.code == "redirect_blocked"
    assert len(requests) == 1 and str(requests[0].url) == ROUTER_URL
    assert TOKEN not in str(failure.value) and "attacker" not in str(failure.value)


@pytest.mark.parametrize(
    "status,code",
    [
        (401, "authentication"),
        (403, "authentication"),
        (402, "credits"),
        (400, "request_rejected"),
        (404, "request_rejected"),
        (500, "unavailable"),
    ],
)
def test_user_errors_never_include_http_body_prompt_or_token(make_adapter, status, code):
    requests = []
    adapter = make_adapter(
        lambda request: (
            requests.append(request)
            or httpx.Response(status, text=TOKEN + PROMPT + " secret provider diagnostics")
        )
    )
    with pytest.raises(HostedVLMError) as failure:
        adapter(PROMPT, [], 100)
    rendered = "".join(traceback.format_exception(failure.value))
    assert failure.value.code == code and len(requests) == 1
    assert TOKEN not in rendered and PROMPT not in rendered
    assert "secret provider diagnostics" not in rendered
    assert TOKEN not in json.dumps(adapter.calls)


@pytest.mark.parametrize("status", [429, 502, 503, 504])
def test_transient_status_retries_once_with_separate_admission(make_adapter, status):
    requests, admissions = [], []

    def handler(request):
        requests.append(request)
        return (
            httpx.Response(status, headers={"Retry-After": "0"})
            if len(requests) == 1
            else completion()
        )

    adapter = make_adapter(handler, before_request=lambda: admissions.append(1))
    assert adapter(PROMPT, [], 100) == "Unwrap it"
    assert len(requests) == len(admissions) == len(adapter.calls[0]["attempts"]) == 2
    assert adapter.calls[0]["attempts"][0]["retry_delay_seconds"] == 0


@pytest.mark.parametrize("error", [httpx.ConnectError, httpx.ConnectTimeout])
def test_connect_failure_retries_but_does_not_log_raw_exception(make_adapter, error):
    requests = []

    def handler(request):
        requests.append(request)
        if len(requests) == 1:
            raise error(TOKEN + PROMPT, request=request)
        return completion()

    adapter = make_adapter(handler)
    assert adapter(PROMPT, [], 100) == "Unwrap it"
    assert len(requests) == 2 and TOKEN not in json.dumps(adapter.calls)


@pytest.mark.parametrize(
    "error", [httpx.ReadTimeout, httpx.WriteTimeout, httpx.RemoteProtocolError]
)
def test_ambiguous_network_failure_is_never_retried(make_adapter, error):
    requests = []

    def handler(request):
        requests.append(request)
        raise error(TOKEN + PROMPT, request=request)

    adapter = make_adapter(handler)
    with pytest.raises(HostedVLMError) as failure:
        adapter(PROMPT, [], 100)
    assert len(requests) == 1
    assert TOKEN not in "".join(traceback.format_exception(failure.value))


def test_two_attempt_ceiling_and_provider_long_retry_after_are_respected(make_adapter):
    requests = []
    adapter = make_adapter(lambda request: requests.append(request) or httpx.Response(503))
    with pytest.raises(HostedVLMError):
        adapter(PROMPT, [], 100)
    assert len(requests) == 2
    requests.clear()
    adapter = make_adapter(
        lambda request: (
            requests.append(request) or httpx.Response(429, headers={"Retry-After": "3600"})
        )
    )
    with pytest.raises(HostedVLMError) as failure:
        adapter(PROMPT, [], 100)
    assert failure.value.code == "rate_limited" and len(requests) == 1


def test_admission_denial_before_retry_prevents_another_http_attempt(make_adapter):
    requests, admissions = [], []

    def admit():
        admissions.append(1)
        if len(admissions) == 2:
            raise RuntimeError(TOKEN + " internal allowance details")

    adapter = make_adapter(
        lambda request: requests.append(request) or httpx.Response(503), before_request=admit
    )
    with pytest.raises(HostedVLMError) as failure:
        adapter(PROMPT, [], 100)
    assert failure.value.code == "admission_denied"
    assert len(admissions) == 2 and len(requests) == 1
    assert not adapter.calls[0]["attempts"][1]["request_attempted"]
    assert TOKEN not in str(failure.value)


@pytest.mark.parametrize(
    "body",
    [
        None,
        [],
        {},
        {"choices": []},
        {"choices": [{"message": {"content": ""}}]},
        {"choices": [{"message": {"content": {"answer": "not text"}}}]},
    ],
)
def test_malformed_completion_is_not_retried_or_substituted(make_adapter, body):
    requests = []
    adapter = make_adapter(
        lambda request: requests.append(request) or httpx.Response(200, json=body)
    )
    with pytest.raises(HostedVLMError) as failure:
        adapter(PROMPT, [], 100)
    assert failure.value.code == "invalid_response" and len(requests) == 1


def test_unexpected_reported_model_is_rejected_without_fallback(make_adapter):
    requests = []
    adapter = make_adapter(
        lambda request: requests.append(request) or completion(model="Qwen/Other-Bigger-Model")
    )
    with pytest.raises(HostedVLMError) as failure:
        adapter(PROMPT, [], 100)
    assert failure.value.code == "model_mismatch" and len(requests) == 1


def test_metadata_is_allowlisted_and_never_copies_credentials(make_adapter):
    adapter = make_adapter(
        lambda request: completion(
            provider=TOKEN,
            id="unsafe " + PROMPT,
            usage={
                "prompt_tokens": -1,
                "completion_tokens": True,
                "total_tokens": 50,
                "secret": TOKEN,
            },
        )
    )
    adapter(PROMPT, [], 100)
    attempt = adapter.calls[0]["attempts"][0]
    assert attempt["usage"] == {"total_tokens": 50}
    assert "reported_provider" not in attempt and "request_id" not in attempt
    assert TOKEN not in json.dumps(adapter.calls)


@pytest.mark.parametrize(
    "prompt,paths,tokens,code",
    [
        ("", [], 100, "invalid_request"),
        ("x" * 32001, [], 100, "invalid_request"),
        ("界" * 12000, [], 100, "local_request_budget"),
        ("x", [], 0, "invalid_request"),
        ("x", [], 1601, "invalid_request"),
        ("x", [], True, "invalid_request"),
        ("x", "not a list", 100, "invalid_request"),
        ("x", ["missing"] * 19, 100, "invalid_request"),
    ],
    ids=["empty", "characters", "context_bytes", "zero_output", "excess_output",
         "boolean_output", "invalid_paths", "excess_images"],
)
def test_request_limits_fail_before_usage_admission_or_http(
    make_adapter, prompt, paths, tokens, code,
):
    requests, admissions = [], []
    adapter = make_adapter(
        lambda request: requests.append(request) or completion(),
        before_request=lambda: admissions.append(1),
    )
    with pytest.raises(HostedVLMError) as failure:
        adapter(prompt, paths, tokens)
    assert failure.value.code == code
    assert not requests and not admissions


def test_invalid_and_oversized_images_fail_before_http(make_adapter, tmp_path, monkeypatch):
    requests = []
    adapter = make_adapter(lambda request: requests.append(request) or completion())
    path = tmp_path / "invalid.png"
    path.write_bytes(b"not an image")
    with pytest.raises(HostedVLMError, match="invalid image"):
        adapter("Inspect", [path], 100)
    Image.new("RGB", (20, 20)).save(path)
    monkeypatch.setattr(remote, "MAX_IMAGE_PIXELS", 100)
    with pytest.raises(HostedVLMError, match="invalid image"):
        adapter("Inspect", [path], 100)
    monkeypatch.setattr(remote, "MAX_IMAGE_PIXELS", 1000)
    monkeypatch.setattr(remote, "MAX_IMAGE_BYTES", 5)
    with pytest.raises(HostedVLMError, match="invalid image"):
        adapter("Inspect", [path], 100)
    assert not requests


def test_actual_multiframe_image_is_rejected_before_http(make_adapter, tmp_path):
    path = tmp_path / "animated.png"
    with Image.new("RGB", (20, 20), "red") as first:
        with Image.new("RGB", (20, 20), "blue") as second:
            first.save(path, save_all=True, append_images=[second], duration=100, loop=0)
    with Image.open(path) as source:
        assert source.n_frames == 2
    requests = []
    adapter = make_adapter(lambda request: requests.append(request) or completion())
    with pytest.raises(HostedVLMError) as failure:
        adapter("Inspect", [path], 100)
    assert failure.value.code == "invalid_request" and not requests


def test_large_response_is_rejected_without_echoing_body(make_adapter, monkeypatch):
    monkeypatch.setattr(remote, "MAX_RESPONSE_BYTES", 20)
    adapter = make_adapter(lambda request: completion(TOKEN * 20))
    with pytest.raises(HostedVLMError) as failure:
        adapter(PROMPT, [], 100)
    assert failure.value.code == "invalid_response" and TOKEN not in str(failure.value)


def test_total_deadline_does_not_retry_or_free_an_inflight_worker(make_adapter):
    entered, release = threading.Event(), threading.Event()
    requests = []

    def handler(request):
        requests.append(request)
        entered.set()
        release.wait(1)
        return completion()

    adapter = make_adapter(handler, timeout_seconds=0.03)
    started = time.monotonic()
    try:
        with pytest.raises(HostedVLMError) as failure:
            adapter(PROMPT, [], 100)
        assert entered.is_set() and failure.value.code == "timeout"
        assert time.monotonic() - started < 0.3
        with pytest.raises(HostedVLMError) as busy:
            adapter(PROMPT, [], 100)
        assert busy.value.code == "busy" and len(requests) == 1
        assert adapter.calls[0]["caller_deadline_exceeded"] is True
    finally:
        release.set()
        assert adapter._worker_slot.acquire(timeout=1)
        adapter._worker_slot.release()


def test_slow_admission_cannot_send_after_the_total_deadline(make_adapter):
    entered, release = threading.Event(), threading.Event()
    requests = []

    def admit():
        entered.set()
        release.wait(1)

    adapter = make_adapter(
        lambda request: requests.append(request) or completion(),
        before_request=admit,
        timeout_seconds=0.03,
    )
    try:
        with pytest.raises(HostedVLMError) as failure:
            adapter(PROMPT, [], 100)
        assert entered.is_set() and failure.value.code == "timeout"
    finally:
        release.set()
        assert adapter._worker_slot.acquire(timeout=1)
        adapter._worker_slot.release()
    assert not requests


def test_adapter_reuse_is_allowed_only_after_a_timed_out_worker_finishes(make_adapter):
    release = threading.Event()
    requests = []

    def handler(request):
        requests.append(request)
        if len(requests) == 1:
            release.wait(1)
        return completion()

    adapter = make_adapter(handler, timeout_seconds=0.05)
    try:
        with pytest.raises(HostedVLMError) as failure:
            adapter(PROMPT, [], 100)
        assert failure.value.code == "timeout"
    finally:
        release.set()
        assert adapter._worker_slot.acquire(timeout=1)
        adapter._worker_slot.release()
    assert adapter(PROMPT, [], 100) == "Unwrap it"
    assert len(requests) == 2
    assert adapter.calls[0]["status"] == "failed"
    assert adapter.calls[0]["error_code"] == "timeout"
    assert adapter.calls[1]["status"] == "complete"


def test_context_admission_is_propagated_and_sequential_calls_are_ready(make_adapter):
    run_id = contextvars.ContextVar("run_id")
    run_id.set("isolated-run")
    observed = []
    adapter = make_adapter(
        lambda request: completion(), before_request=lambda: observed.append(run_id.get())
    )
    for _ in range(10):
        assert adapter(PROMPT, [], 100) == "Unwrap it"
    assert observed == ["isolated-run"] * 10


def test_close_does_not_close_injected_client_and_disallows_new_calls(make_adapter):
    adapter = make_adapter(lambda request: completion())
    adapter.close()
    assert not adapter._client.is_closed
    with pytest.raises(HostedVLMError) as failure:
        adapter(PROMPT, [], 100)
    assert failure.value.code == "closed"


def test_owned_client_disables_environment_proxies_and_closes_without_network():
    adapter = HostedVLM(TOKEN)
    assert adapter._client._trust_env is False
    adapter.close()
    assert adapter._client.is_closed


@pytest.mark.parametrize("token", ["", "secret\nheader", "space token", None])
def test_invalid_credentials_are_rejected_without_echoing_them(token):
    with pytest.raises(HostedVLMError) as failure:
        HostedVLM(token)
    assert failure.value.code == "authentication"


def test_model_allowlist_rejects_unapproved_or_costlier_selection():
    with pytest.raises(HostedVLMError) as failure:
        HostedVLM(TOKEN, model="Qwen/Costlier-Model:other-provider")
    assert failure.value.code == "request_rejected"


def assistant_envelope(content="", finish="stop", message_fields=None, **fields):
    return {
        "id": "chatcmpl-diagnostic-fixture",
        "model": DEFAULT_MODEL.split(":")[0],
        "choices": [
            {
                "message": {"role": "assistant", "content": content, **(message_fields or {})},
                "finish_reason": finish,
            }
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 0, "total_tokens": 12},
        **fields,
    }


@pytest.mark.parametrize("null_error", [False, True])
def test_empty_assistant_stop_gets_one_admitted_retry_with_metadata(make_adapter, null_error):
    requests, admissions = [], []
    body = assistant_envelope(error=None) if null_error else assistant_envelope()

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=body) if len(requests) == 1 else completion("Actual reply")

    adapter = make_adapter(handler, before_request=lambda: admissions.append(1))
    assert adapter(PROMPT, [], 100) == "Actual reply"
    assert len(requests) == len(admissions) == 2
    first = adapter.calls[0]["attempts"][0]
    assert first["response_issue"] == "empty_completion"
    assert first["empty_completion_retry_eligible"] is True
    assert first["content_type"] == "string" and first["content_length"] == 0
    assert first["finish_reason"] == "stop" and first["message_role"] == "assistant"
    assert first["usage"] == body["usage"] and first["request_id"] == body["id"]
    assert first["envelope_type"] == "object" and first["choice_count"] == 1


def test_two_empty_completions_exhaust_existing_two_attempt_limit(make_adapter):
    requests = []
    adapter = make_adapter(
        lambda request: requests.append(request) or httpx.Response(200, json=assistant_envelope())
    )
    with pytest.raises(HostedVLMError) as failure:
        adapter(PROMPT, [], 100)
    assert failure.value.code == "invalid_response" and len(requests) == 2
    assert all(a["response_issue"] == "empty_completion" for a in adapter.calls[0]["attempts"])


def test_transient_status_then_empty_response_does_not_create_a_third_attempt(make_adapter):
    requests = []

    def handler(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(503)
        return httpx.Response(200, json=assistant_envelope())

    adapter = make_adapter(handler)
    with pytest.raises(HostedVLMError):
        adapter(PROMPT, [], 100)
    assert len(requests) == 2


@pytest.mark.parametrize(
    "body,issue",
    [
        (assistant_envelope(finish="length"), "length_limit"),
        (assistant_envelope(finish="unexpected-private-value"), "unknown_finish_reason"),
        (assistant_envelope(finish=None), "empty_completion"),
        (assistant_envelope(message_fields={"role": None}), "empty_completion"),
        (assistant_envelope(content=None), "invalid_content_type"),
        (assistant_envelope(content=[{"type": "text", "text": "answer"}]), "invalid_content_type"),
        (assistant_envelope(message_fields={"reasoning_content": TOKEN}), "reasoning_only"),
        (assistant_envelope(content=None, message_fields={"reasoning": TOKEN}), "reasoning_only"),
        (assistant_envelope(message_fields={"refusal": TOKEN}), "refusal"),
        (assistant_envelope(finish="content_filter"), "refusal"),
        (assistant_envelope(message_fields={"tool_calls": [{"arguments": TOKEN}]}), "tool_calls"),
        (assistant_envelope(finish="tool_calls"), "tool_calls"),
        (assistant_envelope(error={"message": TOKEN}), "provider_error"),
        (assistant_envelope(error={}), "provider_error"),
        (assistant_envelope(message_fields={"reasoning": {}}), "empty_completion"),
        (assistant_envelope(message_fields={"tool_calls": ""}), "empty_completion"),
    ],
)
def test_only_known_empty_assistant_stop_shape_is_retryable(make_adapter, body, issue, caplog):
    requests = []
    adapter = make_adapter(
        lambda request: requests.append(request) or httpx.Response(200, json=body)
    )
    with pytest.raises(HostedVLMError):
        adapter(PROMPT, [], 100)
    assert len(requests) == 1
    attempt = adapter.calls[0]["attempts"][0]
    assert attempt["response_issue"] == issue
    assert attempt["usage"] == body["usage"]
    assert not attempt.get("empty_completion_retry_eligible")
    assert TOKEN not in caplog.text and PROMPT not in caplog.text
    assert TOKEN not in json.dumps(adapter.calls)
    assert "unexpected-private-value" not in caplog.text


def test_valid_completion_with_null_error_is_accepted_unchanged(make_adapter):
    text = ' {"answer":"Actual response"} '
    adapter = make_adapter(
        lambda request: httpx.Response(200, json=assistant_envelope(text, error=None))
    )
    assert adapter(PROMPT, [], 100) == text


def test_nonempty_length_completion_remains_available_to_core_for_json_repair(make_adapter):
    text = '{"answer":"This actual model output was cut'
    requests = []
    adapter = make_adapter(
        lambda request: (
            requests.append(request)
            or httpx.Response(200, json=assistant_envelope(text, finish="length"))
        )
    )
    assert adapter(PROMPT, [], 100) == text
    assert len(requests) == 1
    assert adapter.calls[0]["attempts"][0]["response_issue"] == "length_limit"


def test_reasoning_is_not_used_as_answer_but_actual_final_text_is_preserved(make_adapter):
    adapter = make_adapter(
        lambda request: httpx.Response(
            200, json=assistant_envelope("Actual final text", message_fields={"reasoning": TOKEN})
        )
    )
    assert adapter(PROMPT, [], 100) == "Actual final text"
    attempt = adapter.calls[0]["attempts"][0]
    assert attempt["has_reasoning"] is True and TOKEN not in json.dumps(attempt)


def test_bad_http_json_has_safe_type_size_diagnostics_and_no_retry(make_adapter, caplog):
    requests = []
    adapter = make_adapter(
        lambda request: (
            requests.append(request)
            or httpx.Response(
                200, text=TOKEN + PROMPT, headers={"Content-Type": "text/html; extra=" + TOKEN}
            )
        )
    )
    with pytest.raises(HostedVLMError):
        adapter(PROMPT, [], 100)
    attempt = adapter.calls[0]["attempts"][0]
    assert len(requests) == 1 and attempt["response_issue"] == "invalid_json"
    assert attempt["response_content_type"] == "text/html"
    assert attempt["response_bytes"] == len((TOKEN + PROMPT).encode())
    assert TOKEN not in caplog.text and PROMPT not in caplog.text


def test_empty_response_retry_still_respects_admission_denial(make_adapter):
    requests, admissions = [], []

    def admit():
        admissions.append(1)
        if len(admissions) == 2:
            raise RuntimeError(TOKEN)

    adapter = make_adapter(
        lambda request: requests.append(request) or httpx.Response(200, json=assistant_envelope()),
        before_request=admit,
    )
    with pytest.raises(HostedVLMError) as failure:
        adapter(PROMPT, [], 100)
    assert failure.value.code == "admission_denied"
    assert len(requests) == 1 and len(admissions) == 2


def test_model_mismatch_empty_response_is_never_retried(make_adapter, caplog):
    requests = []
    adapter = make_adapter(
        lambda request: (
            requests.append(request) or httpx.Response(200, json=assistant_envelope(model=TOKEN))
        )
    )
    with pytest.raises(HostedVLMError) as failure:
        adapter(PROMPT, [], 100)
    assert failure.value.code == "model_mismatch" and len(requests) == 1
    assert TOKEN not in caplog.text and TOKEN not in json.dumps(adapter.calls)


@pytest.mark.parametrize(
    "field,value",
    [
        ("audio", {"data": TOKEN}),
        ("reasoning_details", [{"text": TOKEN}]),
        ("annotations", [{"text": TOKEN}]),
        (TOKEN, "private unknown value"),
        ("unknown_extension", []),
        ("unknown_extension", {}),
        ("unknown_extension", ""),
        ("unknown_extension", False),
        ("unknown_extension", 0),
    ],
)
def test_unknown_nonnull_message_fields_block_only_empty_retry(make_adapter, field, value, caplog):
    requests = []
    body = assistant_envelope(message_fields={field: value})
    adapter = make_adapter(
        lambda request: requests.append(request) or httpx.Response(200, json=body)
    )
    with pytest.raises(HostedVLMError) as error:
        adapter(PROMPT, [], 100)
    assert error.value.code == "invalid_response"
    assert len(requests) == 1
    attempt = adapter.calls[0]["attempts"][0]
    assert attempt["has_unknown_nonnull_fields"] is True
    assert attempt["empty_completion_retry_eligible"] is False
    assert TOKEN not in json.dumps(adapter.calls) and TOKEN not in caplog.text
    assert "private unknown value" not in json.dumps(adapter.calls)
    assert "private unknown value" not in caplog.text


def test_unknown_null_metadata_does_not_block_empty_retry(make_adapter):
    requests = []

    def handler(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(200, json=assistant_envelope(message_fields={"extension": None}))
        return completion("Actual reply")

    adapter = make_adapter(handler)
    assert adapter(PROMPT, [], 100) == "Actual reply"
    assert len(requests) == 2
    assert adapter.calls[0]["attempts"][0]["has_unknown_nonnull_fields"] is False


def test_nonempty_text_with_unknown_message_fields_is_preserved_unchanged(make_adapter):
    text = ' {"answer":"Actual provider text"} '
    adapter = make_adapter(
        lambda request: httpx.Response(
            200,
            json=assistant_envelope(text, message_fields={"reasoning_details": [{"text": TOKEN}]}),
        )
    )
    assert adapter(PROMPT, [], 100) == text
    assert TOKEN not in json.dumps(adapter.calls)
