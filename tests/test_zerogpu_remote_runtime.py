"""Real runtime/controller tests with explicit fake clients; never paid inference."""

import builtins
import json
import sqlite3
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType
from unittest.mock import Mock

import httpx
import numpy as np
import pytest

from videomasys.zerogpu import core, prompts, remote_runtime
from videomasys.zerogpu.remote_runtime import CPUEmbedder, RemotePipeline
from videomasys.zerogpu.usage import UsageGuard, UsageLimitError


class FixtureVLM:
    """A labeled test-only transport with the real before-request admission contract."""

    def __init__(self, token, *, model, before_request, responses=None, attempts=1, failure=None,
                 image_bounds=None):
        assert token == "test-only-noncredential"
        assert model == remote_runtime.MODEL
        self.before_request = before_request
        self.image_bounds = image_bounds
        self.responses = iter(responses or ["fixture response"] * 10)
        self.attempts = attempts
        self.failure = failure
        self.calls = []
        self.close = Mock()
        self.threads = []

    def __call__(self, _prompt, _paths, _tokens):
        for _attempt in range(self.attempts):
            self.before_request()
            self.threads.append(threading.get_ident())
            self.calls.append({"attempt": len(self.calls) + 1, "fixture": True})
            if self.failure:
                raise self.failure
        return next(self.responses)


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    for name in (
        "HF_INFERENCE_TOKEN",
        "OPENAI_API_KEY",
        "VLM_MODEL",
        "VLM_PROMPT_PROFILE",
        "DEMO_MAX_RUNS",
        "DEMO_MAX_REQUESTS",
        "DEMO_MAX_REQUESTS_PER_RUN",
        "DEMO_USAGE_DB",
        "EMBEDDING_CACHE_DIR",
    ):
        monkeypatch.delenv(name, raising=False)


def make_pipeline(tmp_path, *, client_options=None, guard_options=None):
    guard = UsageGuard(tmp_path / "usage.sqlite3", **(guard_options or {}))
    clients = []

    def factory(token, **kwargs):
        client = FixtureVLM(token, **kwargs, **(client_options or {}))
        clients.append(client)
        return client

    pipeline = RemotePipeline(
        token="test-only-noncredential", guard=guard, embed=Mock(), vlm_factory=factory
    )
    return pipeline, guard, clients


def test_cpu_boot_is_lazy_without_token_adapter_or_embedding_imports(tmp_path, monkeypatch):
    monkeypatch.setenv("DEMO_USAGE_DB", str(tmp_path / "must-not-exist.sqlite3"))
    original_import = builtins.__import__

    def checked_import(name, *args, **kwargs):
        if name in {"fastembed", "videomasys.zerogpu.remote"} or name.split(".")[0] in {
            "torch",
            "transformers",
            "spaces",
            "sentence_transformers",
        }:
            raise AssertionError("CPU boot must not initialize a model/provider")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", checked_import)
    pipeline, index, qa = remote_runtime.load_runtime()
    assert callable(index) and callable(qa)
    assert isinstance(pipeline.embed, CPUEmbedder) and pipeline.embed.encoder is None
    assert pipeline._guard is None
    with pytest.raises(RuntimeError, match="HF_INFERENCE_TOKEN"):
        with pipeline.analysis_session({}):
            pytest.fail("Missing token must never admit an analysis")
    assert not (tmp_path / "must-not-exist.sqlite3").exists()


@pytest.mark.parametrize("token", ["", " ", "\n\t"])
def test_missing_token_does_not_construct_client_or_consume_allowance(tmp_path, token):
    guard = UsageGuard(tmp_path / "usage.sqlite3")
    factory = Mock()
    pipeline = RemotePipeline(token=token, guard=guard, vlm_factory=factory)
    with pytest.raises(RuntimeError, match="HF_INFERENCE_TOKEN"):
        with pipeline.analysis_session({}):
            pytest.fail("No admission expected")
    factory.assert_not_called()
    assert guard.snapshot()["runs_used"] == 0


def test_unlisted_model_rejected_before_admission(tmp_path, monkeypatch):
    monkeypatch.setenv("VLM_MODEL", "unlisted-model:unlisted-provider")
    pipeline, guard, clients = make_pipeline(tmp_path)
    with pytest.raises(RuntimeError, match="Featherless only"):
        with pipeline.analysis_session({}):
            pytest.fail("No admission expected")
    assert not clients
    assert guard.snapshot()["runs_used"] == 0


def test_openai_selection_is_explicit_and_does_not_reuse_hf_secret(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_INFERENCE_TOKEN", "fixture-hf-credential")
    guard = UsageGuard(tmp_path / "usage.sqlite3")
    pipeline = RemotePipeline(provider="openai", guard=guard, vlm_factory=Mock())
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        with pipeline.analysis_session({}):
            pytest.fail("Must not fall back to HF credentials")
    pipeline._factory.assert_not_called()
    assert guard.snapshot()["runs_used"] == 0


def test_hf_default_does_not_automatically_use_openai_key(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-openai-credential")
    guard = UsageGuard(tmp_path / "usage.sqlite3")
    pipeline = RemotePipeline(guard=guard, vlm_factory=Mock())
    with pytest.raises(RuntimeError, match="HF_INFERENCE_TOKEN"):
        with pipeline.analysis_session({}):
            pytest.fail("Adding a key is not permission to change provider")
    assert guard.snapshot()["runs_used"] == 0


def test_openai_rejects_expensive_model_before_admission(tmp_path, monkeypatch):
    monkeypatch.setenv("VLM_MODEL", "gpt-6-astra")
    guard = UsageGuard(tmp_path / "usage.sqlite3")
    pipeline = RemotePipeline(
        provider="openai", token="test-only-noncredential", guard=guard, vlm_factory=Mock(),
    )
    with pytest.raises(RuntimeError, match="explicitly configured models"):
        with pipeline.analysis_session({}):
            pytest.fail("No automatic cost escalation")
    assert guard.snapshot()["runs_used"] == 0
    pipeline._factory.assert_not_called()


def test_openai_runtime_is_lazy_and_uses_separate_cache_identity(monkeypatch):
    original_import = builtins.__import__

    def checked_import(name, *args, **kwargs):
        if name in {"fastembed", "videomasys.zerogpu.openai", "videomasys.zerogpu.remote"}:
            raise AssertionError("Boot must not initialize model or transport")
        if name.split(".")[0] in {"torch", "transformers", "spaces"}:
            raise AssertionError("OpenAI runtime must remain CPU-only")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", checked_import)
    pipeline, _, _ = remote_runtime.load_runtime(provider="openai")
    assert pipeline._guard is None and pipeline.embed.encoder is None
    assert pipeline.provider_label == "OpenAI"
    assert pipeline.model_id == "gpt-5.6-luna@openai-chat-v1-none-strict"
    assert pipeline.signature != RemotePipeline().signature


def test_openai_session_binds_structured_callback_and_counts_repairs(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-openai-credential")
    monkeypatch.setenv("HF_INFERENCE_TOKEN", "fixture-hf-credential")
    guard = UsageGuard(tmp_path / "usage.sqlite3", max_runs=1, max_requests_per_run=3)
    responses = iter(['{"caption":"x"}', '{"caption":"Fixture caption."}'])
    invocations = []
    clients = []

    class OpenAIFixture:
        def __init__(self, token, *, model, before_request):
            assert token == "fixture-openai-credential"
            assert model == remote_runtime.OPENAI_MODEL
            self.before_request = before_request
            self.calls = []
            self.close = Mock()
            clients.append(self)

        def __call__(self, prompt, paths, tokens):
            self.before_request()
            self.calls.append({"fixture": True, "format": "text"})
            return "Fixture choice"

        def generate_structured(self, prompt, paths, tokens, schema):
            self.before_request()
            invocations.append((prompt, paths, tokens, schema))
            self.calls.append({"fixture": True, "format": "json_schema"})
            return next(responses)

    pipeline = RemotePipeline(
        provider="openai", guard=guard, embed=Mock(), vlm_factory=OpenAIFixture,
    )
    bundle = {"trace": []}
    with pipeline.analysis_session(bundle) as active:
        # Validation repair must use the same explicitly bound schema path.
        caption = active.structured(bundle["trace"], "Caption", "Fixture prompt", core.Caption)
        assert caption.caption == "Fixture caption."
        assert len(invocations) == 2
        assert invocations[0][3] == core.Caption.model_json_schema()
        assert active.generate("Fixture select", [], 120) == "Fixture choice"
        assert bundle["inference"]["allowance"]["request_attempts_used"] == 3
        assert bundle["inference"]["provider"] == "OpenAI"
        assert active.signature == pipeline.signature
    clients[0].close.assert_called_once()
    with pytest.raises(UsageLimitError, match="ended"):
        active.generate_structured("fixture", [], 200, core.Caption.model_json_schema())
    assert guard.snapshot()["request_attempts_used"] == 3
    assert "fixture-openai-credential" not in json.dumps(bundle)
    assert "fixture-hf-credential" not in json.dumps(bundle)


def test_openai_real_adapter_strict_caption_and_choice_share_one_allowance(tmp_path):
    from videomasys.zerogpu.openai import OPENAI_URL, OpenAIVLM

    guard = UsageGuard(tmp_path / "usage.sqlite3", max_runs=1, max_requests_per_run=2)
    requests = []

    def handler(request):
        assert str(request.url) == OPENAI_URL
        payload = json.loads(request.content)
        requests.append(payload)
        if len(requests) == 1:
            contract = payload["response_format"]["json_schema"]
            assert contract["strict"] is True
            assert contract["schema"]["additionalProperties"] is False
            assert contract["schema"]["required"] == ["caption"]
            response = '{"caption":"A fixture person carries a fixture box."}'
        else:
            assert "response_format" not in payload
            response = "Carry the box"
        return httpx.Response(200, json={
            "model": remote_runtime.OPENAI_MODEL,
            "choices": [{"message": {"role": "assistant", "content": response},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120,
                      "prompt_tokens_details": {"cached_tokens": 0}},
        })

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        pipeline = RemotePipeline(
            provider="openai", token="test-only-noncredential", guard=guard, embed=Mock(),
            vlm_factory=lambda token, **kwargs: OpenAIVLM(token, client=client, **kwargs),
        )
        bundle = {"trace": []}
        with pipeline.analysis_session(bundle) as active:
            caption = active.structured(bundle["trace"], "Caption", "Fixture", core.Caption)
            assert caption.caption == "A fixture person carries a fixture box."
            assert active.generate("Select exact choice", [], 120) == "Carry the box"
            assert bundle["inference"]["allowance"]["request_attempts_used"] == 2
    assert guard.snapshot()["runs_used"] == 1
    assert len(bundle["inference"]["calls"]) == 2
    assert bundle["inference"]["calls"][0]["requested_provider"] == "OpenAI"
    assert all(request["reasoning_effort"] == "none" for request in requests)
    assert "test-only-noncredential" not in json.dumps(bundle)


def test_explicit_pinned_mini_uses_its_own_adapter_without_luna_fallback(tmp_path, monkeypatch):
    from videomasys.zerogpu import openai

    luna_identity = RemotePipeline(provider="openai").signature
    monkeypatch.setenv("VLM_MODEL", remote_runtime.GPT41_MINI_MODEL)
    guard = UsageGuard(tmp_path / "usage.sqlite3", max_runs=1, max_requests_per_run=1)
    selected = []

    class MiniFixture:
        def __init__(self, token, *, model, before_request):
            assert token == "fixture-mini-credential"
            assert model == remote_runtime.GPT41_MINI_MODEL
            self.before_request = before_request
            self.calls = []
            self.close = Mock()
            selected.append(self)

        def __call__(self, *_args):
            self.before_request()
            raise RuntimeError("Fixture mini error")

    forbidden = Mock(side_effect=AssertionError("No automatic cheaper-model fallback either"))
    monkeypatch.setattr(openai, "OpenAI41MiniVLM", MiniFixture)
    monkeypatch.setattr(openai, "OpenAIVLM", forbidden)
    pipeline = RemotePipeline(provider="openai", token="fixture-mini-credential", guard=guard)
    assert pipeline.signature != luna_identity
    bundle = {}
    with pytest.raises(RuntimeError, match="Fixture mini error"):
        with pipeline.analysis_session(bundle) as active:
            active.generate("Fixture", [], 100)
    forbidden.assert_not_called()
    assert len(selected) == 1
    selected[0].close.assert_called_once()
    assert "pinned API snapshot: 2025-04-14" in bundle["inference"]["weights_revision"]
    assert guard.snapshot()["request_attempts_used"] == 1


@pytest.mark.parametrize("model_name,adapter_name,snapshot", [
    ("GPT54_MINI_MODEL", "OpenAI54MiniVLM", "2026-03-17"),
    ("GPT54_MODEL", "OpenAI54VLM", "2026-03-05"),
])
def test_explicit_54_models_use_pinned_adapter_without_fallback(
    tmp_path, monkeypatch, model_name, adapter_name, snapshot,
):
    from videomasys.zerogpu import openai

    monkeypatch.setenv("VLM_MODEL", remote_runtime.GPT41_MINI_MODEL)
    mini_identity = RemotePipeline(provider="openai").signature
    selected_model = getattr(remote_runtime, model_name)
    monkeypatch.setenv("VLM_MODEL", selected_model)
    guard = UsageGuard(tmp_path / "usage.sqlite3", max_runs=1, max_requests_per_run=1)
    clients = []

    class StrongerFixture:
        def __init__(self, token, *, model, before_request):
            assert token == "fixture-54-credential"
            assert model == selected_model
            self.before_request = before_request
            self.calls = []
            self.close = Mock()
            clients.append(self)

        def __call__(self, *_args):
            self.before_request()
            raise RuntimeError("Fixture 5.4 provider error")

    forbidden = Mock(side_effect=AssertionError("Model failures never change the model"))
    for name in ("OpenAI54MiniVLM", "OpenAI54VLM", "OpenAI41MiniVLM", "OpenAIVLM"):
        monkeypatch.setattr(openai, name, StrongerFixture if name == adapter_name else forbidden)
    pipeline = RemotePipeline(provider="openai", token="fixture-54-credential", guard=guard)
    assert pipeline.signature != mini_identity
    assert pipeline.model_id == selected_model + "@openai-chat-v1-none-strict"
    bundle = {}
    with pytest.raises(RuntimeError, match="Fixture 5.4 provider error"):
        with pipeline.analysis_session(bundle) as active:
            active.generate("Fixture", [], 100)
    forbidden.assert_not_called()
    assert len(clients) == 1
    clients[0].close.assert_called_once()
    assert "pinned API snapshot: " + snapshot in bundle["inference"]["weights_revision"]
    assert bundle["inference"]["model"] == selected_model
    assert guard.snapshot()["request_attempts_used"] == 1


@pytest.mark.parametrize("profile", ["clean", "tuned", "legacy", "svp", "structured"])
def test_prompt_profile_preserved_in_session_and_inference_trace(tmp_path, monkeypatch, profile):
    monkeypatch.setenv("VLM_PROMPT_PROFILE", profile)
    pipeline, guard, clients = make_pipeline(tmp_path)
    bundle = {}
    with pipeline.analysis_session(bundle) as active:
        assert active.prompt_profile == pipeline.prompt_profile == profile
        assert active.prompt_version == pipeline.prompt_version
        assert active.signature == pipeline.signature
        assert bundle["inference"]["prompt_profile"] == profile
        assert bundle["inference"]["prompt_version"] == active.prompt_version
    assert guard.snapshot()["request_attempts_used"] == 0
    assert clients[0].image_bounds == ((512, 384) if profile == "structured" else None)


def test_prompt_profile_separates_caches_and_rejects_unknown_before_usage(tmp_path, monkeypatch):
    monkeypatch.setenv("VLM_PROMPT_PROFILE", "legacy")
    legacy = RemotePipeline()
    clean = RemotePipeline(prompt_profile="clean")
    assert legacy.prompt_profile == "legacy" and clean.prompt_profile == "clean"
    assert legacy.signature != clean.signature
    monkeypatch.setenv("VLM_PROMPT_PROFILE", "untrusted arbitrary prompt text")
    ledger = tmp_path / "must-not-exist.sqlite3"
    monkeypatch.setenv("DEMO_USAGE_DB", str(ledger))
    with pytest.raises(ValueError):
        RemotePipeline()
    assert not ledger.exists()


def test_invalid_usage_configuration_error_does_not_echo_value(tmp_path, monkeypatch):
    monkeypatch.setenv("DEMO_USAGE_DB", str(tmp_path / "usage.sqlite3"))
    monkeypatch.setenv("DEMO_MAX_RUNS", "test-only-sensitive-config-value")
    pipeline = RemotePipeline(token="test-only-noncredential", vlm_factory=Mock())
    with pytest.raises(RuntimeError, match="misconfigured") as error:
        with pipeline.analysis_session({}):
            pytest.fail("No admission expected")
    assert "sensitive-config" not in str(error.value)


def test_session_clone_preserves_identity_embeddings_cache_and_current_accounting(tmp_path):
    pipeline, guard, clients = make_pipeline(tmp_path)
    bundle = {}
    with pipeline.analysis_session(bundle) as active:
        assert isinstance(active, core.LivePipeline) and active is not pipeline
        assert active.signature == pipeline.signature
        assert active.model_id == pipeline.model_id
        assert active.model_id.endswith("@hosted-api-v2-rp108")
        assert active.embedding_id == pipeline.embedding_id == remote_runtime.EMBEDDING_ID
        assert active.embed is pipeline.embed
        assert "fastembed-0.8.0-onnx-v1" in active.signature
        assert active.generate("fixture", [], 10) == "fixture response"
        # Counts are updated before the next Gradio result, not only at cleanup.
        assert bundle["inference"]["allowance"]["request_attempts_used"] == 1
        assert len(bundle["inference"]["calls"]) == 1
        assert "test-only-noncredential" not in json.dumps(bundle)
    clients[0].close.assert_called_once()
    with pytest.raises(UsageLimitError, match="ended"):
        active.generate("fixture", [], 10)
    assert guard.snapshot()["request_attempts_used"] == 1


def test_unbound_generate_never_uses_transport(tmp_path):
    pipeline, guard, clients = make_pipeline(tmp_path)
    with pytest.raises(RuntimeError, match="admitted analysis session"):
        pipeline.generate("fixture", [], 10)
    assert not clients
    assert guard.snapshot()["runs_used"] == 0


def test_each_transport_retry_reserves_before_submission(tmp_path):
    pipeline, guard, clients = make_pipeline(
        tmp_path, client_options={"attempts": 2}, guard_options={"max_requests_per_run": 1}
    )
    bundle = {}
    with pytest.raises(UsageLimitError, match="Analysis request limit"):
        with pipeline.analysis_session(bundle) as active:
            active.generate("fixture", [], 10)
    assert len(clients[0].calls) == 1  # Rejected retry never reaches fake transport.
    assert guard.snapshot()["request_attempts_used"] == 1
    assert len(bundle["inference"]["calls"]) == 1
    clients[0].close.assert_called_once()


@pytest.mark.parametrize("request_limit", [1, 2])
def test_empty_response_recovery_uses_real_shared_accounting(tmp_path, request_limit):
    from videomasys.zerogpu.remote import HostedVLM, HostedVLMError

    guard = UsageGuard(tmp_path / "usage.sqlite3", max_runs=1,
                       max_requests_per_run=request_limit)
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={
            "choices": [{"message": {"role": "assistant",
                                      "content": "" if len(requests) == 1 else "{}"},
                         "finish_reason": "stop"}],
        })

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        pipeline = RemotePipeline(
            token="test-only-noncredential", guard=guard, embed=Mock(),
            vlm_factory=lambda token, **kwargs: HostedVLM(
                token, client=client, retry_backoff_seconds=0, **kwargs,
            ),
        )
        bundle = {}
        if request_limit == 1:
            with pytest.raises(HostedVLMError) as error:
                with pipeline.analysis_session(bundle) as active:
                    active.generate("fixture", [], 10)
            assert error.value.code == "admission_denied"
        else:
            with pipeline.analysis_session(bundle) as active:
                assert active.generate("fixture", [], 10) == "{}"
    assert len(requests) == request_limit
    assert guard.snapshot()["runs_used"] == 1
    assert guard.snapshot()["request_attempts_used"] == request_limit
    attempts = bundle["inference"]["calls"][0]["attempts"]
    assert attempts[0]["response_issue"] == "empty_completion"
    assert attempts[1]["request_attempted"] == (request_limit == 2)


def test_structured_model_repair_also_consumes_attempt_allowance(tmp_path):
    valid = json.dumps(prompts.DEMONSTRATIONS["local_helmet_attribute"]["plan"])
    pipeline, guard, clients = make_pipeline(
        tmp_path, client_options={"responses": ["not JSON fixture", valid]}
    )
    bundle = {"trace": []}
    with pipeline.analysis_session(bundle) as active:
        active.structured(bundle["trace"], "Fixture planner", "fixture prompt", core.QAPlan)
    assert guard.snapshot()["request_attempts_used"] == 2
    assert len(bundle["trace"][0]["attempts"]) == 2
    assert bundle["trace"][0]["attempts"][0]["raw_response"] == "not JSON fixture"
    clients[0].close.assert_called_once()


def test_provider_failure_is_counted_and_client_disposed(tmp_path):
    pipeline, guard, clients = make_pipeline(
        tmp_path, client_options={"failure": RuntimeError("fixture provider failure")}
    )
    bundle = {}
    with pytest.raises(RuntimeError, match="fixture provider failure"):
        with pipeline.analysis_session(bundle) as active:
            active.generate("fixture", [], 10)
    assert guard.snapshot()["runs_used"] == 1
    assert guard.snapshot()["request_attempts_used"] == 1
    assert bundle["inference"]["allowance"]["request_attempts_used"] == 1
    clients[0].close.assert_called_once()


def test_constructor_failure_consumes_admission_but_no_request(tmp_path):
    guard = UsageGuard(tmp_path / "usage.sqlite3")
    factory = Mock(side_effect=RuntimeError("fixture construction failure"))
    pipeline = RemotePipeline(token="test-only-noncredential", guard=guard, vlm_factory=factory)
    with pytest.raises(RuntimeError, match="fixture construction failure"):
        with pipeline.analysis_session({}):
            pytest.fail("Constructor failed")
    assert guard.snapshot()["runs_used"] == 1
    assert guard.snapshot()["request_attempts_used"] == 0


def test_summary_read_failure_does_not_mask_provider_error_or_skip_disposal(tmp_path, monkeypatch):
    pipeline, guard, clients = make_pipeline(
        tmp_path, client_options={"failure": RuntimeError("fixture provider failure")}
    )
    bundle = {}
    with pytest.raises(RuntimeError, match="fixture provider failure"):
        with pipeline.analysis_session(bundle) as active:
            monkeypatch.setattr(
                guard,
                "snapshot",
                Mock(side_effect=sqlite3.OperationalError("fixture read failure")),
            )
            active.generate("fixture", [], 10)
    assert bundle["inference"]["allowance"] == {"status": "Usage summary unavailable"}
    clients[0].close.assert_called_once()


def test_distinct_sessions_never_share_the_per_run_allowance(tmp_path):
    pipeline, guard, clients = make_pipeline(tmp_path, guard_options={"max_requests_per_run": 1})
    with pipeline.analysis_session({}) as first, pipeline.analysis_session({}) as second:
        first.generate("fixture", [], 10)
        with pytest.raises(UsageLimitError, match="Analysis request limit"):
            first.generate("fixture", [], 10)
        second.generate("fixture", [], 10)
    assert guard.snapshot()["runs_used"] == 2
    assert guard.snapshot()["request_attempts_used"] == 2
    assert all(client.close.call_count == 1 for client in clients)


def test_real_qa_generator_can_advance_and_close_on_different_worker_threads(tmp_path):
    pair = prompts.DEMONSTRATIONS["local_helmet_attribute"]
    pipeline, guard, clients = make_pipeline(
        tmp_path,
        client_options={"responses": [json.dumps(pair["plan"]), json.dumps(pair["program"])]},
    )
    bundle = {
        "trace": [],
        "scenes": [{"index": 0, "start": 0.0, "end": 20.0, "caption": "Fixture helmet"}],
    }

    def staged():
        with pipeline.analysis_session(bundle) as active:
            iterator = active.qa(bundle, pair["question"], [])
            try:
                yield next(iterator)
                yield next(iterator)
            finally:
                iterator.close()  # Never advance into retrieval in this fixture.

    iterator = staged()
    with ThreadPoolExecutor(max_workers=1) as first, ThreadPoolExecutor(max_workers=1) as second:
        assert first.submit(next, iterator).result()["trace"][-1]["stage"] == "Planner"
        assert second.submit(next, iterator).result()["trace"][-1]["stage"] == "Tool agent"
        second.submit(iterator.close).result()
    assert len(set(clients[0].threads)) == 2
    assert guard.snapshot()["request_attempts_used"] == 2
    clients[0].close.assert_called_once()


def fake_embedding_module(monkeypatch, vectors):
    module = ModuleType("fastembed")
    encoder = Mock()
    encoder.embed.side_effect = lambda _texts: iter(vectors)
    module.TextEmbedding = Mock(return_value=encoder)
    monkeypatch.setitem(sys.modules, "fastembed", module)
    return module, encoder


def test_empty_embedding_batch_is_lazy_and_has_expected_shape(monkeypatch):
    module, _encoder = fake_embedding_module(monkeypatch, [])
    embed = CPUEmbedder()
    assert embed([]).shape == (0, 384)
    module.TextEmbedding.assert_not_called()


def test_cpu_embeddings_are_normalized_and_initialized_only_once(monkeypatch):
    module, encoder = fake_embedding_module(monkeypatch, [np.arange(1, 385)])
    embed = CPUEmbedder()
    with ThreadPoolExecutor(max_workers=4) as pool:
        vectors = list(pool.map(lambda _i: embed(["fixture"]), range(8)))
    module.TextEmbedding.assert_called_once()
    config = module.TextEmbedding.call_args.kwargs
    assert config["providers"] == ["CPUExecutionProvider"] and config["threads"] == 2
    assert config["model_name"] == remote_runtime.EMBEDDING_MODEL
    assert encoder.embed.call_count == 8
    assert all(value.shape == (1, 384) and value.dtype == np.float32 for value in vectors)
    assert all(np.allclose(np.linalg.norm(value, axis=1), 1.0) for value in vectors)


@pytest.mark.parametrize(
    "vectors",
    [np.zeros((1, 384)), np.ones((1, 12)), np.full((1, 384), np.nan), np.full((1, 384), np.inf)],
)
def test_invalid_embeddings_never_enter_cache(monkeypatch, vectors):
    fake_embedding_module(monkeypatch, vectors)
    with pytest.raises(RuntimeError, match="invalid embeddings"):
        CPUEmbedder()(["fixture"])


def test_embedding_load_failure_is_safe_and_retryable(monkeypatch):
    module, _encoder = fake_embedding_module(monkeypatch, [])
    module.TextEmbedding.side_effect = RuntimeError("test-only-sensitive-provider-detail")
    embed = CPUEmbedder()
    with pytest.raises(RuntimeError, match="CPU search model is unavailable") as error:
        embed(["fixture"])
    assert "sensitive-provider" not in str(error.value)
    assert embed.encoder is None


@pytest.mark.parametrize("texts", [["x"] * 65, ["x" * 20001]])
def test_oversized_embedding_input_fails_before_loading(monkeypatch, texts):
    module, _encoder = fake_embedding_module(monkeypatch, [])
    with pytest.raises(RuntimeError, match="input exceeds"):
        CPUEmbedder()(texts)
    module.TextEmbedding.assert_not_called()
