"""Offline runner tests: fixtures cannot reach an inference provider."""

import importlib.util
import json
import os
from contextlib import contextmanager
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/evaluate_openai_prompts.py"
spec = importlib.util.spec_from_file_location("evaluate_openai_prompts_fixture", SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_explicit_run_flag_required_without_reading_key(monkeypatch):
    monkeypatch.setattr(runner, "read_owner_key", lambda: pytest.fail("must not read credentials"))
    with pytest.raises(SystemExit) as failure:
        runner.main(["--model", runner.GPT54_MINI_MODEL, "--case", "sofa_mc",
                     "--output-root", "unused"])
    assert failure.value.code == 2


def test_dotenv_only_extracts_key_without_interpolation_or_environment_mutation(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(runner, "REPO", tmp_path)
    key = "sk-fixture-${UNCHANGED}-secret"
    path = tmp_path / ".env"
    path.write_text(f"OPENAI_API_KEY={key}\nVLM_MODEL=expensive\nDEMO_MAX_RUNS=9999\n")
    path.chmod(0o600)
    monkeypatch.setenv("VLM_MODEL", "ambient-model")
    monkeypatch.setenv("UNCHANGED", "must-not-expand")
    assert runner.read_owner_key() == key
    assert os.environ["VLM_MODEL"] == "ambient-model"
    assert "DEMO_MAX_RUNS" not in os.environ or os.environ["DEMO_MAX_RUNS"] != "9999"


def test_owner_secret_permissions_and_missing_key_fail_safely(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "REPO", tmp_path)
    path = tmp_path / ".env"
    path.write_text("UNRELATED=fixture-not-a-key\n")
    path.chmod(0o644)
    with pytest.raises(runner.EvaluationError, match="owner-only"):
        runner.read_owner_key()
    path.chmod(0o600)
    with pytest.raises(runner.EvaluationError, match="OPENAI_API_KEY"):
        runner.read_owner_key()


def test_selected_model_restores_ambient_even_on_failure(monkeypatch):
    monkeypatch.setenv("VLM_MODEL", "ambient-expensive")
    with pytest.raises(RuntimeError):
        with runner.selected_model(runner.GPT54_MINI_MODEL):
            assert os.environ["VLM_MODEL"] == runner.GPT54_MINI_MODEL
            raise RuntimeError("fixture")
    assert os.environ["VLM_MODEL"] == "ambient-expensive"


@pytest.fixture
def isolated_runner(tmp_path, monkeypatch):
    key = "sk-fixture-only-secret"
    monkeypatch.setattr(runner, "read_owner_key", lambda: key)
    monkeypatch.setattr(runner, "source_hashes", lambda: {"source.py": "fixture-source-hash"})
    monkeypatch.setattr(runner, "CPUEmbedder", lambda: "isolated-embedder")
    monkeypatch.setattr(runner, "case_input", lambda case: {
        "video": tmp_path / "video.mp4", "question": runner.KITCHEN_QUESTION,
        "options": [] if case == "kitchen_open" else ["Unwrap it", "Other"],
    })
    monkeypatch.setattr(runner, "prepare_video", lambda path, cache: {
        "trace": [], "scenes": [{"index": 1, "frames": [str(cache / "sample.jpg")]}],
        "cached": False, "database": str(cache / "index.sqlite"),
    })
    calls = []

    class Pipeline:
        signature = "isolated-clean-signature"

        def __init__(self, **kwargs):
            assert kwargs["provider"] == "openai" and kwargs["token"] == key
            assert kwargs["prompt_profile"] == "clean"
            assert kwargs["embed"] == "isolated-embedder"
            self.guard = kwargs["guard"]
            self.model = os.environ["VLM_MODEL"]
            assert self.model in runner.MODELS
            calls.append(self)

        @contextmanager
        def analysis_session(self, bundle):
            allowance = self.guard.start_run()
            bundle["inference"] = {"calls": [], "model": self.model}
            try:
                allowance.before_request()
                bundle["inference"]["calls"].append({
                    "requested_model": self.model,
                    "attempts": [{"request_attempted": True, "usage": {
                        "prompt_tokens": 10, "completion_tokens": 2,
                    }}],
                })
                yield self
            finally:
                allowance.close()
                bundle["inference"]["allowance"] = self.guard.snapshot()
                bundle["inference"]["fixture_closed"] = True

        def index(self, bundle):
            bundle["trace"].append({"stage": "Caption scene 1", "status": "complete"})

        def save_index(self, bundle):
            pass

        def qa(self, bundle, question, options):
            bundle["trace"].append({"stage": "Planner", "status": "complete"})
            yield bundle
            if getattr(self, "fail", False):
                bundle["trace"].append({"stage": "Answer", "status": "failed",
                                         "error": key})
                raise runner.HostedVLMError("credits")
            bundle["selected"] = bundle["scenes"]
            bundle["answer"] = {"answer": "Fixture cleaning", "supporting_scenes": [1]}
            bundle["verification"] = {"accepted": False, "issues": ["Fixture uncertainty"]}
            bundle["trace"].extend([
                {"stage": "Answer", "output": bundle["answer"]},
                {"stage": "Verifier", "output": bundle["verification"]},
            ])
            yield bundle

    monkeypatch.setattr(runner, "RemotePipeline", Pipeline)
    return tmp_path, calls, Pipeline, key


def test_one_run_artifacts_closed_trace_shared_ledger_and_no_overwrite(isolated_runner, capsys):
    root, calls, _, key = isolated_runner
    for model, case in ((runner.GPT41_MINI_MODEL, "sofa_mc"),
                        (runner.GPT54_MINI_MODEL, "kitchen_open")):
        assert runner.evaluate(model, case, root) == 0
        output = root / f"{model}__{case}"
        trace = json.loads((output / "raw-trace.json").read_text())
        report = json.loads((output / "report.json").read_text())
        manifest = json.loads((output / "manifest.json").read_text())
        assert trace[-1]["stage"] == "Hosted inference usage"
        assert trace[-1]["fixture_closed"] is True
        assert report["qa_completed"] is True and report["verifier_accepted"] is False
        assert report["last_pipeline_stage"] == "Verifier"
        assert report["inference_invoked"] is True and report["selected_frame_paths"]
        assert model in report["selected_frame_paths"][0]
        assert manifest["limits"] == {"max_runs": 4, "max_requests": 120,
                                      "max_requests_per_run": 32}
    assert calls[-1].guard.snapshot()["runs_used"] == 2
    assert calls[-1].guard.snapshot()["request_attempts_used"] == 2
    with pytest.raises(runner.EvaluationError, match="no rerun"):
        runner.evaluate(runner.GPT41_MINI_MODEL, "sofa_mc", root)
    assert len(calls) == 2
    assert key not in capsys.readouterr().out


def test_failure_keeps_partial_trace_accounting_and_redacts_key(isolated_runner, capsys):
    root, _, pipeline, key = isolated_runner
    pipeline.fail = True
    assert runner.evaluate(runner.GPT54_MINI_MODEL, "sofa_mc", root) == 1
    output = root / f"{runner.GPT54_MINI_MODEL}__sofa_mc"
    report = json.loads((output / "report.json").read_text())
    trace = json.loads((output / "raw-trace.json").read_text())
    assert report["qa_completed"] is False and report["error_code"] == "credits"
    assert report["last_pipeline_stage"] == "Answer"
    assert trace[-1]["fixture_closed"] is True
    assert report["allowance"]["request_attempts_used"] == 1
    assert all(key not in path.read_text() for path in output.glob("*.json"))
    assert "[REDACTED]" in (output / "raw-trace.json").read_text()
    assert key not in str(capsys.readouterr())


def test_json_redaction_handles_escaped_key_without_overwriting(tmp_path):
    path = tmp_path / "trace.json"
    key = 'sk-fixture-"quoted"'
    runner.write_json(path, {"message": key}, key)
    assert json.loads(path.read_text()) == {"message": "[REDACTED]"}
    with pytest.raises(FileExistsError):
        runner.write_json(path, {}, key)
