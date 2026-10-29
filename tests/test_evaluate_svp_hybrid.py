"""Offline hybrid-runner checks: no owner credential or model request."""

import importlib.util
import json
import os
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/evaluate_svp_hybrid.py"
spec = importlib.util.spec_from_file_location("evaluate_svp_hybrid_fixture", SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.mark.parametrize("case,count", [("sofa_mc", 5), ("kitchen_open", 9)])
def test_dry_run_checks_real_seed_and_historical_controls_without_secret_or_writes(
    case, count, monkeypatch, tmp_path, capsys,
):
    if not runner.HISTORICAL_ROOT.exists():
        pytest.skip("Historical evaluation artifacts are local-only")
    module = runner.importlib.import_module("videomasys.zerogpu.svp_prompts")
    monkeypatch.setattr(module, "PROMPT_VERSION", runner.VERSION)
    monkeypatch.setattr(runner, "ROOT", tmp_path / "not_created")
    monkeypatch.setattr(runner, "read_owner_key", lambda: pytest.fail("credential read"))
    monkeypatch.setattr(runner.RemotePipeline, "analysis_session",
                        lambda *args: pytest.fail("session admitted"))
    monkeypatch.setattr(runner.CPUEmbedder, "__call__", lambda *args: pytest.fail("embedding"))
    monkeypatch.setattr(runner, "UsageGuard", lambda *a, **kw: pytest.fail("ledger opened"))
    before = runner.file_sha(runner.HISTORICAL_ROOT / "usage.sqlite3")
    assert runner.main(["--case", case]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["dry_run"] is True and result["scene_count"] == count
    assert result["credential_read"] is False and result["model_called"] is False
    assert result["prompt_version"] == "svp-roles-v3"
    assert result["limits"] == {"max_runs": 2, "max_requests": 24, "max_requests_per_run": 12}
    assert result["historical_comparison"]["closed_batch"]["request_attempts_used"] == 54
    assert result["historical_comparison"]["closed_batch"]["runs_used"] == 6
    assert set(result["historical_comparison"]["controls"]) == {"clean-roles-v2", "svp-roles-v2"}
    assert runner.file_sha(runner.HISTORICAL_ROOT / "usage.sqlite3") == before
    assert not runner.ROOT.exists()


@pytest.mark.parametrize("version", ["svp-roles-v1", "svp-roles-v2", "svp-roles-v4"])
def test_only_v3_can_enter_the_new_batch(version, monkeypatch):
    module = runner.importlib.import_module("videomasys.zerogpu.svp_prompts")
    monkeypatch.setattr(module, "PROMPT_VERSION", version)
    monkeypatch.setattr(runner, "cache_seed", lambda *a: pytest.fail("input read"))
    with pytest.raises(runner.EvaluationError, match="svp-roles-v3"):
        runner.preflight("sofa_mc")
    with pytest.raises(runner.EvaluationError, match="two authorized"):
        runner.preflight("new_case")


def test_preserved_old_ledger_cannot_be_reopened_or_modified(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "HISTORICAL_ROOT", tmp_path)
    guard = runner.UsageGuard(tmp_path / "usage.sqlite3", max_runs=6, max_requests=60,
                              max_requests_per_run=12)
    for _ in range(6):
        allowance = guard.start_run()
        allowance.before_request()
        allowance.close()
    before = runner.file_sha(guard.path)
    with pytest.raises(runner.EvaluationError, match="preserved closed"):
        runner.historical_controls("sofa_mc", {})
    assert runner.file_sha(guard.path) == before
    assert guard.snapshot()["request_attempts_used"] == 6


def test_new_budget_limits_actual_attempts_and_survives_reopening(tmp_path):
    from videomasys.zerogpu.usage import UsageLimitError

    for _ in range(2):
        guard = runner.UsageGuard(tmp_path / "usage.sqlite3", **runner.LIMITS)
        allowance = guard.start_run()
        for _ in range(12):
            allowance.before_request()
        with pytest.raises(UsageLimitError):
            allowance.before_request()
        allowance.close()
    assert guard.snapshot()["request_attempts_used"] == 24
    assert guard.snapshot()["runs_used"] == 2
    with pytest.raises(UsageLimitError):
        guard.start_run()


def test_two_sessions_close_batch_even_if_requests_remain(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    guard = runner.UsageGuard(tmp_path / "usage.sqlite3", **runner.LIMITS)
    for _ in range(2):
        allowance = guard.start_run()
        allowance.before_request()
        allowance.close()
    with pytest.raises(runner.EvaluationError, match="closed"):
        runner.batch_state({})
    assert guard.snapshot()["request_attempts_used"] == 2


def test_missing_or_symlinked_new_ledger_never_gets_replaced(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    with pytest.raises(runner.EvaluationError, match="missing"):
        runner.batch_state({})
    assert not (tmp_path / "usage.sqlite3").exists()
    (tmp_path / "usage.sqlite3").symlink_to(tmp_path / "not_a_ledger")
    with pytest.raises(runner.EvaluationError, match="missing"):
        runner.batch_state({})
    assert not (tmp_path / "not_a_ledger").exists()


def test_second_run_rejects_changed_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    guard = runner.UsageGuard(tmp_path / "usage.sqlite3", **runner.LIMITS)
    guard.start_run().close()
    directory = tmp_path / "svp__svp-roles-v3__sofa_mc"
    directory.mkdir()
    runner.write_json(directory / "manifest.json", {"source_sha256": {"code": "old"}}, "secret")
    with pytest.raises(runner.EvaluationError, match="changed after the first"):
        runner.batch_state({"code": "new"})
    assert runner.batch_state({"code": "old"})["runs_used"] == 1


@pytest.fixture
def fixed_seed():
    return {
        "key": "fixture-video-hash", "duration": 4.0, "video_path": "/fixture/video.mp4",
        "video_name": "video.mp4", "question": "What is happening?", "options": [],
        "scenes": [{"index": 0, "start": 0.0, "end": 4.0, "frames": [],
                    "frame_times": [], "caption": "The person moves a cup."}],
        "vectors": [[1.0] + [0.0] * 383], "original_signature": "original-clean-index",
        "provenance": {"fixed_input_sha256": "immutable-inputs"},
    }


def test_one_admission_per_case_pinned_model_safe_artifacts_and_session_closure(
    fixed_seed, monkeypatch, tmp_path,
):
    monkeypatch.setattr(runner, "ROOT", tmp_path / "new_batch")
    key = "sk-fixture-not-an-owner-credential"
    monkeypatch.setattr(runner, "read_owner_key", lambda: key)
    source = "scripts/evaluate_svp_hybrid.py"
    hashes = {source: runner.file_sha(runner.REPO / source)}
    monkeypatch.setattr(runner, "current_sources", lambda: hashes)
    admissions = []

    def preflight(case):
        return deepcopy(fixed_seed), SimpleNamespace(
            signature="fixture-v3", embed=None, index=lambda bundle: bundle.update(cached=True),
        ), {"controls": "historical, not fresh"}, hashes, runner.batch_state(hashes)

    class Pipeline:
        def __init__(self, **kwargs):
            assert kwargs["token"] == key and kwargs["provider"] == "openai"
            assert kwargs["prompt_profile"] == "svp"
            assert kwargs["vlm_factory"] is runner.ProvenanceVLM
            assert os.environ["VLM_MODEL"] == runner.GPT41_MINI_MODEL
            self.guard = kwargs["guard"]

        @contextmanager
        def analysis_session(self, bundle):
            allowance = self.guard.start_run()
            admissions.append(allowance)
            bundle["inference"] = {"calls": []}

            def qa(bundle, question, options):
                allowance.before_request()
                bundle["inference"]["calls"].append({"attempts": [{"request_attempted": True}]})
                bundle["trace"].append({"stage": "Fixture QA", "status": "complete"})
                bundle["answer"] = {"answer": f"Moving a cup. {key}"}
                bundle["verification"] = {"accepted": True}
                yield bundle

            try:
                yield SimpleNamespace(signature="fixture-v3", qa=qa)
            finally:
                allowance.close()
                bundle["inference"]["closed_before_publication"] = True

    monkeypatch.setattr(runner, "preflight", preflight)
    monkeypatch.setattr(runner, "RemotePipeline", Pipeline)
    for case in runner.CASES:
        assert runner.evaluate(case, run=True) == 0
        directory = runner.ROOT / f"svp__svp-roles-v3__{case}"
        report = json.loads((directory / "report.json").read_text())
        trace = json.loads((directory / "raw-trace.json").read_text())
        manifest = json.loads((directory / "manifest.json").read_text())
        assert report["qa_completed"] and report["result_is_execution_not_correctness"]
        assert report["inference_invoked"] and report["sources_unchanged_during_run"]
        assert trace[-1]["closed_before_publication"]
        assert not manifest["caption_generation_enabled"]
        assert not manifest["previous_batch_modified"] and not manifest["public_space_modified"]
        assert (directory / "source-snapshots.json").is_file()
        assert all(key not in path.read_text() for path in directory.glob("*.json"))
    assert len(admissions) == 2 and all(run.closed for run in admissions)
    with pytest.raises(runner.EvaluationError, match="closed"):
        runner.evaluate("sofa_mc", run=True)
    assert len(admissions) == 2


def test_cache_miss_fails_before_credential_or_admission(fixed_seed, monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "ROOT", tmp_path / "new_batch")
    monkeypatch.setattr(runner, "read_owner_key", lambda: pytest.fail("credential read"))
    monkeypatch.setattr(runner, "UsageGuard", lambda *a, **kw: pytest.fail("ledger opened"))
    monkeypatch.setattr(runner, "preflight", lambda case: (
        deepcopy(fixed_seed), SimpleNamespace(
            signature="fixture-v3", index=lambda bundle: bundle.update(cached=False),
        ), {}, {}, {},
    ))
    with pytest.raises(runner.EvaluationError, match="exact historical cache hit"):
        runner.evaluate("sofa_mc", run=True)
    assert not (runner.ROOT / "usage.sqlite3").exists()
