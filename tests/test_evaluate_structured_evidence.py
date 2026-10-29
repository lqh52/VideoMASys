"""Offline evaluation guard checks; no owner credentials or paid inference."""

import importlib.util
import json
import os
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/evaluate_structured_evidence.py"
spec = importlib.util.spec_from_file_location("evaluate_structured_evidence_fixture", SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.mark.parametrize("case,count", [("sofa_mc", 5), ("kitchen_open", 9)])
def test_cached_preflight_with_fixture_version_is_read_only_and_preserves_closed_batches(
    case, count, monkeypatch, tmp_path, capsys,
):
    if not runner.HYBRID_ROOT.exists():
        pytest.skip("Historical evaluation artifacts are local-only")
    # Exercise archival runner guards without claiming current prompts are the tested v1.
    module = runner.importlib.import_module("videomasys.zerogpu.structured_prompts")
    monkeypatch.setattr(module, "PROMPT_VERSION", runner.VERSION)
    hybrid = runner.importlib.import_module("videomasys.zerogpu.svp_prompts")
    for role in ("planner_prompt", "tool_prompt", "caption_prompt"):
        monkeypatch.setattr(module, role, getattr(hybrid, role))
    monkeypatch.setattr(runner, "ROOT", tmp_path / "not_created")
    monkeypatch.setattr(runner, "read_owner_key", lambda: pytest.fail("credential read"))
    monkeypatch.setattr(runner.RemotePipeline, "analysis_session",
                        lambda *a: pytest.fail("admission"))
    monkeypatch.setattr(runner.CPUEmbedder, "__call__", lambda *a: pytest.fail("embedding"))
    monkeypatch.setattr(runner, "UsageGuard", lambda *a, **kw: pytest.fail("new ledger"))
    ledgers = [root / "usage.sqlite3" for root in (runner.HYBRID_ROOT, runner.SVP_ROOT)]
    before = [runner.file_sha(path) for path in ledgers]
    assert runner.main(["--case", case]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["dry_run"] and result["scene_count"] == count
    assert result["model"] == runner.GPT41_MINI_MODEL
    assert result["prompt_version"] == "structured-evidence-v1"
    assert result["credential_read"] is False and result["model_called"] is False
    assert result["limits"] == {"max_runs": 2, "max_requests": 24, "max_requests_per_run": 12}
    assert [(item["runs_used"], item["request_attempts_used"])
            for item in result["historical_comparison"]["closed_ledgers"]] == [(2, 17), (6, 54)]
    assert [runner.file_sha(path) for path in ledgers] == before
    assert not runner.ROOT.exists()


def test_retrieval_prompt_replacement_is_not_a_downstream_only_experiment(monkeypatch):
    module = runner.importlib.import_module("videomasys.zerogpu.structured_prompts")
    monkeypatch.setattr(module, "PROMPT_VERSION", runner.VERSION)
    monkeypatch.setattr(module, "tool_prompt", lambda *a: "Different compiler")
    monkeypatch.setattr(runner, "cache_seed", lambda *a: pytest.fail("read inputs"))
    with pytest.raises(runner.EvaluationError, match="unchanged hybrid retrieval"):
        runner.preflight("sofa_mc")


def test_closed_v1_runner_rejects_new_version_or_case_before_inputs_or_key(monkeypatch):
    module = runner.importlib.import_module("videomasys.zerogpu.structured_prompts")
    assert module.PROMPT_VERSION == "structured-evidence-v3"
    monkeypatch.setattr(runner, "cache_seed", lambda *a: pytest.fail("read inputs"))
    monkeypatch.setattr(runner, "read_owner_key", lambda: pytest.fail("credential read"))
    with pytest.raises(runner.EvaluationError, match="Only structured-evidence-v1"):
        runner.preflight("sofa_mc")
    with pytest.raises(runner.EvaluationError, match="two authorized"):
        runner.preflight("new_case")


def test_historical_counts_must_match_closed_state_without_modification(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "HYBRID_ROOT", tmp_path)
    guard = runner.UsageGuard(tmp_path / "usage.sqlite3", **runner.LIMITS)
    guard.start_run().close()
    before = runner.file_sha(guard.path)
    with pytest.raises(runner.EvaluationError, match="preserved closed state"):
        runner.historical_control("sofa_mc", {})
    assert runner.file_sha(guard.path) == before
    assert guard.snapshot()["runs_used"] == 1


def test_two_sessions_close_batch_even_with_unused_requests(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    guard = runner.UsageGuard(tmp_path / "usage.sqlite3", **runner.LIMITS)
    for _ in range(2):
        allowance = guard.start_run()
        allowance.before_request()
        allowance.close()
    with pytest.raises(runner.EvaluationError, match="closed"):
        runner.batch_state({})
    assert guard.snapshot()["request_attempts_used"] == 2


def test_missing_or_symlinked_ledger_is_never_replaced(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    with pytest.raises(runner.EvaluationError):
        runner.batch_state({})
    (tmp_path / "usage.sqlite3").symlink_to(tmp_path / "missing")
    with pytest.raises(runner.EvaluationError):
        runner.batch_state({})
    assert not (tmp_path / "missing").exists()


def test_second_case_requires_exact_frozen_source_hashes(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    guard = runner.UsageGuard(tmp_path / "usage.sqlite3", **runner.LIMITS)
    guard.start_run().close()
    directory = runner.output_directory("sofa_mc")
    directory.mkdir()
    runner.write_json(directory / "manifest.json", {"source_sha256": {"code": "old"}}, "secret")
    with pytest.raises(runner.EvaluationError, match="changed after the first"):
        runner.batch_state({"code": "new"})
    assert runner.batch_state({"code": "old"})["runs_used"] == 1


@pytest.fixture
def seed():
    return {
        "key": "fixture-hash", "duration": 4.0, "video_path": "/fixture/video.mp4",
        "video_name": "video.mp4", "question": "What is happening?", "options": [],
        "scenes": [{"index": 0, "start": 0.0, "end": 4.0, "frames": [],
                    "frame_times": [], "caption": "The person moves an item."}],
        "vectors": [[1.0] + [0.0] * 383], "original_signature": "original-clean-index",
        "provenance": {"fixed_input_sha256": "fixed-input-hash"},
    }


def test_real_admission_pattern_pins_model_closes_session_and_preserves_artifacts(
    seed, monkeypatch, tmp_path,
):
    monkeypatch.setattr(runner, "ROOT", tmp_path / "new_batch")
    key = "sk-fixture-only"
    monkeypatch.setattr(runner, "read_owner_key", lambda: key)
    source = "scripts/evaluate_structured_evidence.py"
    hashes = {source: runner.file_sha(runner.REPO / source)}
    monkeypatch.setattr(runner, "current_sources", lambda: hashes)
    admissions = []

    def preflight(case):
        return deepcopy(seed), SimpleNamespace(
            signature="fixture-structured", embed=None,
            index=lambda bundle: bundle.update(cached=True),
        ), {"control_files_sha256": {}, "closed_ledgers": []}, hashes, runner.batch_state(hashes)

    class Pipeline:
        def __init__(self, **kwargs):
            assert kwargs["token"] == key and kwargs["provider"] == "openai"
            assert kwargs["prompt_profile"] == "structured"
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
                bundle["answer"] = {"answer": f"Moving an item. {key}"}
                bundle["verification"] = {"accepted": True}
                yield bundle

            try:
                yield SimpleNamespace(signature="fixture-structured", qa=qa)
            finally:
                allowance.close()
                bundle["inference"]["closed_before_publication"] = True

    monkeypatch.setattr(runner, "preflight", preflight)
    monkeypatch.setattr(runner, "RemotePipeline", Pipeline)
    for case in runner.CASES:
        assert runner.evaluate(case, run=True) == 0
        directory = runner.output_directory(case)
        report = json.loads((directory / "report.json").read_text())
        trace = json.loads((directory / "raw-trace.json").read_text())
        manifest = json.loads((directory / "manifest.json").read_text())
        assert report["qa_completed"] and report["result_is_execution_not_correctness"]
        assert report["inference_invoked"] and report["sources_unchanged_during_run"]
        assert report["historical_controls_unchanged"]
        assert trace[-1]["closed_before_publication"]
        assert not manifest["caption_generation_enabled"]
        assert not manifest["previous_batches_modified"] and not manifest["public_space_modified"]
        assert (directory / "source-snapshots.json").is_file()
        assert all(key not in path.read_text() for path in directory.glob("*.json"))
    assert len(admissions) == 2 and all(run.closed for run in admissions)
    with pytest.raises(runner.EvaluationError, match="closed"):
        runner.evaluate("sofa_mc", run=True)
    assert len(admissions) == 2


def test_existing_case_cannot_rerun_before_session_admission(seed, monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    runner.output_directory("sofa_mc").mkdir()
    monkeypatch.setattr(runner, "preflight", lambda *a: (
        seed, SimpleNamespace(signature="fixture"), {}, {}, {},
    ))
    monkeypatch.setattr(runner, "read_owner_key", lambda: pytest.fail("credential read"))
    with pytest.raises(runner.EvaluationError, match="no rerun"):
        runner.evaluate("sofa_mc", run=True)


def test_cache_miss_stops_before_key_or_admission(seed, monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "ROOT", tmp_path / "new_batch")
    monkeypatch.setattr(runner, "read_owner_key", lambda: pytest.fail("credential read"))
    monkeypatch.setattr(runner, "UsageGuard", lambda *a, **kw: pytest.fail("ledger created"))
    monkeypatch.setattr(runner, "preflight", lambda *a: (
        seed, SimpleNamespace(
            signature="fixture", index=lambda bundle: bundle.update(cached=False)
        ),
        {}, {}, {},
    ))
    with pytest.raises(runner.EvaluationError, match="exact historical cache hit"):
        runner.evaluate("sofa_mc", run=True)
    assert not (runner.ROOT / "usage.sqlite3").exists()
