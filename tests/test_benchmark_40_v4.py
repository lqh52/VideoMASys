"""Offline gate/admission fixtures; no real model requests or owner credentials."""

import importlib.util
import json
import sys
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
spec = importlib.util.spec_from_file_location(
    "benchmark40_v4_fixture", REPO / "scripts/evaluate_benchmark_40_v4.py",
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def final_bundle():
    answer = {"answer": "The item moves.", "selected_option": 1, "selected_choice": "B"}
    review = {"accepted": False, "checks": []}
    return {
        "options": ["A", "B", "C", "D", "E"], "answer": answer, "verification": review,
        "final_answer_complete": True, "review_complete": True,
        "resolution": "answer_proposed", "status": "unverified",
        "trace": [{"stage": "Answer", "status": "complete", "revision": 1, "output": answer},
                  {"stage": "Verifier", "status": "complete", "revision": 1, "output": review}],
    }


def test_loading_v4_does_not_reconfigure_the_historical_runner():
    import evaluate_benchmark_40 as historical

    assert historical.VERSION == "structured-evidence-v3"
    assert historical.ROOT == REPO / "artifacts/benchmark_40_2026-09-08"
    assert runner.base.VERSION == "structured-evidence-v4"
    assert runner.base.ROOT != historical.ROOT
    assert runner.base.LIMITS == {"max_runs": 40, "max_requests": 1600,
                                 "max_requests_per_run": 40}
    assert runner.base.MAX_USD == 4


def test_review_disagreement_does_not_disqualify_normally_completed_final_answer():
    state = final_bundle()
    assert not state["verification"]["accepted"]
    assert runner.final_state(state) and runner.nonabstaining(state)


@pytest.mark.parametrize("change", [
    {"review_complete": False}, {"final_answer_complete": False},
    {"retained_previous_draft": True}, {"review_complete": "true"},
    {"verification": {"accepted": True}},
])
def test_previous_drafts_and_unmatched_reviews_never_qualify(change):
    state = {**final_bundle(), **change}
    assert not runner.final_state(state)


def test_failed_latest_answer_or_wrong_revision_cannot_reuse_previous_review():
    state = final_bundle()
    state["trace"].append({"stage": "Answer", "status": "failed", "revision": 2})
    assert not runner.final_state(state)
    state = final_bundle()
    state["trace"][-1]["revision"] = 0
    assert not runner.final_state(state)


@pytest.mark.parametrize("change", [
    {"selected_option": None}, {"selected_option": True}, {"selected_option": 5},
    {"selected_choice": "A"}, {"answer": "  "},
])
def test_abstentions_and_inconsistent_selections_are_not_release_answers(change):
    state = final_bundle()
    state["answer"].update(change)
    assert not runner.nonabstaining(state)


@pytest.mark.parametrize("retained", [False, True])
def test_live_path_keeps_labels_out_and_never_scores_retained_draft(
    tmp_path, monkeypatch, retained,
):
    base = runner.base
    monkeypatch.setattr(base, "ROOT", tmp_path)
    monkeypatch.setattr(base, "sources", lambda: {})
    (tmp_path / "cases").mkdir()
    case = {"id": "nextqa-fixture", "dataset": "nextqa", "split": "val",
        "video": "test.mp4", "video_sha256": "fixture", "duration_seconds": 20,
        "question": "What happens?", "options": ["A", "B", "C", "D", "E"],
        "reference": {"correct_option": 1, "correct_choice": "B"}}
    monkeypatch.setattr(base, "prepare_video", lambda *_: {
        "key": "fixture", "scenes": [], "trace": [], "cached": False,
    })

    class Pipeline:
        def __init__(self, **kwargs):
            assert kwargs["provider"] == "openai" and kwargs["prompt_profile"] == "structured"

        @contextmanager
        def analysis_session(self, bundle):
            assert "reference" not in bundle and "correct_option" not in bundle
            yield self

        def index(self, bundle):
            assert "reference" not in bundle

        def save_index(self, bundle):
            bundle["inference"] = {"calls": []}

        def qa(self, bundle, question, options):
            assert question == case["question"] and options == case["options"]
            assert "reference" not in bundle
            bundle.update(final_bundle())
            if retained:
                bundle.update(retained_previous_draft=True, final_answer_complete=False,
                              review_complete=False)
            yield bundle

    monkeypatch.setattr(base, "RemotePipeline", Pipeline)
    report = runner.run_case(case, "fixture-key", None, None, None, {})
    assert report["completed"] is (not retained)
    assert report["correct"] is report["release_eligible_answer"] is (not retained)
    saved = json.loads((tmp_path / "cases/nextqa-fixture/bundle.json").read_text())
    assert "reference" not in saved
    with pytest.raises(FileExistsError):
        runner.run_case(case, "fixture-key", None, None, None, {})


@pytest.mark.parametrize("eligible,correct,finished,passes", [
    (36, 28, True, True), (35, 28, True, False), (36, 27, True, False),
    (40, 40, False, False),
])
def test_release_gate_requires_both_thresholds_and_all_cases_finalized(
    monkeypatch, eligible, correct, finished, passes,
):
    result = {"finished": finished, "datasets": {"nextqa": {}, "egoschema": {}},
              "results": [{"dataset": "nextqa" if i < 20 else "egoschema",
                           "release_eligible_answer": i < eligible, "correct": i < correct,
                           "sources_unchanged": True}
                          for i in range(40)]}
    monkeypatch.setattr(runner, "_summarize", lambda *_: deepcopy(result))
    monkeypatch.setattr(runner, "controls_unchanged", lambda *_: True)
    monkeypatch.setattr(runner, "frozen_inputs_unchanged", lambda: True)
    summary = runner.summarize({})
    assert summary["release_gate"]["passed"] is passes
    counted = sum(v["nonabstaining_reviewed_answers"] for v in summary["datasets"].values())
    assert counted == eligible


@pytest.mark.parametrize("source_change,manifest_change", [(True, False), (False, True)])
def test_release_rejects_source_or_manifest_drift(tmp_path, monkeypatch,
                                                 source_change, manifest_change):
    monkeypatch.setattr(runner.base, "ROOT", tmp_path)
    monkeypatch.setattr(runner.base, "sources", lambda: {"source.py": "new" if source_change
                                                       else "old"})
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    config = {"source_sha256": {"source.py": "old"},
              "manifest_sha256": runner.base.file_sha(manifest)}
    (tmp_path / "run-config.json").write_text(json.dumps(config))
    if manifest_change:
        manifest.write_text('{"changed": true}')
    assert not runner.frozen_inputs_unchanged()


def test_new_source_snapshot_list_includes_new_and_reused_helpers():
    helpers = runner.base.HELPERS
    for name in ("evaluate_benchmark_40_v4.py", "prepare_benchmark_40_v4.py",
                 "evaluate_benchmark_40.py", "prepare_benchmark_40.py", "benchmark_accounting.py"):
        assert "scripts/" + name in helpers


def test_manifest_cannot_be_resampled_or_overwritten(tmp_path):
    from prepare_benchmark_40_v4 import build_manifest

    path = tmp_path / "manifest.json"
    path.write_text("already frozen")
    with pytest.raises(FileExistsError, match="no resampling"):
        build_manifest(tmp_path, tmp_path, tmp_path, path)
    assert path.read_text() == "already frozen"


def test_new_preflight_is_read_only_and_refuses_credentials(monkeypatch, capsys):
    if not (runner.base.ROOT / "manifest.json").exists():
        pytest.skip("Local-only new cohort not prepared yet")
    monkeypatch.setattr(runner.base, "PROMPT_VERSION", runner.VERSION)
    monkeypatch.setattr(runner.base, "read_owner_key", lambda: pytest.fail("credential read"))
    monkeypatch.setattr(runner.base, "UsageGuard", lambda *_a, **_k: pytest.fail("admission"))
    assert runner.main([]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["cases"] == 40 and result["seed"] == 20260909
    assert not result["credential_read"] and not result["model_called"]
    assert result["release_gate"]["minimum_correct"] == 28
