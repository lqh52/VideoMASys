"""Offline saved-input isolation and explicit reviewed-pair completion checks."""

import importlib.util
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "downstream_replay_fixture", REPO / "scripts/replay_structured_v6_answers.py",
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_saved_inputs_exclude_old_answers_and_labels_and_patch_only_isolated_runtime():
    from videomasys.zerogpu import structured_pipeline as live

    if not (runner.PRIOR / "live-audit.json").is_file():
        pytest.skip("Saved local development records unavailable")
    manifest = runner.prepare_manifest()
    assert tuple(case["id"] for case in manifest["cases"]) == runner.CASE_IDS
    case = manifest["cases"][0]
    bundle, frames, observations, graph, plan, retrieval, candidate = runner.seed_inputs(case)
    assert candidate == "Unwrap it"
    assert not {"answer", "verification", "reference", "video_path", "frame_directory"} \
        & bundle.keys()
    assert not {"reasoning", "review", "claim_provenance"} & graph.keys()
    original = live.inspect_evidence
    local = runner.isolated("_seeded_fixture", "src/videomasys/zerogpu/structured_pipeline.py")
    pipeline = SimpleNamespace()
    runner.install_seed(local, pipeline, bundle, frames, observations, graph, candidate)
    assert live.inspect_evidence is original
    assert local.inspect_evidence(pipeline, bundle, case["question"], plan, retrieval) == (
        frames, observations, graph,
    )
    assert pipeline.choose_option(bundle["trace"], case["question"], case["options"],
                                  [frame.path for frame in frames]) == candidate
    assert all(step["status"] == "seeded" and step["not_executed"]
               and not step["attempts"] for step in bundle["trace"])


def pair_fixture():
    raw = {"fixture": "structured review"}
    verdict = {"accepted": True, "explanation_accepted": False,
               "revision_scope": "explanation", "needs_revision": True,
               "advisories": ["A contextual note"], "direct_assessments": raw}
    answer, reasoning = {"selected_choice": "fixture answer"}, {"claims": ["fixture"]}
    return {"answer": answer, "verification": deepcopy(verdict),
            "evidence_graph": {"reasoning": reasoning},
            "reviewed_pair": {"answer_trace_index": 0, "review_trace_index": 1},
            "final_answer_complete": True, "review_complete": True,
            "trace": [{"stage": "Answer", "status": "complete", "revision": 0,
                       "output": answer, "structured_output": reasoning},
                      {"stage": "Verifier", "status": "complete", "revision": 0,
                       "output": verdict, "structured_output": raw}]}


def test_only_matching_bound_pair_can_complete_and_restore_annotations_are_narrow():
    bundle = pair_fixture()
    assert runner.matching_final_pair(bundle)
    bundle["trace"].append({"stage": "Answer", "status": "failed"})
    assert not runner.matching_final_pair(bundle)
    bundle.update(explanation_repair_failed=True, retained_reviewed_answer=True)
    bundle["verification"].update(needs_revision=False, explanation_repair_failed=True,
                                 advisories=["A contextual note", runner.RESTORE_NOTE])
    assert runner.matching_final_pair(bundle)
    for change in ({"accepted": False}, {"advisories": ["Invented annotation"]}):
        changed = deepcopy(bundle)
        changed["verification"].update(change)
        assert not runner.matching_final_pair(changed)
    for change in ({"retained_previous_draft": True}, {"review_complete": False},
                   {"final_answer_complete": False}, {"reviewed_pair": {}}):
        assert not runner.matching_final_pair({**bundle, **change})


def test_downstream_budget_remains_isolated_and_capped(tmp_path):
    import benchmark_accounting as old

    budget = runner.base.Budget(tmp_path / "fixture.sqlite3", max_usd=runner.MAX_USD)
    for _ in range(5):
        budget.reserve("fixture")
    assert budget.snapshot()["reserved_usd"] == 0.15
    with pytest.raises(runner.shared.accounting.BudgetExceeded):
        budget.reserve("fixture")
    assert old.RESERVATION_USD == 0.016
    assert runner.LIMITS == {"max_runs": 2, "max_requests": 16, "max_requests_per_run": 8}


def test_default_preflight_has_no_credentials_embeddings_or_admission(monkeypatch, capsys):
    monkeypatch.setattr(runner, "load_manifest", lambda: {})
    monkeypatch.setattr(runner.base, "read_owner_key", lambda: pytest.fail("credential read"))
    monkeypatch.setattr(runner.base, "CPUEmbedder", lambda: pytest.fail("embedding"))
    monkeypatch.setattr(runner.base, "UsageGuard", lambda *_a, **_k: pytest.fail("admission"))
    assert runner.main([]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["network_calls"] == 0 and not result["credential_read"]
