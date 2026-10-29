"""No paid calls: final recheck cohort, accounting isolation and completion gate."""

import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "final_request_smoke_fixture", REPO / "scripts/evaluate_request_contract_smoke_round2.py",
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_exact_six_cases_and_prior_results_are_bound():
    if not (runner.PRIOR / "live-audit.json").is_file():
        pytest.skip("Completed local development artifacts unavailable")
    manifest = runner.prepare_manifest()
    assert tuple(case["id"] for case in manifest["cases"]) == runner.CASE_IDS
    assert manifest["previous_reported_usd"] + runner.base.MAX_USD < 0.5
    assert manifest["limits"] == {"max_runs": 6, "max_requests": 240,
                                  "max_requests_per_run": 40, "max_usd": 0.3}
    assert str(runner.PRIOR / "usage.sqlite3") in manifest["controls_sha256"]
    assert str(runner.PRIOR / "live-audit.json") in manifest["controls_sha256"]
    assert not manifest["publication_authorized"]


def test_accounting_isolation_and_hard_cap(tmp_path):
    import benchmark_accounting as old

    assert old.RESERVATION_UNITS == 160_000
    assert runner.previous.accounting.RESERVATION_UNITS == 300_000
    budget = runner.base.Budget(tmp_path / "new.sqlite3", max_usd=0.3)
    for _ in range(10):
        budget.reserve("case")
    assert budget.snapshot()["reserved_usd"] == 0.3
    with pytest.raises(runner.previous.accounting.BudgetExceeded, match="allowance"):
        budget.reserve("case")
    assert old.RESERVATION_UNITS == 160_000


@pytest.mark.parametrize("complete,passes", [(6, True), (5, False)])
def test_six_completed_gate_allows_valid_abstention_but_not_retained_draft(
    monkeypatch, complete, passes,
):
    rows = [{"id": f"case{i}", "completed": i < complete, "correct": i < 4}
            for i in range(6)]
    result = {"results": rows, "frozen_inputs_unchanged": True,
              "answer_metrics": {}, "library_gate": {"passed": True},
              "budget": {"charged_usd": 0.1}}
    monkeypatch.setattr(runner.previous, "summarize", lambda *_: deepcopy(result))
    summary = runner.summarize({"previous_reported_usd": 0.1880604})
    assert summary["finished"]
    assert summary["development_gate"]["passed"] is passes
    assert summary["answer_metrics"]["accuracy_all_planned"] == 4 / 6
    assert summary["combined_charged_usd"] == 0.2880604
    assert not summary["publication_authorized"]


def test_default_preflight_does_not_admit_or_read_credentials(monkeypatch, capsys):
    monkeypatch.setattr(runner, "load_manifest", lambda: {"previous_reported_usd": 0.1880604})
    monkeypatch.setattr(runner.base, "read_owner_key", lambda: pytest.fail("credential read"))
    monkeypatch.setattr(runner.base, "UsageGuard", lambda *_a, **_k: pytest.fail("admission"))
    assert runner.main([]) == 0
    value = json.loads(capsys.readouterr().out)
    assert value["cases"] == 6 and value["max_usd"] == 0.3
    assert not value["credential_read"] and not value["model_called"]
