"""No model calls: isolated conservative reservations and development denominators."""

import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "request_contract_smoke_fixture", REPO / "scripts/evaluate_request_contract_smoke.py",
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_new_reservation_isolated_from_old_helper_and_covers_larger_allowance(tmp_path):
    import benchmark_accounting as old

    assert old.RESERVATION_USD == 0.016 and old.RESERVATION_UNITS == 160_000
    assert runner.accounting.RESERVATION_UNITS == 300_000
    assert 4 * 65536 + 16 * 1600 < runner.accounting.RESERVATION_UNITS
    budget = runner.base.Budget(tmp_path / "new.sqlite3", max_usd=0.5)
    ticket = budget.reserve("case")
    assert budget.snapshot()["reserved_usd"] == 0.03
    budget.settle(ticket, None)
    assert budget.snapshot()["ambiguous_usd"] == 0.03
    assert old.RESERVATION_USD == 0.016


@pytest.mark.parametrize("completed,eligible,passes", [(9, 8, True), (8, 8, False)])
def test_development_completion_gate_is_separate_from_answer_metrics(
    monkeypatch, completed, eligible, passes,
):
    rows = [{"id": f"case{i}", "dataset": "nextqa" if i < 6 else "egoschema",
        "completed": i < completed, "release_eligible_answer": i < eligible,
        "correct": i < eligible, "selected_option": 1 if i < eligible else None,
        "sources_unchanged": True} for i in range(9)]
    result = {"results": rows, "datasets": {name: {
        "correct": sum(row["correct"] for row in rows if row["dataset"] == name),
        "attempted": sum(row["dataset"] == name for row in rows),
    } for name in ("nextqa", "egoschema")}}
    monkeypatch.setattr(runner.shared, "_summarize", lambda *_: deepcopy(result))
    monkeypatch.setattr(runner.shared, "frozen_inputs_unchanged", lambda: True)
    monkeypatch.setattr(runner, "controls_unchanged", lambda _: True)
    summary = runner.summarize({"cases": rows})
    assert summary["development_gate"]["passed"] is passes
    assert summary["answer_metrics"]["nonabstaining_reviewed_answers"] == eligible
    assert not summary["library_gate"]["publication_authorized"]
    assert summary["datasets"]["nextqa"]["planned"] == 6
    assert summary["datasets"]["egoschema"]["planned"] == 3


def test_smoke_default_preflight_cannot_read_keys_or_admit(monkeypatch, capsys):
    if not (runner.base.ROOT / "manifest.json").is_file():
        pytest.skip("Local development manifest unavailable")
    monkeypatch.setattr(runner.base, "read_owner_key", lambda: pytest.fail("credential read"))
    monkeypatch.setattr(runner.base, "UsageGuard", lambda *_a, **_k: pytest.fail("admission"))
    assert runner.main([]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["cases"] == 9 and result["max_usd"] == 0.5
    assert not result["model_called"] and not result["credential_read"]
    assert result["request_reservation_usd"] == 0.03
