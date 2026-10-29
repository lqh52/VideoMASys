"""No API calls: separate round-two admission and immutable first-round controls."""

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "round2_library_fixture", REPO / "scripts/evaluate_library_v4_round2.py",
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_round_two_has_independent_budget_and_same_strict_case_runner():
    import evaluate_library_v4 as first

    assert runner.base.ROOT != first.base.ROOT
    assert runner.base.MAX_USD == 0.9 and first.base.MAX_USD == 1
    assert runner.base.LIMITS == {"max_runs": 4, "max_requests": 160,
                                 "max_requests_per_run": 40}
    assert runner.base.run_case is runner.library.shared.run_case
    assert runner.base.summarize is runner.library.summarize


def test_completed_round_is_bound_and_combined_cap_is_below_one_dollar():
    if not (runner.FIRST_ROUND / "cost-audit.json").is_file():
        pytest.skip("Local completed first round unavailable")
    controls, spent = runner.first_round_controls()
    assert len(controls) == 15
    assert spent == 0.0790316 and spent + runner.base.MAX_USD < 1
    assert str(runner.FIRST_ROUND / "usage.sqlite3") in controls
    assert str(runner.FIRST_ROUND / "budget.sqlite3") in controls
    assert str(runner.FIRST_ROUND / "cost-audit.json") in controls


def test_preflight_never_admits_or_reads_keys(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(runner.base, "ROOT", tmp_path)
    monkeypatch.setattr(runner.base, "read_owner_key", lambda: pytest.fail("credential read"))
    monkeypatch.setattr(runner.base, "UsageGuard", lambda *_a, **_k: pytest.fail("admission"))
    monkeypatch.setattr(runner.library, "prepare_manifest", lambda: {"cases": []})
    monkeypatch.setattr(runner.library.shared, "controls_unchanged", lambda _: True)
    assert runner.main(["--prepare"]) == 0
    assert not json.loads(capsys.readouterr().out)["model_called"]
    assert runner.main([]) == 0
    dry = json.loads(capsys.readouterr().out)
    assert dry["cases"] == 4 and dry["max_usd"] == 0.9
    assert not dry["credential_read"] and not dry["model_called"]
    assert not (tmp_path / "usage.sqlite3").exists()
    with pytest.raises(runner.base.EvaluationError, match="no overwrite"):
        runner.main(["--prepare"])
