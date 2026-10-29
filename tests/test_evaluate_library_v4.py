"""Offline library-only budget, official references, and release-gate checks."""

import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
spec = importlib.util.spec_from_file_location(
    "library_v4_fixture", REPO / "scripts/evaluate_library_v4.py",
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_library_budget_and_namespace_are_separate():
    import evaluate_benchmark_40_v4 as benchmark

    assert runner.base.ROOT != benchmark.base.ROOT
    assert runner.base.MAX_USD == 1 and benchmark.base.MAX_USD == 4
    assert runner.base.LIMITS == {"max_runs": 4, "max_requests": 160,
                                 "max_requests_per_run": 40}
    assert benchmark.base.LIMITS["max_runs"] == 40
    assert runner.base.run_case is runner.shared.run_case
    assert runner.base.load_manifest is runner.load_manifest
    assert "scripts/evaluate_library_v4.py" in runner.base.HELPERS


@pytest.mark.parametrize("eligible,correct,finished,stable,passes", [
    (4, 4, True, True, True), (3, 4, True, True, False), (4, 3, True, True, False),
    (3, 3, False, True, False), (4, 4, True, False, False),
])
def test_library_gate_requires_all_four_and_frozen_inputs(
    monkeypatch, eligible, correct, finished, stable, passes,
):
    rows = [{"dataset": "nextqa" if i < 2 else "egoschema",
             "release_eligible_answer": i < eligible, "correct": i < correct,
             "sources_unchanged": True} for i in range(4 if finished else 3)]
    result = {"results": rows, "datasets": {
        dataset: {"correct": sum(row["correct"] for row in rows if row["dataset"] == dataset),
                  "attempted": sum(row["dataset"] == dataset for row in rows)}
        for dataset in ("nextqa", "egoschema")}}
    monkeypatch.setattr(runner.shared, "_summarize", lambda *_: deepcopy(result))
    monkeypatch.setattr(runner.shared, "frozen_inputs_unchanged", lambda: stable)
    monkeypatch.setattr(runner.shared, "controls_unchanged", lambda *_: True)
    summary = runner.summarize({})
    assert summary["planned"] == 4 and summary["finished"] is finished
    assert summary["release_gate"]["passed"] is passes
    assert all(value["planned"] == 2 for value in summary["datasets"].values())


def test_official_library_references_and_cold_cache_protocol():
    if not runner.FRESH_MANIFEST.is_file():
        pytest.skip("Local datasets and new fixed manifest are unavailable")
    manifest = runner.prepare_manifest()
    cases = manifest["cases"]
    assert [case["reference"]["correct_option"] for case in cases] == [2, 2, 3, 4]
    assert [case["id"] for case in cases] == list(runner.EXAMPLE_IDS)
    assert len({case["video_sha256"] for case in cases}) == 3
    assert cases[0]["video_sha256"] == cases[1]["video_sha256"]
    assert {Path(value["path"]).name for value in manifest["label_sources"]} == {
        "val.csv", "questions.json", "subset_answers.json"}
    assert runner.shared.controls_unchanged(manifest)


def test_library_preflight_reads_no_credentials_and_manifest_is_exclusive(
    tmp_path, monkeypatch, capsys,
):
    monkeypatch.setattr(runner.base, "ROOT", tmp_path)
    monkeypatch.setattr(runner.base, "PROMPT_VERSION", runner.VERSION)
    monkeypatch.setattr(runner.base, "read_owner_key", lambda: pytest.fail("credentials read"))
    monkeypatch.setattr(runner.base, "UsageGuard", lambda *_a, **_k: pytest.fail("admission"))
    monkeypatch.setattr(runner, "prepare_manifest", lambda: {"cases": []})
    monkeypatch.setattr(runner.shared, "controls_unchanged", lambda _: True)
    assert runner.main(["--prepare"]) == 0
    assert not json.loads(capsys.readouterr().out)["model_called"]
    frozen = (tmp_path / "manifest.json").read_bytes()
    with pytest.raises(runner.base.EvaluationError, match="no overwrite"):
        runner.main(["--prepare"])
    assert (tmp_path / "manifest.json").read_bytes() == frozen
    assert runner.main([]) == 0
    dry = json.loads(capsys.readouterr().out)
    assert dry["cases"] == 4 and dry["max_usd"] == 1
    assert not dry["credential_read"] and not dry["model_called"]
    assert not (tmp_path / "usage.sqlite3").exists()
    assert not (tmp_path / "budget.sqlite3").exists()
