"""V7 wrapper-only isolation/version/preflight control; no inference."""

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v7_wrapper_preserves_inherited_guards_and_isolates_version_and_preflight(
    tmp_path, monkeypatch, capsys,
):
    previous = load("gpt54_v6_isolation_fixture", "evaluate_gpt54_library.py")
    wrapper = load("gpt54_v7_wrapper_fixture", "evaluate_gpt54_v7_library.py")
    base, runner = wrapper.base, wrapper.runner
    assert previous.VERSION == previous.base.VERSION == "structured-evidence-v6"
    assert previous.ROOT != wrapper.ROOT == REPO / "artifacts/gpt54_v7_library_2026-09-08"
    assert base.VERSION == runner.VERSION == runner.library.VERSION == wrapper.VERSION
    assert base.Budget is runner.Budget and base.LoggedVLM is runner.LoggedVLM
    assert runner.shared.final_state is runner.final_state
    assert base.GPT41_MINI_MODEL == "gpt-5.4-2026-03-05"
    assert base.MAX_USD == 2 and runner.accounting.RESERVATION_USD == 0.19
    assert base.LIMITS == {"max_runs": 4, "max_requests": 100, "max_requests_per_run": 25}
    assert "scripts/evaluate_gpt54_v7_library.py" in base.HELPERS
    assert "scripts/evaluate_gpt54_v7_library.py" not in previous.base.HELPERS

    def forbidden(*_args, **_kwargs):
        pytest.fail("Wrapper preparation/preflight must not admit inference or read credentials")

    monkeypatch.setattr(base, "ROOT", tmp_path / "v7-run")
    monkeypatch.setattr(runner, "_prepare_manifest", lambda: {
        "cases": [{"id": name} for name in runner.library.EXAMPLE_IDS],
        "historical_files_sha256": {},
    })
    monkeypatch.setattr(runner.shared, "controls_unchanged", lambda _manifest: True)
    monkeypatch.setattr(base, "sources", lambda: {"fixture": "source-hash"})
    for name in ("read_owner_key", "CPUEmbedder", "UsageGuard", "Budget", "LoggedVLM", "run"):
        monkeypatch.setattr(base, name, forbidden)
    monkeypatch.setattr(base, "PROMPT_VERSION", "structured-evidence-v6")
    with pytest.raises(base.EvaluationError):
        wrapper.main(["--prepare"])
    assert not base.ROOT.exists()
    monkeypatch.setattr(base, "PROMPT_VERSION", wrapper.VERSION)
    assert wrapper.main(["--prepare"]) == 0
    assert json.loads(capsys.readouterr().out)["model_called"] is False
    manifest = json.loads((base.ROOT / "manifest.json").read_text())
    assert manifest["intended_prompt_version"] == "structured-evidence-v7"
    assert manifest["model"] == base.GPT41_MINI_MODEL and manifest["reasoning_effort"] == "none"
    assert len(manifest["cases"]) == 4
    assert wrapper.main([]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["dry_run"] and not result["credential_read"] and not result["model_called"]
    assert result["prompt_version"] == wrapper.VERSION
    assert set(path.name for path in base.ROOT.iterdir()) == {"manifest.json"}
