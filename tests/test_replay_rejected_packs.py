"""Local serialization and manifest tests; no provider requests or credentials."""

import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "rejected_pack_fixture", REPO / "scripts/replay_rejected_packs.py",
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def schema():
    return {"type": "object", "additionalProperties": False, "required": ["choice"],
            "properties": {"choice": {"type": "string", "enum": ["A", "B"],
                                      "minLength": 1, "maxLength": 10}}}


def test_annotations_and_shared_local_refs_do_not_change_constraints():
    original = schema()
    annotated = deepcopy(original)
    annotated["title"] = "An annotation"
    annotated["$defs"] = {"Choice": annotated["properties"]["choice"]}
    annotated["properties"]["choice"] = {"$ref": "#/$defs/Choice"}
    assert runner.schema_constraints(original) == runner.schema_constraints(annotated)
    annotated["$defs"]["Choice"]["enum"] = ["A", "B", "C"]
    assert runner.schema_constraints(original) != runner.schema_constraints(annotated)


def test_real_local_payload_has_no_client_or_key_and_preserves_constraints(monkeypatch):
    import httpx

    monkeypatch.setattr(httpx, "Client", lambda *_a, **_k: pytest.fail("HTTP client constructed"))
    adapter = runner.payload_adapter()
    assert not hasattr(adapter, "_client") and not hasattr(adapter, "_token")
    call = {"prompt": "Select from supplied evidence.", "images": [],
            "max_output_tokens": 100, "schema": schema()}
    report = runner.replay_pack(call, 33000, schema(), adapter)
    assert report["after_admitted"] and report["prompt_preserved"]
    assert report["schema_constraints_preserved"] and report["images_preserved"]
    assert not report["model_called"] and not report["credential_read"]


def test_offline_report_writer_never_overwrites(tmp_path):
    path = tmp_path / "new/report.json"
    runner.write_exclusive(path, {"model_called": False})
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        runner.write_exclusive(path, {"changed": True})
    assert path.read_bytes() == before


def test_smoke_plan_is_exactly_nine_exposed_cases_with_no_launcher():
    if not (runner.BASELINE / "cost-audit.json").is_file():
        pytest.skip("Local completed evaluation artifacts unavailable")
    plan = runner.smoke_manifest()
    assert len(plan["cases"]) == len({case["id"] for case in plan["cases"]}) == 9
    assert plan["limits"] == {"max_runs": 9, "max_requests": 360,
                              "max_requests_per_run": 40, "max_usd": 0.5}
    assert plan["launch_enabled"] is False
    assert "NOT held-out" in plan["kind"]
    old = json.loads((runner.BASELINE / "manifest.json").read_text())
    assert len({case["id"] for case in plan["cases"]} &
               {case["id"] for case in old["cases"]}) == 5
