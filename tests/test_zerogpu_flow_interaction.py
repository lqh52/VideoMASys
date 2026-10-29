"""Interaction contracts; live browser checks are in smoke_zerogpu_ui.py."""

import re

from videomasys.zerogpu.presentation import render_flow
from videomasys.zerogpu.ui_behavior import EMBED_HEIGHT_JS, EVIDENCE_JS, FLOW_JS


def test_step_keys_remain_unique_and_stable_as_trace_appends():
    entries = [
        {"stage": "Caption scene 0", "status": "complete"},
        {"stage": "Caption scene 1", "status": "complete"},
        {"stage": "Planner", "status": "complete"},
        {"stage": "Answer", "status": "complete"},
        {"stage": "Verifier", "status": "complete"},
        {"stage": "Answer", "status": "complete", "revision": 1},
        {"stage": "Verifier", "status": "complete", "revision": 1},
    ]
    previous = []
    for length in range(1, len(entries) + 1):
        rendered = render_flow({"trace": entries[:length]})
        keys = re.findall(r'data-step-key="([^"]+)"', rendered)
        assert keys[: len(previous)] == previous
        assert len(keys) == len(set(keys))
        assert rendered.count('data-flow-action="expand"') == 1
        assert rendered.count('data-flow-action="collapse"') == 1
        previous = keys
    assert len(previous) == 6
    assert render_flow(None) == ""
    assert render_flow({"trace": []}) == ""


def test_pipeline_uses_client_state_without_autoscroll(tmp_path, monkeypatch):
    monkeypatch.setenv("GRADIO_TEMP_DIR", str(tmp_path / "gradio"))
    from videomasys.zerogpu.ui import build_demo

    demo = build_demo(cache_root=tmp_path / "cache", library_root=tmp_path / "library")
    flow = next(
        component
        for component in demo.config["components"]
        if component["props"].get("elem_id") == "pipeline-flow"
    )
    assert flow["props"]["js_on_load"] == FLOW_JS
    assert flow["props"]["autoscroll"] is False
    assert flow["props"].get("max_height") is None
    assert "watch('value', restoreSteps)" in FLOW_JS
    assert "states.clear()" in FLOW_JS
    evidence = next(
        c for c in demo.config["components"]
        if c["props"].get("elem_id") == "evidence-observations"
    )
    assert evidence["props"]["js_on_load"] == EVIDENCE_JS
    assert evidence["props"]["autoscroll"] is False
    demo.close()


def test_embed_notifications_follow_real_content_changes_not_iframe_height():
    assert FLOW_JS.startswith(EMBED_HEIGHT_JS)
    assert EVIDENCE_JS.startswith(EMBED_HEIGHT_JS)
    assert "window.parent === window" in EMBED_HEIGHT_JS
    assert "requestAnimationFrame" in EMBED_HEIGHT_JS
    assert EMBED_HEIGHT_JS.index("trigger('gradio_expand')") < EMBED_HEIGHT_JS.index(
        "element.dataset.layoutRevision ="
    )
    assert "window.innerWidth === lastWidth" in EMBED_HEIGHT_JS
    assert "innerHeight" not in EMBED_HEIGHT_JS
    assert "parentIFrame" not in EMBED_HEIGHT_JS  # Gradio still owns the size protocol.
    assert "setInterval" not in EMBED_HEIGHT_JS


def test_normal_step_outputs_do_not_create_nested_vertical_scrollbars(tmp_path, monkeypatch):
    monkeypatch.setenv("GRADIO_TEMP_DIR", str(tmp_path / "gradio"))
    from videomasys.zerogpu.ui_style import STYLE

    rule = re.search(r"\.flow-body pre \{([^}]+)\}", STYLE).group(1)
    assert "max-height:none" in rule
    assert "overflow:visible" in rule


def test_pipeline_precedes_evidence_without_hidden_results_parent(tmp_path, monkeypatch):
    monkeypatch.setenv("GRADIO_TEMP_DIR", str(tmp_path / "gradio"))
    from videomasys.zerogpu.ui import build_demo

    demo = build_demo(cache_root=tmp_path / "cache", library_root=tmp_path / "library")
    config = demo.get_config_file()
    ids = {c["props"].get("elem_id"): c["id"] for c in config["components"]}
    # Independent siblings: planner/error outputs appear before any answer or evidence.
    siblings = [child["id"] for child in config["layout"]["children"]]
    assert siblings.index(ids["results"]) < siblings.index(ids["run-details"])
    assert siblings.index(ids["run-details"]) < siblings.index(ids["evidence-section"])
    demo.close()
