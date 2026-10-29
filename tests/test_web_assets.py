from pathlib import Path

import pytest

from videomasys.api.app import app, browser_demo, browser_demo_scenes


def test_web_application_assets_exist_and_are_linked() -> None:
    web_root = Path(__file__).parents[1] / "src" / "videomasys" / "web"
    index = (web_root / "index.html").read_text(encoding="utf-8")

    assert (web_root / "styles.css").is_file()
    assert (web_root / "app.js").is_file()
    assert (web_root / "browser-vlm-worker.js").is_file()
    assert (web_root / "browser-live-pipeline.js").is_file()
    assert 'href="/static/styles.css?' in index
    assert 'src="/static/app.js?' in index
    assert 'src="/static/browser-live-pipeline.js?' in index
    assert 'id="browserLab"' in index
    assert 'id="browserModel"' in index
    assert 'id="demoExampleSelect"' in index
    assert 'id="pipelineDetail"' in index
    assert "renderFlowDetail" in (web_root / "app.js").read_text(encoding="utf-8")


def test_browser_vlm_worker_uses_selective_supported_models() -> None:
    web_root = Path(__file__).parents[1] / "src" / "videomasys" / "web"
    worker = (web_root / "browser-vlm-worker.js").read_text(encoding="utf-8")
    app_script = (web_root / "app.js").read_text(encoding="utf-8")
    index = (web_root / "index.html").read_text(encoding="utf-8")

    assert "LiquidAI/LFM2.5-VL-3B-ONNX" in worker
    assert "onnx-community/Qwen3.5-4B-ONNX-OPT" in worker
    assert "onnx-community/LFM2.5-VL-450M-ONNX" in worker
    assert "LiquidAI/LFM2.5-VL-1.6B-ONNX" in worker
    assert (web_root / "lfm-browser-runtime.js").is_file()
    assert 'device: "webgpu"' in worker
    assert 'device: "wasm"' in worker
    assert 'value="lfm-1.6b" selected' in index
    assert "selectBrowserFrames" in app_script
    assert "buildEvidenceContactSheet" in app_script
    assert "/v1/browser-demo/examples" in app_script
    assert "if (state.browserDemo)" in app_script


def test_web_and_media_routes_are_registered() -> None:
    paths = {route.path for route in app.routes}

    assert "/" in paths
    assert "/static" in paths
    assert "/v1/videos/{video_id}/content" in paths
    assert "/v1/tasks/{task_id}/evidence/{node_id}/frames/{frame_index}" in paths
    assert "/v1/browser-demo" in paths
    assert "/v1/browser-demo/examples" in paths
    assert "/v1/browser-demo/assets/{video_key}/content" in paths
    assert (
        "/v1/browser-demo/assets/{video_key}/frames/{segment}/{frame_index}" in paths
    )
    assert "/v1/browser-demo/assets/{video_key}/scenes" in paths


@pytest.mark.asyncio
async def test_browser_demo_starts_without_fabricated_results() -> None:
    payload = await browser_demo("nextqa_gift_reason")
    task = payload["task"]

    assert task["status"] == "idle"
    assert task["plan"] is None
    assert task["result"] is None
    assert task["evidence"] == []
    assert task["steps"] == []

    scenes = await browser_demo_scenes("nextqa")
    assert len(scenes["scenes"]) == 5
    assert all(scene["frame_urls"] for scene in scenes["scenes"])
