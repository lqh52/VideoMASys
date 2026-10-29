"""Local Gradio editor integration smoke: real browser/media, no model calls."""

import argparse
import json
import subprocess
from pathlib import Path
from urllib.parse import urlsplit

from playwright.sync_api import expect, sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8211")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if urlsplit(args.url).hostname not in {"127.0.0.1", "localhost"}:
        parser.error("Only loopback UI preview servers are allowed.")
    args.output.mkdir(parents=True, exist_ok=True)
    expect.set_options(timeout=30000)
    errors = []
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1080})
        page.set_default_timeout(30000)
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(args.url, wait_until="domcontentloaded")
        expect(page.locator(".model-badge")).to_contain_text("UI preview")
        page.locator("#task-mode").get_by_text("Edit", exact=True).click()
        page.locator("#example-library label").first.click()
        expect(page.locator(".ve-count")).to_contain_text("1 clip ·")
        editor = page.locator(".ve")

        # Test the untrimmed full-video edges through real hit testing, including
        # the start edge underneath the initial playhead position.
        rail = editor.locator('[data-ui="source-track"]')
        rail.scroll_into_view_if_needed()
        original_rail = rail.bounding_box()
        original_thumbnails = editor.locator('[data-ui="source-thumbnails"] img').evaluate_all(
            "nodes => nodes.map(n => n.src)")
        assert original_rail and original_thumbnails
        for edge, fraction, mask in (("start", .12, "trim-before"), ("end", -.15, "trim-after")):
            handle = editor.locator(f'.ve-handle[data-edge="{edge}"]').bounding_box()
            assert handle
            x, y = handle["x"] + handle["width"] / 2, handle["y"] + handle["height"] / 2
            assert page.evaluate(
                "([x,y]) => document.elementFromPoint(x,y)?.closest('[data-edge]')?.dataset.edge",
                [x, y],
            ) == edge
            page.mouse.move(x, y)
            page.mouse.down()
            page.mouse.move(x + fraction * original_rail["width"], y, steps=8)
            assert abs(rail.bounding_box()["width"] - original_rail["width"]) < 1
            page.mouse.up()
            assert editor.locator(f'[data-ui="{mask}"]').bounding_box()[
                "width"] > original_rail["width"] * .1
            assert abs(rail.bounding_box()["width"] - original_rail["width"]) < 1
            assert editor.locator('[data-ui="source-thumbnails"] img').evaluate_all(
                "nodes => nodes.map(n => n.src)") == original_thumbnails
        assert float(editor.locator('[data-field="start"]').input_value()) > 0
        assert float(editor.locator('[data-field="end"]').input_value()) < 90
        page.screenshot(path=str(args.output / "source-trim-light.png"), full_page=True)

        def field(name, value):
            target = editor.locator(f'[data-field="{name}"]')
            target.fill(value)
            target.press("Tab")
            expect(target).to_have_value(value)

        field("start", "55")
        field("end", "65")
        field("title", "A present for Christmas")
        editor.locator('[data-field="mute"]').check()
        editor.locator('[data-action="play"]').click()
        expect(editor.locator('[data-action="play"]')).to_have_text("Pause")
        page.wait_for_timeout(550)
        editor.locator('[data-action="play"]').click()
        editor.locator('[data-action="split"]').click()
        expect(editor.locator(".ve-clip")).to_have_count(2)
        editor.locator('[data-action="earlier"]').click()
        editor.locator('[data-action="undo"]').click()
        editor.locator('[data-action="redo"]').click()
        expect(page.locator("#edit-status")).not_to_contain_text("invalid")
        page.screenshot(path=str(args.output / "edited-light.png"), full_page=True)
        page.get_by_role("button", name="Export MP4", exact=True).click()
        expect(page.locator("#edit-status")).to_contain_text("Export ready", timeout=120000)
        expect(editor).to_have_attribute("data-locked", "false")
        with page.expect_download() as pending:
            page.get_by_role("button", name="Download MP4 · 720p", exact=True).click()
        saved = args.output / "manual-edit.mp4"
        pending.value.save_as(saved)
        metadata = json.loads(subprocess.check_output([
            "ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(saved)
        ]))
        video = next(stream for stream in metadata["streams"] if stream["codec_type"] == "video")
        assert (video["width"], video["height"]) == (1280, 720)
        assert abs(float(metadata["format"]["duration"]) - 10) < 0.15
        # Any mutation invalidates the previous export and rendered preview.
        field("label", "Updated clip")
        expect(page.get_by_role("button", name="Download MP4 · 720p", exact=True)).to_have_count(0)
        page.locator("#task-mode").get_by_text("Ask", exact=True).click()
        expect(page.locator("#workspace")).to_be_visible()
        expect(page.locator("#editing-workspace")).to_be_hidden()
        page.locator("#task-mode").get_by_text("Edit", exact=True).click()
        expect(editor.locator(".ve-clip")).to_have_count(2)
        page.locator("#example-library label").nth(2).click()
        expect(editor.locator(".ve-clip")).to_have_count(1)
        expect(editor.locator('[data-field="title"]')).to_have_value("")
        page.emulate_media(color_scheme="dark")
        page.evaluate("document.documentElement.classList.add('dark'); "
                      "document.body.classList.add('dark')")
        page.screenshot(path=str(args.output / "editor-dark.png"), full_page=True)
        page.set_viewport_size({"width": 390, "height": 844})
        page.evaluate("new Promise(resolve => requestAnimationFrame(() => "
                      "requestAnimationFrame(resolve)))")
        expect(editor.locator(".ve-clip")).to_have_count(1)
        expect(editor.locator(".ve-ruler span").first).to_be_visible()
        page.screenshot(path=str(args.output / "editor-mobile.png"), full_page=True)
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
        assert not errors, errors
        browser.close()
    report = {"result": "passed", "checks": [
        "library load", "full-source edge hit testing", "stable source rail and thumbnails",
        "gray excluded ranges", "trim", "title", "mute", "playback", "split", "reorder",
        "undo/redo",
        "720p export", "10-second duration", "stale export invalidation", "mode preservation",
        "source reset", "mobile overflow", "no JavaScript errors",
    ], "model_calls": 0}
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
