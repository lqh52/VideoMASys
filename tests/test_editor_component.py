"""Small editor contract check and opt-in real Gradio/browser interaction smoke."""

import os
import socket
from copy import deepcopy
from pathlib import Path

import pytest

from videomasys.zerogpu.editor_component import CSS, JS, TEMPLATE, editor_value


def test_editor_payload_is_detached_and_component_does_not_interpolate_user_html():
    document = {"version": "1.0", "source_id": "a" * 64, "clips": [], "title": "<Title>"}
    thumbnails = [{"time": 0, "url": "/gradio_api/file=thumb.jpg"}]
    value = editor_value(document, source_url="/gradio_api/file=source.mp4", duration=90,
                         thumbnails=thumbnails, locked=True)
    value["document"]["title"] = "changed"
    value["thumbnails"][0]["time"] = 1
    assert document["title"] == "<Title>" and thumbnails[0]["time"] == 0
    assert value["locked"] and value["duration"] == 90 and value["generation"] == 0
    assert "${" not in TEMPLATE and "innerHTML" not in JS
    assert "trigger('commit')" in JS and "watch('value', incoming)" in JS


@pytest.mark.skipif(os.environ.get("RUN_EDITOR_BROWSER") != "1",
                    reason="Opt-in local browser smoke")
def test_editor_real_gradio_interactions(tmp_path, monkeypatch):
    monkeypatch.setenv("GRADIO_TEMP_DIR", str(tmp_path / "gradio-cache"))
    gr = pytest.importorskip("gradio")
    playwright = pytest.importorskip("playwright.sync_api")
    from videomasys.zerogpu.editing import validate_document

    video_path = Path(__file__).resolve().parents[1] / "space_library/nextqa-4882821564.mp4"
    if not video_path.is_file():
        pytest.skip("Local demo media unavailable")
    document = {"version": "1.0", "source_id": "a" * 64, "title": "First sequence", "clips": [
        {"id": "a", "start": 0.0, "end": 4.0, "label": "Opening", "mute": False},
        {"id": "b", "start": 10.0, "end": 14.0, "label": "Later", "mute": True},
    ]}
    commits = []
    with gr.Blocks() as app:
        editor = gr.HTML(value=None, html_template=TEMPLATE, css_template=CSS, js_on_load=JS)
        source_url = editor.serve_static_file(video_path)["url"]
        thumb_url = editor.serve_static_file(
            video_path.parent / "thumbnails/christmas-sofa.jpg")["url"]
        thumbnails = [{"time": 0.0, "url": thumb_url}]
        value = editor_value(document, source_url=source_url, duration=90.024,
                             thumbnails=thumbnails)
        editor.value = value
        lock = gr.Button("Test lock")
        external = gr.Button("Test new proposal")
        stale = gr.Button("Test stale response")
        quick = deepcopy(document)
        quick["title"] = "A new proposal"
        quick["clips"] = [
            {"id": "new_a", "start": 0.0, "end": 0.4, "label": "First", "mute": False},
            {"id": "new_b", "start": 15.0, "end": 15.4, "label": "Second", "mute": True},
        ]

        def committed(current):
            validate_document(current["document"], "a" * 64, 90.024)
            commits.append(deepcopy(current))
            return gr.skip()

        editor.commit(committed, editor, editor, queue=False)
        lock.click(lambda current: {**current, "locked": True}, editor, editor)
        external.click(lambda: editor_value(quick, source_url=source_url, duration=90.024,
                                            thumbnails=thumbnails, generation=1),
                       outputs=editor)
        stale.click(lambda: value, outputs=editor)
    with socket.socket() as available:
        available.bind(("127.0.0.1", 0))
        port = available.getsockname()[1]
    try:
        app.launch(server_name="127.0.0.1", server_port=port, prevent_thread_lock=True,
                   quiet=True, show_error=True)
        with playwright.sync_playwright() as runtime:
            browser = runtime.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1100, "height": 1000}, has_touch=True)
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.route("**/*", lambda route: route.continue_()
                       if route.request.url.startswith(f"http://127.0.0.1:{port}/")
                       else route.abort())
            page.goto(f"http://127.0.0.1:{port}")
            page.wait_for_function("document.querySelectorAll('.ve-clip').length === 2")
            editor_root = page.locator(".ve")
            def action(name):
                return editor_root.locator(f'[data-action="{name}"]')
            assert editor_root.locator('[data-field="title"]').input_value() == "First sequence"
            assert action("undo").is_disabled()
            editor_root.locator('[data-field="title"]').fill('<img src=x onerror="alert(1)">')
            editor_root.locator('[data-field="title"]').press("Tab")
            page.wait_for_function("document.querySelector('.ve-title-overlay').textContent.includes('<img')")
            assert editor_root.locator(".ve-title-overlay img").count() == 0
            action("undo").click()
            assert editor_root.locator('[data-field="title"]').input_value() == "First sequence"
            action("redo").click()
            assert "<img" in editor_root.locator('[data-field="title"]').input_value()
            # Numeric trim, selection and keyboard-accessible reordering.
            editor_root.locator('[data-field="start"]').fill("1.5")
            editor_root.locator('[data-field="start"]').press("Tab")
            assert editor_root.locator('[data-field="start"]').input_value() == "1.5"
            action("later").click()
            assert editor_root.locator(".ve-clip").last.get_attribute("data-clip") == "a"
            action("earlier").click()
            editor_root.locator('[data-clip="b"]').drag_to(
                editor_root.locator('[data-clip="a"]'))
            assert editor_root.locator(".ve-clip").first.get_attribute("data-clip") == "b"
            action("undo").click()
            editor_root.locator('[data-clip="a"]').press("Enter")
            editor_root.locator('[data-field="mute"]').check()
            assert editor_root.locator('[data-field="mute"]').is_checked()
            action("undo").click()
            # Source coordinates remain independent of the assembled sequence.
            head = editor_root.locator('[data-ui="playhead"]')
            head.focus()
            head.press("Home")
            assert action("split").is_disabled()  # Source 0 is excluded from this clip.
            head.press("Shift+ArrowRight")
            head.press("Shift+ArrowRight")
            action("split").click()
            assert editor_root.locator(".ve-clip").count() == 3
            action("delete").click()
            assert editor_root.locator(".ve-clip").count() == 2
            action("reset").click()
            assert editor_root.locator(".ve-clip").count() == 1
            assert editor_root.locator('[data-field="title"]').input_value() == ""

            rail = editor_root.locator('[data-ui="source-track"]')
            mask_before = editor_root.locator('[data-ui="trim-before"]')
            mask_after = editor_root.locator('[data-ui="trim-after"]')
            strip = editor_root.locator('[data-ui="source-thumbnails"]')
            rail.scroll_into_view_if_needed()
            rail_box = rail.bounding_box()
            source_thumbnails = strip.locator("img").evaluate_all("nodes => nodes.map(n => n.src)")
            assert rail_box and source_thumbnails
            touch = page.context.new_cdp_session(page)

            def wait_commits(expected):
                for _ in range(60):
                    if len(commits) >= expected:
                        break
                    page.wait_for_timeout(50)
                assert len(commits) == expected

            def source_drag(edge, fraction, *, touch_input=False, cancel=False):
                # Browser-generated input: no dispatchEvent shortcut that could hide
                # an overlapping playhead or broken native pointer capture.
                handle = editor_root.locator(f'.ve-handle[data-edge="{edge}"]')
                handle.scroll_into_view_if_needed()
                box = handle.bounding_box()
                assert box
                x, y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
                assert page.evaluate(
                    "([x,y]) => document.elementFromPoint(x,y)"
                    "?.closest('[data-edge]')?.dataset.edge",
                    [x, y],
                ) == edge
                page.wait_for_timeout(200)  # Drain any prior reset/edit callback.
                before = len(commits)
                original_start = editor_root.locator('[data-field="start"]').input_value()
                original_end = editor_root.locator('[data-field="end"]').input_value()
                final_x = x + fraction * rail_box["width"]
                if touch_input:
                    touch.send("Input.dispatchTouchEvent", {"type": "touchStart", "touchPoints": [
                        {"x": x, "y": y, "id": 7}]})
                    for step in range(1, 6):
                        touch.send("Input.dispatchTouchEvent", {
                            "type": "touchMove", "touchPoints": [
                                {"x": x + (final_x - x) * step / 5, "y": y, "id": 7}]})
                else:
                    page.mouse.move(x, y)
                    page.mouse.down()
                    page.mouse.move(final_x, y, steps=5)
                page.wait_for_timeout(150)
                assert len(commits) == before
                during = rail.bounding_box()
                assert during and abs(during["width"] - rail_box["width"]) < 1
                assert abs(during["x"] - rail_box["x"]) < 1
                assert strip.locator("img").evaluate_all(
                    "nodes => nodes.map(n => n.src)") == source_thumbnails
                if touch_input:
                    touch.send("Input.dispatchTouchEvent", {
                        "type": "touchCancel" if cancel else "touchEnd", "touchPoints": []})
                else:
                    assert not cancel
                    page.mouse.up()
                if cancel:
                    page.wait_for_timeout(200)
                    assert len(commits) == before
                    assert editor_root.locator(
                        '[data-field="start"]').input_value() == original_start
                    assert editor_root.locator('[data-field="end"]').input_value() == original_end
                else:
                    wait_commits(before + 1)
                assert abs(rail.bounding_box()["width"] - rail_box["width"]) < 1

            # Both original full-video edges are draggable, even at playhead 0.
            assert mask_before.bounding_box()["width"] < 1
            assert mask_after.bounding_box()["width"] < 1
            source_drag("start", .15)
            assert 12 < float(editor_root.locator('[data-field="start"]').input_value()) < 15
            assert mask_before.bounding_box()["width"] > rail_box["width"] * .14
            source_drag("end", -.2)
            assert 70 < float(editor_root.locator('[data-field="end"]').input_value()) < 75
            assert mask_after.bounding_box()["width"] > rail_box["width"] * .19
            for mask in (mask_before, mask_after):
                assert mask.evaluate(
                    "n => getComputedStyle(n).backgroundColor") != "rgba(0, 0, 0, 0)"
            assert editor_root.locator('[data-ui="trim-selection"]').bounding_box()[
                "width"] < rail_box["width"] * .7
            page.screenshot(path=str(tmp_path / "editor-trimmed.png"), full_page=True)
            action("undo").click()
            assert float(editor_root.locator('[data-field="end"]').input_value()) == 90.024
            action("redo").click()
            assert float(editor_root.locator('[data-field="end"]').input_value()) < 75
            # Extending outside the source clamps, it never grows/reflows the rail.
            source_drag("start", -.3)
            source_drag("end", .4)
            assert editor_root.locator('[data-field="start"]').input_value() == "0"
            assert editor_root.locator('[data-field="end"]').input_value() == "90.024"
            # A narrow touch viewport still exposes both source endpoints at once.
            page.set_viewport_size({"width": 390, "height": 844})
            page.wait_for_timeout(150)
            rail.scroll_into_view_if_needed()
            rail_box = rail.bounding_box()
            source_thumbnails = strip.locator("img").evaluate_all("nodes => nodes.map(n => n.src)")
            assert rail_box
            for edge in ("start", "end"):
                box = editor_root.locator(f'.ve-handle[data-edge="{edge}"]').bounding_box()
                assert box and box["x"] >= 0 and box["x"] + box["width"] <= 390
            source_drag("start", .15, touch_input=True, cancel=True)
            source_drag("start", .12, touch_input=True)
            source_drag("end", -.12, touch_input=True)
            assert 9 < float(editor_root.locator('[data-field="start"]').input_value()) < 12
            assert 77 < float(editor_root.locator('[data-field="end"]').input_value()) < 81
            assert all(mask.bounding_box()["width"] > 0 for mask in (mask_before, mask_after))
            page.screenshot(path=str(tmp_path / "editor-trimmed-mobile.png"), full_page=True)
            assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
            page.set_viewport_size({"width": 1100, "height": 1000})

            action("delete").click()
            assert editor_root.locator(".ve-clip").count() == 0 and action("play").is_disabled()
            action("reset").click()
            assert editor_root.locator(".ve-clip").count() == 1
            action("undo").click()
            assert editor_root.locator(".ve-clip").count() == 0
            # Server props replace a proposal and lock all mutations without hiding UI.
            page.get_by_role("button", name="Test new proposal", exact=True).click()
            page.wait_for_function("document.querySelector('.ve-title').value === 'A new proposal'")
            assert editor_root.locator(".ve-clip").count() == 2 and action("undo").is_disabled()
            page.wait_for_function("document.querySelector('.ve video').readyState > 0")
            action("play").click()
            page.wait_for_function("document.querySelector('.ve video').currentTime >= 15")
            page.wait_for_function(
                "document.querySelector('[data-action=play]').textContent === 'Play'")
            assert editor_root.locator('[data-ui="clock"]').inner_text() == "0:00.8 / 0:00.8"
            before = len(commits)
            page.get_by_role("button", name="Test stale response", exact=True).click()
            page.wait_for_timeout(250)
            assert editor_root.locator('[data-field="title"]').input_value() == "A new proposal"
            assert len(commits) == before
            editor_root.locator('[data-field="title"]').fill("Current generation")
            editor_root.locator('[data-field="title"]').press("Tab")
            page.wait_for_timeout(200)
            assert commits[-1]["generation"] == 1
            assert commits[-1]["document"]["clips"][0]["id"] == "new_a"
            page.get_by_role("button", name="Test lock", exact=True).click()
            page.wait_for_function("document.querySelector('.ve').dataset.locked === 'true'")
            assert action("reset").is_disabled() and action("play").is_disabled()
            page.screenshot(path=str(tmp_path / "editor-desktop.png"), full_page=True)
            page.set_viewport_size({"width": 390, "height": 844})
            page.screenshot(path=str(tmp_path / "editor-mobile.png"), full_page=True)
            assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
            assert errors == []
            for committed_value in commits:
                validate_document(committed_value["document"], "a" * 64, 90.024)
            print("EDITOR_SCREENSHOTS", tmp_path)
            browser.close()
    finally:
        app.close()
