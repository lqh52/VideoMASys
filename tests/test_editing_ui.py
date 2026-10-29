"""Offline integration controls for editor admission and stale callback handling."""

import threading
from copy import deepcopy
from types import SimpleNamespace

import pytest


@pytest.fixture
def ui(tmp_path, monkeypatch):
    monkeypatch.setenv("GRADIO_TEMP_DIR", str(tmp_path / "gradio"))
    import gradio as gr

    from videomasys.zerogpu import editing_ui

    # Gradio may already be imported by a differently ordered test module.
    monkeypatch.setattr(gr.routes, "DEFAULT_TEMP_DIR", str(tmp_path / "gradio"))
    return editing_ui


@pytest.fixture
def editor(tmp_path, monkeypatch, ui):
    gr = ui.gr
    ui._SESSIONS.clear()
    source = tmp_path / "one.mp4"
    source.write_bytes(b"trusted upload")
    second = tmp_path / "two.mp4"
    second.write_bytes(b"another trusted upload")

    def prepare(path, root):
        return {
            "key": ("a" if path == source else "b") * 64,
            "duration": 3.0,
            "scenes": [],
            "video_path": str(path),
            "database": str(root / "cache.sqlite"),
        }

    monkeypatch.setattr(ui, "prepare_video", prepare)
    monkeypatch.setattr(
        gr.HTML, "serve_static_file", lambda self, path: {"url": "/file=" + str(path)}
    )
    with gr.Blocks() as app:
        video = gr.Video()
        mode = gr.Radio(["Ask", "Edit"], value="Edit")
        ui.build_editor(None, video, mode, None, tmp_path / "cache")
    functions = {}
    for fn in app.fns.values():
        call = fn.fn
        functions[call.__name__] = call
    request = SimpleNamespace(session_hash="test-session")
    loaded = functions["load_source"](str(source), "Edit", None, request)
    yield SimpleNamespace(
        fns=functions,
        app=app,
        request=request,
        source=source,
        second=second,
        value=loaded[0],
        current=loaded[1],
        prepare=prepare,
    )
    ui._SESSIONS.clear()


def test_invalid_begin_does_not_render_previous_cut_or_trust_client_state(editor, monkeypatch, ui):
    value = deepcopy(editor.value)
    value["document"]["clips"][0]["end"] = 9
    forged = {"bundle": {"video_path": "/etc/passwd"}}
    started = editor.fns["begin_export"](value, forged, "", 30, editor.request)
    assert started[0]["document"]["clips"][0]["end"] == 9
    assert started[0]["source_url"] == editor.value["source_url"]
    monkeypatch.setattr(ui, "render_edit", lambda *a, **kw: pytest.fail("Invalid edit rendered"))
    outputs = list(editor.fns["render"](started[0], forged, editor.request))
    assert len(outputs) == 1 and "within the source" in outputs[0][0]
    finished = editor.fns["finish"](started[0], outputs[0][0], editor.request)
    assert finished[0]["document"] == editor.value["document"]
    assert not finished[0]["locked"] and ui._SESSIONS["test-session"]["job"] is None


def test_busy_stale_commit_and_source_change_discard_old_render(editor, monkeypatch, ui):
    gr = ui.gr
    started = editor.fns["begin_export"](editor.value, editor.current, "", 30, editor.request)
    with pytest.raises(gr.Error, match="already running"):
        editor.fns["begin_preview"](started[0], started[1], "", 30, editor.request)
    assert all(
        item == gr.skip()
        for item in editor.fns["commit"](
            editor.value,
            editor.current,
            editor.request,
        )
    )
    pending = editor.fns["render"](started[0], started[1], editor.request)
    assert "Exporting" in next(pending)[0]
    replacement = editor.fns["load_source"](str(editor.second), "Edit", started[1], editor.request)

    def render(path, document, root, **kwargs):
        assert path == editor.source and kwargs["cancel_event"].is_set()
        return root / "old-output.mp4"

    monkeypatch.setattr(ui, "render_edit", render)
    assert list(pending) == []
    assert replacement[0]["generation"] > started[0]["generation"]
    assert all(
        item == gr.skip()
        for item in editor.fns["finish"](
            started[0],
            "old status",
            editor.request,
        )
    )
    assert ui._SESSIONS["test-session"]["current"]["bundle"]["key"] == "b" * 64


def test_pending_cancel_and_preview_label_use_immutable_admitted_source(editor, monkeypatch, ui):
    gr = ui.gr
    started = editor.fns["begin_preview"](editor.value, editor.current, "", 30, editor.request)
    stopped = editor.fns["stop_render"](editor.request)
    assert not stopped[0]["locked"] and "canceled" in stopped[1]
    assert list(editor.fns["render_preview"](started[0], started[1], editor.request)) == [
        (gr.skip(),) * 5
    ]
    started = editor.fns["begin_preview"](stopped[0], editor.current, "", 30, editor.request)
    seen = []

    def render(path, document, root, **kwargs):
        seen.append((path, document.source_id, kwargs["preview"]))
        return root / "preview.mp4"

    monkeypatch.setattr(ui, "render_edit", render)
    forged = {"bundle": {"video_path": "/etc/passwd"}}
    outputs = list(editor.fns["render_preview"](started[0], forged, editor.request))
    assert seen == [(editor.source, "a" * 64, True)]
    assert outputs[-1][3]["label"] == "Download preview · 360p"
    finished = editor.fns["finish"](started[0], outputs[-1][0], editor.request)
    assert finished[0]["document"] == editor.value["document"]


def test_late_source_load_cannot_replace_newer_source(editor, monkeypatch, ui):
    gr = ui.gr
    entered, release = threading.Event(), threading.Event()
    old = editor.source.parent / "old.mp4"
    old.write_bytes(b"old source")

    def prepare(path, root):
        if path == old:
            entered.set()
            assert release.wait(5)
        return editor.prepare(path, root)

    monkeypatch.setattr(ui, "prepare_video", prepare)
    outputs = []
    thread = threading.Thread(
        target=lambda: outputs.append(
            editor.fns["load_source"](
                str(old),
                "Edit",
                editor.current,
                editor.request,
            )
        )
    )
    thread.start()
    assert entered.wait(5)
    try:
        newer = editor.fns["load_source"](
            str(editor.second), "Edit", editor.current, editor.request
        )
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive() and all(item == gr.skip() for item in outputs[0])
    assert ui._SESSIONS["test-session"]["current"] == newer[1]


def test_queue_failure_unlocks_but_stale_worker_and_finish_cannot_take_active_job(editor, ui):
    first = editor.fns["begin_preview"](editor.value, editor.current, "", 30, editor.request)
    released = editor.fns["finish"](first[0], first[2], editor.request)
    assert not released[0]["locked"] and "did not finish" in released[1]
    second = editor.fns["begin_preview"](released[0], first[1], "", 30, editor.request)
    assert list(editor.fns["render_preview"](first[0], first[1], editor.request)) == [
        (ui.gr.skip(),) * 5
    ]
    assert all(
        item == ui.gr.skip()
        for item in editor.fns["finish"](
            first[0],
            first[2],
            editor.request,
        )
    )
    active = editor.fns["render_preview"](second[0], second[1], editor.request)
    assert "Rendering" in next(active)[0]
    assert all(
        item == ui.gr.skip()
        for item in editor.fns["finish"](
            second[0],
            second[2],
            editor.request,
        )
    )
    active.close()
    assert not editor.fns["finish"](second[0], second[2], editor.request)[0]["locked"]


def test_session_registry_is_bounded_and_expired_jobs_are_cancelled(monkeypatch, ui):
    ui._SESSIONS.clear()
    monkeypatch.setattr(ui, "_SESSION_LIMIT", 2)
    monkeypatch.setattr(ui.time, "monotonic", lambda: 0)
    with ui._SESSIONS_LOCK:
        first = ui._session(SimpleNamespace(session_hash="first"))
        cancel = threading.Event()
        first["job"] = {"cancel": cancel}
        ui._session(SimpleNamespace(session_hash="second"))
        with pytest.raises(ValueError, match="busy"):
            ui._session(SimpleNamespace(session_hash="third"))
        monkeypatch.setattr(ui.time, "monotonic", lambda: ui._SESSION_TTL + 1)
        ui._session(SimpleNamespace(session_hash="third"))
    assert cancel.is_set() and list(ui._SESSIONS) == ["third"]
    ui._SESSIONS.clear()


def test_trace_paths_redacted_recursively_without_changing_evidence_or_prompt(ui):
    original = {
        "trace": [
            {
                "stage": "Evidence inspection",
                "calls": [
                    {
                        "frame_registry": [
                            {"id": "f_real", "path": "/cache/hash/frame.jpg", "sha256": "abc"}
                        ],
                        "attempts": [
                            {
                                "prompt": "/literal user instruction must remain exact",
                                "images": ["/cache/hash/actual.jpg"],
                            }
                        ],
                    }
                ],
            }
        ]
    }
    result = ui.clean_trace(original)
    call = result[0]["calls"][0]
    assert call["frame_registry"] == [{"id": "f_real", "path": "frame.jpg", "sha256": "abc"}]
    assert call["attempts"][0]["prompt"] == "/literal user instruction must remain exact"
    assert call["attempts"][0]["images"] == ["actual.jpg"]
    assert original["trace"][0]["calls"][0]["frame_registry"][0]["path"].startswith("/")


def test_mutating_callbacks_share_serial_queue_and_begin_requires_success(editor):
    for fn in editor.app.fns.values():
        name = getattr(fn.fn, "__name__", "")
        if name in {"commit", "load_source", "finish"} or name.startswith("begin_"):
            assert fn.concurrency_id == "edit-state" and fn.concurrency_limit == 1
    dependent = [
        item for item in editor.app.config["dependencies"] if item["trigger_after"] is not None
    ]
    assert sum(item["trigger_only_on_success"] for item in dependent) == 3
