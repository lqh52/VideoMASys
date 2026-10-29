"""Single-video editing UI; server-owned source identity and model-free rendering."""

from __future__ import annotations

import copy
import html
import threading
import time
from contextlib import ExitStack
from pathlib import Path

import gradio as gr

from videomasys.zerogpu.core import SceneCache, prepare_video
from videomasys.zerogpu.editing import initial_document, render_edit, validate_document
from videomasys.zerogpu.editing_agent import propose_edit
from videomasys.zerogpu.editor_component import CSS, JS, TEMPLATE, editor_value
from videomasys.zerogpu.ui_behavior import EMBED_HEIGHT_JS

_SESSIONS: dict[str, dict] = {}
_SESSIONS_LOCK = threading.RLock()
_SESSION_LIMIT = 128
_SESSION_TTL = 3600


def _session(request):
    """Caller holds the lock; this bounded registry is authoritative, not client state."""
    key = request.session_hash
    if not isinstance(key, str) or not 1 <= len(key) <= 128:
        raise ValueError("An active browser session is required.")
    now = time.monotonic()
    for old_key, old in list(_SESSIONS.items()):
        if now - old["touched"] > _SESSION_TTL:
            if old["job"]:
                old["job"]["cancel"].set()
            del _SESSIONS[old_key]
    if key not in _SESSIONS:
        if len(_SESSIONS) >= _SESSION_LIMIT:
            raise ValueError("The editor is busy. Please try again later.")
        _SESSIONS[key] = {"generation": 0, "current": None, "job": None, "touched": now}
    entry = _SESSIONS[key]
    entry["touched"] = now
    return entry


def _active(request, job):
    with _SESSIONS_LOCK:
        entry = _session(request)
        return entry["job"] is job and entry["generation"] == job["generation"]


def _start_job(request, kind, generation):
    with _SESSIONS_LOCK:
        job = _session(request)["job"]
        if (
            not job
            or job["kind"] != kind
            or job["started"]
            or job["done"]
            or (job["generation"] != generation)
        ):
            return None
        job["started"] = True
        return job


def _complete_job(request, job):
    with _SESSIONS_LOCK:
        if _active(request, job):
            job["done"] = True


def edit_status(detail, state="idle"):
    return (
        f'<div class="edit-status" data-state="{state}" role="status" aria-live="polite">'
        f"{html.escape(str(detail))}</div>"
    )


def checked_timeline(value, state):
    """Ignore client media URLs, duration and paths; validate only the edit list."""
    if not state or not state.get("bundle"):
        raise ValueError("Choose a video first.")
    if not isinstance(value, dict):
        raise ValueError("The timeline was invalid. Reload the video to continue.")
    bundle = state["bundle"]
    return validate_document(value.get("document"), bundle["key"], bundle["duration"])


def build_editor(pipeline, source_video, mode, library_select, cache_root):
    """Add an independent editor; the existing video input remains the shared source."""
    ai_enabled = pipeline is not None and hasattr(pipeline, "analysis_session")
    cache_root = Path(cache_root)
    output_root = cache_root.parent / (cache_root.name + "-edits")
    blank = editor_value(None, source_url="", duration=0)
    with gr.Column(visible=False, elem_id="editing-workspace") as panel:
        with gr.Row():
            gr.Markdown("### Edit your video\nKeep the moments that matter. Adjust every cut.")
            upload = gr.UploadButton(
                "Choose video",
                file_types=[".mp4", ".webm", ".mov", ".mkv", ".avi"],
                file_count="single",
                size="sm",
                scale=0,
                min_width=135,
            )
        with gr.Row(equal_height=False):
            with gr.Column(scale=8, min_width=300):
                timeline = gr.HTML(
                    value=blank,
                    html_template=TEMPLATE,
                    css_template=CSS,
                    js_on_load=EMBED_HEIGHT_JS
                    + JS
                    + "\nwatch('value', refreshEmbedHeight); refreshEmbedHeight();",
                    apply_default_css=False,
                    elem_id="cut-editor",
                    autoscroll=False,
                )
            with gr.Column(scale=3, min_width=260, elem_id="edit-actions"):
                gr.Markdown("### Find clips with AI")
                instruction = gr.Textbox(
                    label="What should we keep?",
                    lines=3,
                    max_lines=6,
                    placeholder="Keep the moments when the boy unwraps his present",
                    elem_id="edit-instruction",
                )
                maximum = gr.Slider(5, 60, value=30, step=5, label="Maximum length · seconds")
                suggest = gr.Button(
                    "Find clips", variant="primary", interactive=ai_enabled, elem_id="edit-suggest"
                )
                gr.Markdown(
                    "Finds up to three clips using cached scenes and sampled frames. "
                    "Replaces the timeline with a rough cut you can adjust."
                )
                message = gr.HTML(
                    edit_status("Choose a library video or upload your own."),
                    elem_id="edit-status",
                    apply_default_css=False,
                )
                gr.Markdown("### Preview & export")
                with gr.Row():
                    preview_button = gr.Button("Render preview", size="sm")
                    export_button = gr.Button("Export MP4", variant="primary", size="sm")
                stop = gr.Button("Cancel render", visible=False, size="sm")
                download = gr.DownloadButton("Download MP4", visible=False)
                gr.Markdown(
                    "MP4 · 720p · Original audio unless muted. Your source video is never changed."
                )
        with gr.Accordion("Rendered preview", open=False, visible=False) as rendered_panel:
            rendered = gr.Video(
                label="Rendered sequence", interactive=False, buttons=[], height=360
            )
        with gr.Accordion("How AI chose the clips", open=False):
            caveats = gr.Markdown("Run Find clips to inspect its actual evidence and tool calls.")
            trace = gr.JSON(label="Editing pipeline · actual outputs")
        state = gr.State(None, time_to_live=3600)

    controls = [upload, instruction, maximum, preview_button, export_button, mode]
    if library_select is not None:
        controls.append(library_select)

    def value_for(current, *, locked=False):
        if not current:
            return {**blank, "locked": locked, "generation": 0}
        value = editor_value(
            current["document"],
            source_url=current["source_url"],
            duration=current["bundle"]["duration"],
            thumbnails=current["thumbnails"],
            locked=locked,
        )
        return {**value, "generation": current["generation"]}

    def unlocked():
        return [
            gr.update(interactive=ai_enabled),
            *[gr.update(interactive=True) for _ in controls],
            gr.update(visible=False),
        ]

    def load_source(path, current_mode, current, request: gr.Request):
        if current_mode != "Edit":
            return (gr.skip(),) * (10 + len(controls))
        with _SESSIONS_LOCK:
            entry = _session(request)
            current = entry["current"]
            if path and current:
                source = Path(path)
                try:
                    signature = (
                        str(source.resolve()),
                        source.stat().st_size,
                        source.stat().st_mtime_ns,
                    )
                    if tuple(current["source_stat"]) == signature:
                        return (gr.skip(),) * (10 + len(controls))
                except OSError:
                    pass
            if entry["job"]:
                entry["job"]["cancel"].set()
            entry["generation"] += 1
            generation = entry["generation"]
            job = {
                "kind": "load",
                "generation": generation,
                "cancel": threading.Event(),
                "started": True,
                "done": False,
            }
            entry["job"] = job
            entry["current"] = None
        current = None
        detail = "Choose a library video or upload your own."
        try:
            if path:
                source = Path(path)
                signature = (
                    str(source.resolve()),
                    source.stat().st_size,
                    source.stat().st_mtime_ns,
                )
                bundle = prepare_video(source, cache_root)
                document = initial_document(bundle["key"], bundle["duration"])
                thumbs = []
                for scene in bundle["scenes"]:
                    for frame, stamp in zip(scene["frames"], scene["frame_times"], strict=True):
                        thumbs.append(
                            {"time": stamp, "url": timeline.serve_static_file(frame)["url"]}
                        )
                current = {
                    "bundle": bundle,
                    "document": document,
                    "source_stat": signature,
                    "source_url": timeline.serve_static_file(source)["url"],
                    "thumbnails": thumbs,
                    "generation": generation,
                }
                detail = "Ready. Trim manually or find clips with AI."
        except Exception:
            detail = "Could not open this video. Use an MP4 or WebM, up to 3 minutes and 100 MiB."
        with _SESSIONS_LOCK:
            if not _active(request, job):
                return (gr.skip(),) * (10 + len(controls))
            entry["current"], entry["job"] = current, None
        value = {**value_for(current), "generation": generation}
        return (
            value,
            current,
            edit_status(detail, "error" if path and not current else "idle"),
            None,
            gr.update(visible=False),
            gr.update(value=None, visible=False),
            "",
            [],
            *unlocked(),
        )

    source_outputs = [timeline, state, message, rendered, rendered_panel, download, caveats, trace]
    for event in (mode.change, source_video.change):
        event(
            load_source,
            [source_video, mode, state],
            [*source_outputs, suggest, *controls, stop],
            concurrency_limit=1,
            concurrency_id="edit-state",
            trigger_mode="always_last",
            api_visibility="private",
            show_progress="hidden",
        )
    upload.upload(
        lambda path: path,
        upload,
        source_video,
        queue=False,
        api_visibility="private",
        show_progress="hidden",
    )

    def commit(value, current, request: gr.Request):
        with _SESSIONS_LOCK:
            entry = _session(request)
            if (
                entry["job"]
                or not isinstance(value, dict)
                or (value.get("generation") != entry["generation"])
            ):
                return (gr.skip(),) * 6
            current = entry["current"]
        try:
            document = checked_timeline(value, current)
            current = {**current, "document": document.model_dump()}
            with _SESSIONS_LOCK:
                if entry["job"] or entry["generation"] != value["generation"]:
                    return (gr.skip(),) * 6
                entry["current"] = current
            return (
                gr.skip(),
                current,
                edit_status("Timeline updated · preview and export use these cuts."),
                None,
                gr.update(visible=False),
                gr.update(value=None, visible=False),
            )
        except ValueError:
            with _SESSIONS_LOCK:
                if entry["job"] or entry["generation"] != value["generation"]:
                    return (gr.skip(),) * 6
            return (
                value_for(current),
                current,
                edit_status("That edit was invalid; restored the last valid timeline.", "error"),
                gr.skip(),
                gr.skip(),
                gr.skip(),
            )

    timeline.commit(
        commit,
        [timeline, state],
        source_outputs[:6],
        concurrency_limit=1,
        concurrency_id="edit-state",
        trigger_mode="always_last",
        api_visibility="private",
        show_progress="hidden",
    )

    def begin(value, current, text, seconds, request: gr.Request, *, kind):
        # Keep the browser's just-completed manual edit, even if commit is still in flight.
        with _SESSIONS_LOCK:
            entry = _session(request)
            if entry["job"]:
                raise gr.Error("An editor operation is already running. Wait or cancel the render.")
            if not isinstance(value, dict) or value.get("generation") != entry["generation"]:
                raise gr.Error("The source changed. Wait for the current video to load.")
            current = entry["current"]
            try:
                document = checked_timeline(value, current)
                current = {**current, "document": document.model_dump()}
                entry["current"] = current
            except ValueError:
                # Preserve the submitted document so the worker rejects it. Never
                # replace an invalid cut with the previous valid cut and export that.
                pass
            entry["generation"] += 1
            if current:
                current = {**current, "generation": entry["generation"]}
                entry["current"] = current
            entry["job"] = {
                "kind": kind,
                "generation": entry["generation"],
                "cancel": threading.Event(),
                "started": False,
                "done": False,
                "value": copy.deepcopy(value),
                "current": copy.deepcopy(current),
                "text": text,
                "seconds": seconds,
            }
            payload = {
                **value_for(current, locked=True),
                "document": copy.deepcopy(value.get("document")),
                "generation": entry["generation"],
            }
        return (
            payload,
            current,
            edit_status("Request received…", "running"),
            gr.update(interactive=False),
            *[gr.update(interactive=False) for _ in controls],
            gr.update(visible=kind in {"preview", "export"}),
        )

    def finish(value, current_message, request: gr.Request):
        with _SESSIONS_LOCK:
            entry = _session(request)
            if (
                not entry["job"]
                or (entry["job"]["started"] and not entry["job"]["done"])
                or not isinstance(value, dict)
                or value.get("generation") != entry["generation"]
            ):
                return (gr.skip(),) * (4 + len(controls))
            entry["job"] = None
            payload = {**value_for(entry["current"]), "generation": entry["generation"]}
        if 'data-state="running"' in (current_message or ""):
            current_message = edit_status("The request did not finish. You can try again.", "error")
        return (
            payload,
            current_message,
            *unlocked(),
        )

    def find_clips(value, current, text, seconds, request: gr.Request):
        job = _start_job(
            request, "find", value.get("generation") if isinstance(value, dict) else None
        )
        if job is None:
            yield (gr.skip(),) * 8
            return
        value, current, text, seconds = (
            job[key] for key in ("value", "current", "text", "seconds")
        )
        bundle = None
        started = time.monotonic()
        stack = ExitStack()
        try:
            if not ai_enabled:
                raise ValueError("AI clip selection is unavailable in this local UI preview.")
            document = checked_timeline(value, current)
            if not isinstance(text, str) or not 2 <= len(text.strip()) <= 2000:
                raise ValueError("Describe the clips to keep in 2–2000 characters.")
            current = {**current, "document": document.model_dump()}
            bundle = copy.deepcopy(current["bundle"])
            bundle["trace"] = []
            active = stack.enter_context(pipeline.analysis_session(bundle))
            cache = SceneCache(bundle["database"], pipeline.signature)
            cached = cache.load(bundle["scenes"])
            if cached:
                bundle["scenes"], _ = cached
                bundle["cached"] = True
            else:
                if not _active(request, job):
                    return
                yield (
                    gr.skip(),
                    current,
                    edit_status("Indexing scenes for clip search…", "running"),
                    [],
                    "",
                    None,
                    gr.update(visible=False),
                    gr.update(value=None, visible=False),
                )
                if not _active(request, job):
                    return
                active.index(bundle)
                active.save_index(bundle)
            if not _active(request, job):
                return
            bundle["trace"].append(
                {
                    "stage": "Video cache",
                    "status": "complete",
                    "cache_hit": bool(cached),
                    "scene_count": len(bundle["scenes"]),
                }
            )
            updates = propose_edit(active, bundle, text.strip(), seconds)
            stack.callback(updates.close)
            for bundle in updates:
                if not _active(request, job):
                    return
                stage = bundle["trace"][-1]["stage"]
                yield (
                    gr.skip(),
                    current,
                    edit_status(f"{stage} complete…", "running"),
                    clean_trace(bundle),
                    "",
                    None,
                    gr.update(visible=False),
                    gr.update(value=None, visible=False),
                )
            stack.close()
            document = validate_document(bundle["edit_document"], bundle["key"], bundle["duration"])
            current = {**current, "document": document.model_dump()}
            with _SESSIONS_LOCK:
                if not _active(request, job):
                    return
                _session(request)["current"] = current
            summary = (
                f"{len(document.clips)} clips ready · {time.monotonic() - started:.1f}s. "
                "Preview and adjust the cuts."
            )
            yield (
                value_for(current, locked=True),
                current,
                edit_status(summary, "complete"),
                clean_trace(bundle),
                " ".join(bundle.get("edit_caveats", [])),
                None,
                gr.update(visible=False),
                gr.update(value=None, visible=False),
            )
        except Exception as exc:
            stack.close()
            if not _active(request, job):
                return
            # Expected contract/budget messages are safe; do not expose transport details.
            detail = (
                str(exc) if isinstance(exc, ValueError) else "The clip search did not complete."
            )
            yield (
                gr.skip(),
                current,
                edit_status(f"{detail} Timeline unchanged.", "error"),
                clean_trace(bundle or {}),
                "No new AI cut was applied.",
                gr.skip(),
                gr.skip(),
                gr.skip(),
            )
        finally:
            stack.close()
            _complete_job(request, job)

    def render(value, current, request: gr.Request, *, preview=False):
        job = _start_job(
            request,
            "preview" if preview else "export",
            value.get("generation") if isinstance(value, dict) else None,
        )
        if job is None:
            yield (gr.skip(),) * 5
            return
        value, current, cancel = job["value"], job["current"], job["cancel"]
        try:
            document = checked_timeline(value, current)
            if not document.clips:
                raise ValueError("Add at least one clip before rendering.")
            if not _active(request, job):
                return
            yield (
                edit_status("Rendering preview…" if preview else "Exporting MP4…", "running"),
                gr.skip(),
                gr.skip(),
                gr.update(value=None, visible=False),
                gr.update(visible=True),
            )
            path = render_edit(
                Path(current["bundle"]["video_path"]),
                document,
                output_root,
                preview=preview,
                cancel_event=cancel,
            )
            if not _active(request, job):
                return
            yield (
                edit_status(
                    "Preview ready." if preview else "Export ready to download.", "complete"
                ),
                str(path),
                gr.update(visible=True, open=True),
                gr.update(
                    value=str(path),
                    visible=True,
                    label="Download preview · 360p" if preview else "Download MP4 · 720p",
                ),
                gr.update(visible=False),
            )
        except InterruptedError:
            if not _active(request, job):
                return
            yield (
                edit_status("Render canceled. Your timeline is unchanged."),
                gr.skip(),
                gr.skip(),
                gr.update(value=None, visible=False),
                gr.update(visible=False),
            )
        except Exception as exc:
            if not _active(request, job):
                return
            detail = (
                str(exc)
                if isinstance(exc, (ValueError, RuntimeError))
                else "The render did not complete."
            )
            yield (
                edit_status(detail, "error"),
                gr.skip(),
                gr.skip(),
                gr.update(value=None, visible=False),
                gr.update(visible=False),
            )
        finally:
            _complete_job(request, job)

    def render_preview(value, current, request: gr.Request):
        yield from render(value, current, request, preview=True)

    def stop_render(request: gr.Request):
        with _SESSIONS_LOCK:
            entry = _session(request)
            job = entry["job"]
            if not job or job["kind"] not in {"preview", "export"}:
                return (gr.skip(),) * (4 + len(controls))
            job["cancel"].set()
            entry["job"] = None
            value = {**value_for(entry["current"]), "generation": entry["generation"]}
        return value, edit_status("Render canceled. Your timeline is unchanged."), *unlocked()

    def starter(kind):
        def start(value, current, text, seconds, request: gr.Request):
            return begin(value, current, text, seconds, request, kind=kind)

        start.__name__ = "begin_" + kind
        return start

    for kind, button, handler, inputs, outputs, concurrency in (
        (
            "find",
            suggest,
            find_clips,
            [timeline, state, instruction, maximum],
            [timeline, state, message, trace, caveats, rendered, rendered_panel, download],
            "analysis",
        ),
        (
            "preview",
            preview_button,
            render_preview,
            [timeline, state],
            [message, rendered, rendered_panel, download, stop],
            "edit-render",
        ),
        (
            "export",
            export_button,
            render,
            [timeline, state],
            [message, rendered, rendered_panel, download, stop],
            "edit-render",
        ),
    ):
        started = button.click(
            starter(kind),
            [timeline, state, instruction, maximum],
            [timeline, state, message, suggest, *controls, stop],
            concurrency_limit=1,
            concurrency_id="edit-state",
            api_visibility="private",
            show_progress="hidden",
        )
        running = started.success(
            handler,
            inputs,
            outputs,
            concurrency_limit=1,
            concurrency_id=concurrency,
            api_visibility="private",
            show_progress="hidden",
        )
        running.then(
            finish,
            [timeline, message],
            [timeline, message, suggest, *controls, stop],
            concurrency_limit=1,
            concurrency_id="edit-state",
            api_visibility="private",
            show_progress="hidden",
        )
    stop.click(
        stop_render,
        None,
        [timeline, message, suggest, *controls, stop],
        queue=False,
        api_visibility="private",
        show_progress="hidden",
    )
    return panel


def clean_trace(bundle):
    """Downloadable provenance, without internal media/cache filesystem paths."""

    def clean(value, key=""):
        if isinstance(value, dict):
            return {name: clean(item, name) for name, item in value.items()}
        if isinstance(value, list):
            return [clean(item, key) for item in value]
        if (
            isinstance(value, str)
            and key
            in {
                "path",
                "video_path",
                "database",
                "frame_directory",
                "frames",
                "images",
                "image_paths",
            }
            and Path(value).is_absolute()
        ):
            return Path(value).name
        return value

    trace = clean(bundle.get("trace", []))
    if bundle.get("inference"):
        trace.append({"stage": "Hosted inference usage", **clean(bundle["inference"])})
    return trace
