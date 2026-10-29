"""Gradio UI for the video pipeline; browser inference is not imported."""

from __future__ import annotations

import copy
import html
import json
import os
import time
from contextlib import ExitStack
from pathlib import Path

import gradio as gr

from videomasys.zerogpu.core import SceneCache, prepare_video
from videomasys.zerogpu.editing_ui import build_editor
from videomasys.zerogpu.library import thumbnail_css, thumbnail_data_uri
from videomasys.zerogpu.presentation import (
    gallery_items,
    render_answer,
    render_evidence,
    render_flow,
    render_metrics,
)
from videomasys.zerogpu.ui_behavior import EVIDENCE_JS, FLOW_JS
from videomasys.zerogpu.ui_style import STYLE, THEME  # noqa: F401 -- public launch configuration


def status_panel(detail: str, state: str = "running") -> str:
    headings = {
        "idle": "Ready to analyze",
        "running": "Analysis in progress",
        "complete": "Analysis complete",
        "error": "Analysis stopped",
    }
    return (
        f'<div class="run-status" data-state="{state}" role="status" aria-live="polite">'
        '<span class="indicator" aria-hidden="true"></span><div>'
        f'<strong>{headings[state]}</strong><span class="detail">{html.escape(detail)}</span>'
        "</div></div>"
    )


IDLE_STATUS = status_panel("Choose a video and question, then click Run analysis.", "idle")


def read_library(root: Path) -> list[dict]:
    """Optional owner-provided media; never reads outside the allowlisted folder."""
    manifest = root / "library.json"
    if not manifest.is_file():
        return []
    result = []
    for item in json.loads(manifest.read_text()):
        path = (root / item["file"]).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError("Invalid library path; files must stay inside the library folder.")
        preview = thumbnail_data_uri(root, item["thumbnail"]) if item.get("thumbnail") else None
        result.append({**item, "path": str(path), "thumbnail_data": preview})
    return result


def display(bundle):
    trace = copy.deepcopy(bundle.get("trace", []))
    if bundle.get("inference"):
        trace.append({"stage": "Hosted inference usage", **copy.deepcopy(bundle["inference"])})
    # Internal absolute paths are not useful in a downloadable trace.
    for step in trace:
        if step.get("stage") == "Retrieval":
            for scene in step["output"].get("selected", []):
                scene["frames"] = [Path(p).name for p in scene["frames"]]
    gallery = gallery_items(bundle)
    return render_answer(bundle), gallery, trace


def result_updates(bundle):
    """Additive UI outputs; the first four API outputs keep their original order."""
    bundle = bundle or {"trace": []}
    answer, gallery, trace = display(bundle)
    return (
        answer,
        gallery,
        trace,
        bundle,
        gr.update(visible=bool(answer or gallery)),
        gr.update(visible=bool(gallery)),
        gr.update(visible=bool(trace)),
        render_metrics(bundle),
        render_flow(bundle),
        render_evidence(bundle),
    )


def example_label(item):
    parts = item["title"].split(" · ")
    detail = parts[1] if len(parts) > 1 else parts[0]
    return f"{detail}\n{item.get('dataset', 'Example')} · {parts[-1]}"


def build_demo(pipeline=None, index_gpu=None, qa_gpu=None, *, cache_root=None, library_root=None):
    preview = pipeline is None
    root = Path(cache_root or os.environ.get("DEMO_CACHE_DIR", "/tmp/videomasys-scenes"))
    root.mkdir(parents=True, exist_ok=True)
    library = read_library(Path(library_root or "space_library"))

    def choose(index):
        if index is None:
            return None, "", "", *clear_results(), gr.update(open=False)
        item = library[int(index)]
        return (
            item["path"],
            item.get("question", ""),
            "\n".join(item.get("options", [])),
            *clear_results(),
            gr.update(open=bool(item.get("options"))),
        )

    def clear_results():
        return IDLE_STATUS, *result_updates(None)

    def run(video, question, choices, previous):
        bundle = None
        started = time.monotonic()
        session = ExitStack()
        try:
            if preview:
                raise ValueError("UI preview only: no model is loaded and inference is disabled.")
            if not video:
                raise ValueError("Upload a video or choose one from the library.")
            question = (question or "").strip()
            options = [line.strip() for line in (choices or "").splitlines() if line.strip()]
            if not 2 <= len(question) <= 2000:
                raise ValueError("Enter a question between 2 and 2000 characters.")
            if options and (not 2 <= len(options) <= 5 or any(len(x) > 300 for x in options)):
                raise ValueError(
                    "Enter 2–5 answer choices, one per line, at most 300 characters each."
                )
            if len(set(options)) != len(options):
                raise ValueError("Answer choices must be distinct.")
            yield status_panel("Reading the video and extracting frames."), *result_updates(None)
            bundle = prepare_video(Path(video), root)
            bundle["options"] = options
            bundle["question"] = question
            bundle["video_name"] = Path(video).name
            active_pipeline = pipeline
            remote = hasattr(pipeline, "analysis_session")
            if remote:
                active_pipeline = session.enter_context(pipeline.analysis_session(bundle))
            cache = SceneCache(bundle["database"], pipeline.signature)
            cached = cache.load(bundle["scenes"])
            if cached:
                bundle["scenes"], _ = cached
                bundle["cached"] = True
            else:
                yield (
                    status_panel("Generating scene captions for the video cache."),
                    *result_updates(bundle),
                )
                if remote:
                    active_pipeline.index(bundle)
                    active_pipeline.save_index(bundle)
                else:
                    bundle = index_gpu(bundle)
                if bundle.get("error"):
                    raise ValueError(bundle["error"])
            bundle["trace"].append(
                {
                    "stage": "Video cache",
                    "status": "complete",
                    "cache_hit": bundle["cached"],
                    "segmentation": "uniform temporal windows; not learned scene boundaries",
                    "scene_count": len(bundle["scenes"]),
                    "scenes": [
                        {k: s[k] for k in ("index", "start", "end", "caption")}
                        for s in bundle["scenes"]
                    ],
                    "cache_version": pipeline.signature,
                }
            )
            yield status_panel("Planning the evidence search."), *result_updates(bundle)
            updates = (active_pipeline.qa(bundle, question, options) if remote
                       else qa_gpu(bundle, question, options))
            if hasattr(updates, "close"):
                session.callback(updates.close)
            for update in updates:
                bundle = update
                if bundle.get("error"):
                    raise ValueError(bundle["error"])
                stage = bundle["trace"][-1]["stage"]
                next_stage = {
                    "Planner": "Compiling the plan into tool calls.",
                    "Tool agent": "Searching the video cache.",
                    "Retrieval": "Inspecting the retrieved frames.",
                    "Evidence inspection": "Comparing the answer choices."
                    if options
                    else "Writing an answer from the evidence.",
                    "Choice selection": "Explaining the selected answer.",
                    "Answer": "Checking the answer against the evidence.",
                    "Verifier": "Finalizing the result.",
                }.get(stage, f"{stage} complete.")
                if bundle["trace"][-1].get("status") == "failed":
                    next_stage = f"{stage} stopped; checking the failure details."
                if (
                    stage == "Verifier"
                    and bundle.get("verification", {}).get("needs_revision")
                    and not bundle["trace"][-1].get("revision")
                ):
                    review = bundle.get("verification", {})
                    next_stage = (
                        "Refining the explanation using review feedback."
                        if review.get("review_scope") == "answer_and_explanation"
                        and review.get("revision_scope") == "explanation"
                        else "Correcting the answer using review feedback."
                    )
                bundle["elapsed_seconds"] = round(time.monotonic() - started, 1)
                yield status_panel(next_stage), *result_updates(bundle)
            session.close()
            status = "Model check passed" if bundle.get("status") == "verified" else "Needs review"
            state = "complete"
            review = bundle.get("verification", {})
            if review.get("review_scope") == "answer_and_explanation":
                complete = (bundle.get("final_answer_complete") is True
                            and bundle.get("review_complete") is True
                            and not bundle.get("retained_previous_draft"))
                if not complete:
                    status, state = "Answer or review did not complete; draft only", "error"
                elif bundle.get("resolution") == "insufficient_evidence":
                    status = "Insufficient evidence"
                elif review.get("accepted") is not True:
                    status = "Answer needs review"
                elif review.get("explanation_accepted") is not True:
                    status = "Answer checked · explanation needs review"
            bundle["elapsed_seconds"] = round(time.monotonic() - started, 1)
            yield (
                status_panel(f"{status} · {time.monotonic() - started:.1f}s total", state),
                *result_updates(bundle),
            )
        except Exception as exc:
            bundle = bundle or {"trace": []}
            # Release the run before yielding its final failure view. Gradio may
            # not advance this generator again after a client disconnects.
            cleanup_failed = False
            try:
                session.close()
            except Exception:
                # A cleanup failure must not hide the original analysis error
                # or expose transport internals to the browser.
                cleanup_failed = True
            failed_step = next(
                (step for step in reversed(bundle["trace"])
                 if step.get("status") == "failed"),
                {},
            )
            failed_stage = failed_step.get("stage")
            error = {"stage": "Run", "status": "failed", "error": str(exc)}
            if cleanup_failed:
                error["cleanup_warning"] = "Analysis cleanup did not complete."
            if failed_stage:
                error["failed_stage"] = failed_stage
            bundle["trace"].append(error)
            bundle["elapsed_seconds"] = round(time.monotonic() - started, 1)
            detail = f"{failed_stage}: {exc}" if failed_stage else str(exc)
            yield status_panel(detail, "error"), *result_updates(bundle)
            gr.Warning(detail)
        finally:
            session.close()

    def finish(current_status):
        # Also runs after queue/server errors, so the button never stays disabled.
        if 'data-state="running"' in (current_status or ""):
            current_status = status_panel("The request did not finish. Please try again.", "error")
        return (
            gr.update(value="Run analysis", interactive=not preview),
            current_status,
            *[gr.update(interactive=True) for _ in controls],
        )

    with gr.Blocks(title="VideoMASys", delete_cache=(3600, 86400)) as demo:
        model_name = (
            pipeline.model_id.split("@", 1)[0].split("/")[-1].split(":", 1)[0]
            if pipeline else "UI preview"
        )
        gr.HTML(
            '<header class="app-header"><div class="brand"><div class="brand-mark">V</div>'
            '<div><h1>VideoMASys</h1><div class="eyebrow">'
            "Understand & edit video</div></div></div>"
            f'<div class="model-badge">{html.escape(model_name)}</div></header>',
            apply_default_css=False,
        )
        if preview:
            gr.Markdown("**UI preview — live analysis is disabled.**")
        mode = gr.Radio(["Ask", "Edit"], value="Ask", show_label=False, container=False,
                        elem_id="task-mode")
        if library:
            # Keep native radio semantics, keyboard access, and run-time locking.
            # The images are decorative; the complete title remains the accessible label.
            with gr.HTML(
                html_template="@children",
                css_template=thumbnail_css(library),
                apply_default_css=False,
                js_on_load=None,
                elem_id="example-cards",
            ):
                select = gr.Radio(
                    [example_label(item) for item in library],
                    type="index",
                    label="Try an example",
                    value=None,
                    elem_id="example-library",
                    elem_classes=["with-thumbnails"]
                    if any(item.get("thumbnail_data") for item in library)
                    else [],
                )
        with gr.Row(elem_id="workspace") as workspace:
            with gr.Column(scale=6, elem_id="video-column", min_width=280):
                gr.HTML(
                    '<div class="section-heading"><span class="number">01</span>'
                    "<h2>Video</h2></div>",
                    apply_default_css=False,
                )
                video = gr.Video(
                    label="Video",
                    show_label=False,
                    sources=["upload"],
                    height=300,
                    elem_id="video-input",
                    buttons=[],
                )
                gr.HTML(
                    '<p class="small-note">Upload MP4 or WebM · Up to 3 min · 100 MiB</p>',
                    apply_default_css=False,
                )
                if hasattr(pipeline, "analysis_session"):
                    provider_label = html.escape(
                        getattr(pipeline, "provider_label", "the provider")
                    )
                    gr.HTML(
                        '<p class="small-note">Analysis sends sampled frames and your question '
                        f'to {provider_label}.</p>',
                        apply_default_css=False,
                    )
            with gr.Column(scale=5, elem_id="question-column", min_width=280):
                gr.HTML(
                    '<div class="section-heading"><span class="number">02</span>'
                    "<h2>Question</h2></div>",
                    apply_default_css=False,
                )
                question = gr.Textbox(
                    label="Question",
                    show_label=False,
                    lines=4,
                    max_lines=8,
                    placeholder="What would you like to understand about this video?",
                    elem_id="question-input",
                )
                with gr.Accordion(
                    "Answer choices · optional", open=False, elem_id="choices-panel"
                ) as choices_panel:
                    choices = gr.Textbox(label="One choice per line", lines=5)
                run_button = gr.Button(
                    "Run analysis", variant="primary", elem_id="run", interactive=not preview
                )
                status = gr.HTML(IDLE_STATUS, elem_id="status", apply_default_css=False)
        with gr.Column(visible=False, elem_id="results") as results:
            answer = gr.HTML(elem_id="answer", apply_default_css=False)
            metrics = gr.HTML(elem_id="run-metrics", apply_default_css=False)
        with gr.Accordion(
            "Inspect the pipeline", open=True, visible=False, elem_id="run-details"
        ) as details:
            gr.HTML(
                '<p class="small-note">Actual outputs from this run. '
                "Expand a stage to see its results.</p>",
                apply_default_css=False,
            )
            flow = gr.HTML(
                elem_id="pipeline-flow",
                apply_default_css=False,
                autoscroll=False,
                js_on_load=FLOW_JS,
            )
            with gr.Accordion("Raw prompts & responses", open=False):
                trace = gr.JSON(label="Complete run trace", elem_id="raw-trace")
        with gr.Column(visible=False, elem_id="evidence-section") as evidence_section:
            gr.HTML(
                '<div class="section-heading"><h2>Evidence</h2></div>'
                '<p class="small-note">Six sampled frames per selected window. '
                "Click a frame to enlarge.</p>",
                apply_default_css=False,
            )
            gallery = gr.Gallery(
                label="Retrieved evidence",
                show_label=False,
                columns=6,
                height="auto",
                object_fit="contain",
                interactive=False,
                buttons=[],
                elem_id="evidence-gallery",
            )
            evidence = gr.HTML(
                elem_id="evidence-observations",
                apply_default_css=False,
                autoscroll=False,
                js_on_load=EVIDENCE_JS,
            )
        editor_panel = build_editor(pipeline, video, mode, select if library else None, root)
        gr.HTML(
            '<footer class="app-footer"><span>VideoMASys · Video workspace</span>'
            "<span>Sparse-frame evidence · Hugging Face</span></footer>",
            apply_default_css=False,
        )
        session = gr.State(value=None, time_to_live=3600)

        def show_mode(value, current):
            editing = value == "Edit"
            current = current or {}
            has_evidence = bool(gallery_items(current))
            return (
                gr.update(visible=not editing), gr.update(visible=editing),
                gr.update(visible=not editing and bool(render_answer(current) or has_evidence)),
                gr.update(visible=not editing and bool(current.get("trace"))),
                gr.update(visible=not editing and has_evidence),
            )

        mode.change(show_mode, [mode, session],
                    [workspace, editor_panel, results, details, evidence_section],
                    queue=False, api_visibility="private", show_progress="hidden")
        run_outputs = [
            status,
            answer,
            gallery,
            trace,
            session,
            results,
            evidence_section,
            details,
            metrics,
            flow,
            evidence,
        ]
        controls = [video, question, choices, mode] + ([select] if library else [])
        requested = status_panel("Request sent — waiting for the analysis to start.")
        # Client-side outputs acknowledge the click before any server/queue round trip.
        begin_js = "() => " + json.dumps(
            [
                {"__type__": "update", "value": "Running analysis…", "interactive": False},
                requested,
                *[{"__type__": "update", "interactive": False} for _ in controls],
                {"__type__": "update", "visible": False},
                {"__type__": "update", "visible": False},
                {"__type__": "update", "visible": False},
            ]
        )
        begin_event = run_button.click(
            fn=None,
            inputs=[],
            outputs=[run_button, status, *controls, results, details, evidence_section],
            js=begin_js,
            queue=False,
            api_visibility="private",
            show_progress="hidden",
        )
        analysis_event = begin_event.then(
            run,
            [video, question, choices, session],
            run_outputs,
            concurrency_limit=1,
            concurrency_id="analysis",
            api_name="analyze",
            show_progress="hidden",
        )
        analysis_event.then(
            finish,
            status,
            [run_button, status, *controls],
            queue=False,
            api_visibility="private",
            show_progress="hidden",
        )
        if library:
            select.change(
                choose,
                select,
                [video, question, choices, *run_outputs, choices_panel],
                api_visibility="private",
                queue=False,
                cancels=[analysis_event],
            )
        for component in (question, choices):
            # Gradio 6 Textbox.input misses paste/delete in some frontend builds.
            # change also catches those edits; library-driven changes should clear too.
            component.change(
                clear_results,
                None,
                run_outputs,
                queue=False,
                show_progress="hidden",
                api_visibility="private",
            )
        # Changes include replacement, removal, and edits made by Gradio's trim
        # controls. Interactive-only locking does not change the video value.
        video.change(
            clear_results,
            None,
            run_outputs,
            queue=False,
            show_progress="hidden",
            api_visibility="private",
        )
    demo.queue(max_size=4, default_concurrency_limit=1)
    return demo
