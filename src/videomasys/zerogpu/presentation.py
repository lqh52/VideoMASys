"""Escaped, read-only views of real pipeline state; no models or Gradio imports."""

from __future__ import annotations

import html
import json
import math


def _text(value) -> str:
    return html.escape(str(value), quote=True)


def _paragraph(value, class_name: str = "") -> str:
    attribute = f' class="{class_name}"' if class_name else ""
    return f"<p{attribute}>{_text(value).replace(chr(10), '<br>')}</p>"


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _seconds(value) -> str:
    return f"{value:.1f}".rstrip("0").rstrip(".") + "s"


def _timestamp(value) -> str:
    # Use tenths only when present; do not silently round short spans to zero.
    tenths = round(value * 10)
    minutes, remainder = divmod(tenths, 600)
    seconds, fraction = divmod(remainder, 10)
    return f"{minutes}:{seconds:02d}" + (f".{fraction}" if fraction else "")


def _scene_label(scene: dict) -> str:
    label = f"Scene {scene.get('index', '?')}"
    if all(_number(scene.get(key)) for key in ("start", "end")):
        label += f" · {_timestamp(scene['start'])}–{_timestamp(scene['end'])}"
    return label


def gallery_items(bundle: dict | None) -> list[tuple[str, str]]:
    """Pair each real frame with its approximate seek time, without inventing times."""
    items = []
    for scene in (bundle or {}).get("selected", []):
        times = scene.get("frame_times", [])
        for index, frame in enumerate(scene.get("frames", [])):
            label = _scene_label(scene)
            if index < len(times) and _number(times[index]):
                label = f"Scene {scene.get('index', '?')} · ≈{_timestamp(times[index])}"
            items.append((frame, label))
    return items


def _code(value) -> str:
    rendered = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)
    return f'<pre class="mono"><code>{_text(rendered)}</code></pre>'


def _list(values) -> str:
    return "<ul>" + "".join(f"<li>{_text(value)}</li>" for value in values) + "</ul>"


def render_answer(bundle: dict | None) -> str:
    """Show a draft or model-checked answer without claiming independent verification."""
    bundle = bundle or {}
    answer = bundle.get("answer")
    if not answer:
        return ""
    review = bundle.get("verification", {})
    scoped = review.get("review_scope") == "answer_and_explanation"
    complete = (bundle.get("final_answer_complete") is True
                and bundle.get("review_complete") is True
                and not bundle.get("retained_previous_draft"))
    checked = bundle.get("status") == "verified" and (
        not scoped or (complete and review.get("accepted") is True
                       and review.get("explanation_accepted") is True)
    )
    badge = "Model-checked" if checked else "Draft · not verified"
    badge_class = "badge-success" if checked else "badge-warning"
    if scoped and complete and not checked:
        if review.get("accepted") is not True:
            badge = "Answer needs review"
        elif review.get("explanation_accepted") is not True:
            badge = "Answer checked · explanation needs review"
    if bundle.get("resolution") == "insufficient_evidence" and (not scoped or complete):
        badge, badge_class = "Insufficient evidence", "badge-warning"
    parts = [
        '<section class="result-card">',
        '<div class="eyebrow">Answer</div>',
        f'<span class="badge {badge_class}">{badge}</span>',
    ]
    context = []
    if bundle.get("video_name"):
        context.append(str(bundle["video_name"]))
    if bundle.get("question"):
        context.append(str(bundle["question"]))
    if context:
        parts.append(_paragraph(" · ".join(context), "muted"))
    parts.append(_paragraph(answer.get("answer", ""), "answer-text"))
    option = answer.get("selected_option")
    options = bundle.get("options", [])
    if isinstance(option, int) and not isinstance(option, bool) and 0 <= option < len(options):
        parts.append(_paragraph(f"Choice {option + 1}: {options[option]}", "muted"))
    if answer.get("inference"):
        parts.append('<div class="eyebrow">Why this fits</div>')
        parts.append(_paragraph(answer["inference"]))
    has_refinement_note = (
        review.get("explanation_repair_failed")
        and any(str(note).startswith("Explanation refinement did not complete;")
                for note in review.get("advisories", []))
    )
    if (scoped and complete and bundle.get("explanation_repair_failed")
        and bundle.get("retained_reviewed_answer") and not has_refinement_note):
        parts.append(_paragraph(
            "Explanation refinement did not complete; the reviewed answer is unchanged.", "muted",
        ))
    scenes = {scene["index"]: scene for scene in bundle.get("selected", [])}
    citations = answer.get("supporting_scenes", [])
    if citations:
        labels = [
            _scene_label(scenes[index]) if index in scenes else f"Scene {index}"
            for index in citations
        ]
        parts.append(_paragraph("Evidence: " + " · ".join(labels), "muted"))
    if scoped:
        for heading, values in (
            ("Limitations", answer.get("limitations", [])),
            ("Answer concerns", review.get("issues", [])),
            ("Explanation concerns", review.get("explanation_issues", [])),
            ("Review notes", review.get("advisories", [])),
        ):
            if values:
                parts.append(f'<div class="eyebrow">{heading}</div>')
                parts.append(_list(dict.fromkeys(values)))
    else:
        issues = list(
            dict.fromkeys(answer.get("limitations", []) + review.get("issues", []))
        )
        if issues:
            parts.append('<div class="eyebrow">Limitations & review notes</div>')
            parts.append(_list(issues))
    return "".join(parts) + "</section>"


def render_metrics(bundle: dict | None) -> str:
    """Only display measurements present in, or directly calculable from, this run."""
    bundle = bundle or {}
    metrics = []
    duration = bundle.get("duration")
    if _number(duration):
        metrics.append(("Video length", _timestamp(duration)))
    if "selected" in bundle:
        selected = bundle["selected"]
        metrics.append(("Selected scenes", str(len(selected))))
        if all(_number(scene.get("start")) and _number(scene.get("end")) for scene in selected):
            # Count the union of spans, including for traces with overlapping intervals.
            seconds, previous_end = 0.0, None
            for scene in sorted(selected, key=lambda item: item["start"]):
                start, end = scene["start"], scene["end"]
                seconds += max(
                    0, end - max(start, previous_end if previous_end is not None else start)
                )
                previous_end = max(end, previous_end if previous_end is not None else end)
            metrics.append(("Evidence spans", _seconds(seconds)))
    elif "scenes" in bundle:
        metrics.append(("Video scenes", str(len(bundle["scenes"]))))
    cache = next(
        (
            entry
            for entry in reversed(bundle.get("trace", []))
            if entry.get("stage") == "Video cache"
        ),
        {},
    )
    # prepare_video sets cached=False before checking the cache. Until Video cache
    # completes, that default is not evidence of a completed fresh index.
    if isinstance(cache.get("cache_hit"), bool):
        metrics.append(("Video cache", "Reused" if cache["cache_hit"] else "New index"))
    elif bundle.get("cached") is True:
        metrics.append(("Video cache", "Reused"))
    elapsed = bundle.get("elapsed_seconds")
    if _number(elapsed):
        metrics.append(("Elapsed", _seconds(elapsed)))
    if not metrics:
        return ""
    cards = "".join(
        f'<div class="metric"><span class="muted">{_text(label)}</span>'
        f"<strong>{_text(value)}</strong></div>"
        for label, value in metrics
    )
    timeline = ""
    if _number(duration) and duration > 0 and "selected" in bundle:
        windows = []
        for scene in bundle["selected"]:
            if not all(_number(scene.get(key)) for key in ("start", "end")):
                continue
            start = min(100.0, max(0.0, scene["start"] / duration * 100))
            end = min(100.0, max(0.0, scene["end"] / duration * 100))
            if end <= start:
                continue
            label = _text(_scene_label(scene))
            windows.append(
                f'<span class="timeline-window" role="listitem" title="{label}" '
                f'aria-label="{label}" style="left:{start:g}%;width:{end - start:g}%"></span>'
            )
        timeline = (
            '<div class="evidence-timeline"><div class="eyebrow">Retrieved windows</div>'
            '<div class="timeline-track" role="list" aria-label="Retrieved video windows">'
            + "".join(windows)
            + '</div><div class="timeline-labels"><span>0:00</span>'
            + f"<span>{_text(_timestamp(duration))}</span></div>"
            + _paragraph("Sampled frames from highlighted spans", "muted")
            + "</div>"
        )
    return f'<div class="metric-grid">{cards}</div>' + timeline


def _inspection_observations(entry: dict) -> list[dict]:
    """Retain completed real window observations if a later window fails."""
    output = entry.get("output", {})
    if "evidence" in output:
        return output["evidence"]
    return [
        call["output"]
        for call in entry.get("calls", [])
        if call.get("status") == "complete" and "scene" in call.get("output", {})
    ]


def render_evidence(bundle: dict | None) -> str:
    """Selected scene cards distinguish new visual observations from cache hints."""
    bundle = bundle or {}
    if not bundle.get("selected"):
        return ""
    inspection = next(
        (
            entry
            for entry in reversed(bundle.get("trace", []))
            if entry.get("stage") == "Evidence inspection"
        ),
        {},
    )
    observations = {item["scene"]: item for item in _inspection_observations(inspection)}
    cited = bundle.get("answer", {}).get("supporting_scenes", [])
    run_failed = any(
        entry.get("stage") == "Run" and entry.get("status") == "failed"
        for entry in bundle.get("trace", [])
    )
    parts = ['<div class="evidence-grid">']
    for scene in bundle.get("selected", []):
        parts.append('<section class="evidence-card">')
        parts.append(f'<div class="eyebrow">{_text(_scene_label(scene))}</div>')
        if scene["index"] in cited:
            parts.append('<span class="badge">Cited in answer</span>')
        observation = observations.get(scene["index"])
        if observation:
            parts.append(_paragraph("Sampled-frame observation", "muted"))
            parts.append(_paragraph(observation.get("observation", "")))
            if observation.get("uncertainty"):
                parts.append(_paragraph("Uncertainty: " + observation["uncertainty"], "muted"))
            if "relevant" in observation:
                parts.append(
                    _paragraph(
                        "Marked relevant" if observation["relevant"] else "Marked not relevant",
                        "muted",
                    )
                )
        else:
            failed = inspection.get("status") == "failed" or bool(bundle.get("error")) or run_failed
            parts.append(
                _paragraph(
                    "Inspection did not complete" if failed else "Inspection pending", "muted"
                )
            )
        if scene.get("caption"):
            parts.append(
                '<details><summary class="muted">Cached caption · search hint</summary>'
                + _paragraph(scene["caption"])
                + "</details>"
            )
        parts.append("</section>")
    return "".join(parts) + "</div>"


def _planner(output: dict) -> str:
    parts = []
    if output.get("objective"):
        parts.append(_paragraph(output["objective"]))
    for need in output.get("evidence_needs", []):
        meta = [f"ID: {need['id']}"] if "id" in need else []
        if "temporal_role" in need:
            meta.append(f"Role: {need['temporal_role']}")
        if need.get("depends_on"):
            meta.append("Depends on: " + ", ".join(map(str, need["depends_on"])))
        parts.append(
            '<div class="evidence-need">'
            + _paragraph(need.get("description", ""))
            + (_paragraph(" · ".join(meta), "muted") if meta else "")
            + "</div>"
        )
    fields = (
        ("temporal_constraints", "Temporal constraints"),
        ("option_strategy", "Choice strategy"),
        ("stop_condition", "Stop condition"),
        ("reasoning", "Planner rationale"),
    )
    for key, label in fields:
        value = output.get(key)
        if value:
            parts.append(f'<div class="eyebrow">{label}</div>')
            parts.append(_list(value) if isinstance(value, list) else _paragraph(value))
    parts.append(_budgets(output))
    return "".join(parts)


def _budgets(output: dict) -> str:
    limits = []
    if "max_intervals" in output:
        limits.append(f"{output['max_intervals']} intervals")
    if "max_inspected_seconds" in output:
        limits.append(f"{output['max_inspected_seconds']} seconds of scene spans")
    return _paragraph("Budget: " + " · ".join(limits), "muted") if limits else ""


def _typed_call(name: str, call: dict) -> str:
    assignment = f"{call['id']} = " if "id" in call else ""
    arguments = ",\n".join(
        f"  {key}={json.dumps(value, ensure_ascii=False)}"
        for key, value in call.items()
        if key not in {"id", "op"}
    )
    return f"{assignment}{name}(\n{arguments}\n)"


def _program(output: dict) -> str:
    if output.get("version") == "3.0":
        calls = [_typed_call(call.get("op", "unknown"), call)
                 for call in output.get("steps", [])]
    else:
        calls = [_typed_call("locate", call) for call in output.get("locate", [])]
        calls += [_typed_call("relate", call) for call in output.get("relate", [])]
    if isinstance(output.get("inspect"), dict):
        calls.append(_typed_call("inspect", output["inspect"]))
    parts = [_paragraph(output["rationale"])] if output.get("rationale") else []
    if calls:
        parts.append(_code("\n\n".join(calls)))
    parts.append(_budgets(output))
    return "".join(parts)


def _retrieval_hit_label(hit: dict, fallback_score_type: str) -> str:
    labels = [f"Scene {hit.get('scene', '?')}"]
    scores = (
        ("semantic_cosine", "Cosine similarity"),
        ("temporal_proximity", "Temporal proximity"),
    )
    for field, label in scores:
        if field in hit:
            labels.append(f"{label}: {hit[field]}")
    if "representative_score" in hit:
        labels.append(f"Representative score: {hit['representative_score']}")
        if hit.get("selection_method"):
            labels.append("Method: " + str(hit["selection_method"]).replace("_", " "))
    if not any(field in hit for field, _ in scores) and "score" in hit:
        labels.append(f"{fallback_score_type}: {hit['score']}")
    if hit.get("anchor_overlap") or hit.get("window_relation") == "anchor_overlap":
        labels.append("Overlaps anchor window; event order unproven")
    elif hit.get("window_relation") in {"strict_before", "strict_after"}:
        labels.append("Window " + hit["window_relation"].removeprefix("strict_") + " anchor")
    return " · ".join(labels)


def _selection_reason(reason: dict) -> str:
    prefix = (
        "Additional candidate" if reason.get("purpose") == "additional_candidate" else "Selected"
    )
    label = f"{prefix} via {reason.get('source', '?')}"
    if "rank" in reason:
        label += f" (rank {reason['rank']})"
    if reason.get("evidence_need_ids"):
        label += " · needs: " + ", ".join(map(str, reason["evidence_need_ids"]))
    if reason.get("anchor_overlap") or reason.get("window_relation") == "anchor_overlap":
        label += " · overlapping-window candidate, not confirmed event order"
    return label


def _temporal_coverage(output: dict) -> str:
    coverage = output.get("temporal_coverage")
    if not isinstance(coverage, dict):
        return ""
    positions = ("beginning", "middle", "end")
    selected = {scene.get("index"): scene for scene in output.get("selected", [])}
    representatives, used = {}, set()
    for representative in coverage.get("representatives", []):
        position, index = representative.get("position"), representative.get("scene")
        scene = selected.get(index)
        if (
            position in positions
            and position not in representatives
            and index not in used
            and scene is not None
            and all(_number(scene.get(key)) for key in ("start", "end"))
            and 0 <= scene["start"] < scene["end"]
        ):
            representatives[position] = scene
            used.add(index)
    count = len(representatives)
    complete = coverage.get("complete") is True and count == 3
    title = f"Temporal coverage: {count}/3 video thirds represented"
    if not complete:
        title = f"Temporal coverage incomplete: {count}/3 video thirds represented"
    rows = [
        f"{position.title()}: "
        + (
            _scene_label(representatives[position])
            if position in representatives
            else "No representative retained"
        )
        for position in positions
    ]
    return (
        _paragraph(title)
        + _list(rows)
        + _paragraph(
            "Representatives are assigned by window midpoint. Only sampled frames are "
            "inspected; this does not mean every event was seen.",
            "muted",
        )
    )


def _retrieval(output: dict) -> str:
    parts = [_temporal_coverage(output)]
    for item in output.get("calls", []):
        call = item.get("call")
        if isinstance(call, dict):
            operation = call.get("op", "locate" if "query" in call else "relate")
            parts.append(_code(_typed_call(operation, call)))
            if "source_scene_ids" in item:
                parts.append(_paragraph(
                    "Source candidates: " + ", ".join(map(str, item["source_scene_ids"])),
                    "muted",
                ))
            score_type = {
                "semantic_cosine": "Cosine similarity",
                "temporal_proximity": "Temporal proximity",
            }.get(
                item.get("score_type"),
                "Cosine similarity" if operation == "locate" else "Temporal proximity",
            )
            hits = item.get("hits", [])
            if hits:
                parts.append(_list(_retrieval_hit_label(hit, score_type) for hit in hits))
            else:
                parts.append(_paragraph("No hits returned.", "muted"))
        elif call == "inspect":
            parts.append(
                _paragraph("inspect → scenes " + ", ".join(map(str, item.get("selected", []))))
            )
            if "inspected_seconds" in item:
                parts.append(
                    _paragraph(
                        f"Selected scene spans: {item['inspected_seconds']} seconds", "muted"
                    )
                )
            if "budget_seconds" in item:
                parts.append(_paragraph(f"Budget: {item['budget_seconds']} seconds", "muted"))
        elif call is not None:
            parts.append(_code(item))
    reasons = {
        item["scene"]: item.get("reasons", []) for item in output.get("selection_reasons", [])
    }
    for scene in output.get("selected", []):
        parts.append(
            '<div class="evidence-need">'
            + _paragraph(_scene_label(scene))
            + _paragraph("Cached caption — search hint", "muted")
            + _paragraph(scene.get("caption", ""))
            + (
                _list(_selection_reason(reason) for reason in reasons[scene["index"]])
                if reasons.get(scene["index"])
                else ""
            )
            + "</div>"
        )
    if output.get("uncovered_evidence_needs"):
        parts.append(
            _paragraph(
                "No candidate retained for needs: "
                + ", ".join(map(str, output["uncovered_evidence_needs"])),
                "muted",
            )
        )
    if output.get("uncovered_temporal_outputs"):
        parts.append(
            _paragraph(
                "No candidate retained from temporal outputs: "
                + ", ".join(map(str, output["uncovered_temporal_outputs"])),
                "muted",
            )
        )
    if output.get("selected"):
        parts.append(
            _paragraph("Inspection uses sampled frames from these spans, not every frame.", "muted")
        )
    return "".join(parts)


def _stage_body(entry: dict) -> str:
    stage, output = entry.get("stage", ""), entry.get("output", {})
    if stage == "Planner" and output:
        return _planner(output)
    if stage in {"Tool agent", "Tool agent execution repair"} and output:
        return _program(output)
    if stage in {"Retrieval", "Retrieval execution repair"} and output:
        return _retrieval(output)
    if stage == "Video cache":
        parts = []
        if "cache_hit" in entry:
            parts.append(
                _paragraph(
                    "Existing video index reused."
                    if entry["cache_hit"]
                    else "New video index created."
                )
            )
        if entry.get("segmentation"):
            parts.append(_paragraph(entry["segmentation"], "muted"))
        for scene in entry.get("scenes", []):
            parts.append(
                '<div class="evidence-need">'
                + _paragraph(_scene_label(scene))
                + _paragraph("Cached caption — search hint", "muted")
                + _paragraph(scene.get("caption", ""))
                + "</div>"
            )
        return "".join(parts)
    if stage.startswith("Caption scene ") and "caption" in output:
        return _paragraph("Scene caption — search hint", "muted") + _paragraph(output["caption"])
    if stage == "Evidence inspection":
        return "".join(
            '<div class="evidence-need">'
            + _paragraph(f"Scene {item.get('scene', '?')} · Sampled-frame observation", "muted")
            + _paragraph(item.get("observation", ""))
            + (
                _paragraph("Uncertainty: " + item["uncertainty"], "muted")
                if item.get("uncertainty")
                else ""
            )
            + (
                _paragraph(
                    "Marked relevant" if item["relevant"] else "Marked not relevant", "muted"
                )
                if "relevant" in item
                else ""
            )
            + "</div>"
            for item in _inspection_observations(entry)
        )
    if stage == "Verifier" and output:
        parts = []
        if output.get("review_scope") == "answer_and_explanation":
            if output.get("review_complete") is False:
                parts.append(_paragraph("Review incomplete; no final verdict."))
            elif "accepted" in output:
                parts.append(_paragraph(
                    "Main answer accepted by the model reviewer."
                    if output["accepted"] else "Main answer needs review."
                ))
            for heading, values in (
                ("Answer concerns", output.get("issues", [])),
                ("Explanation concerns", output.get("explanation_issues", [])),
                ("Review notes", output.get("advisories", [])),
            ):
                if values:
                    parts.append(f'<div class="eyebrow">{heading}</div>')
                    parts.append(_list(dict.fromkeys(values)))
        elif "accepted" in output:
            parts.append(
                _paragraph(
                    "Draft accepted by the model checker."
                    if output["accepted"]
                    else "Draft rejected by the model checker."
                )
            )
        if output.get("issues") and output.get("review_scope") != "answer_and_explanation":
            parts.append(_list(output["issues"]))
        if output.get("checks"):
            parts.append('<div class="eyebrow">Claim and option checks</div>')
            for check in output["checks"]:
                parts.append(
                    '<div class="evidence-need">'
                    + _paragraph(f"{check.get('target_id', '')} · {check.get('status', '')}")
                    + _paragraph(check.get("reason", ""), "muted")
                    + "</div>"
                )
        return "".join(parts)
    if stage == "Answer" and output.get("option_comparisons"):
        parts = ['<div class="eyebrow">Model option comparisons</div>']
        for comparison in output["option_comparisons"]:
            parts.append(
                '<div class="evidence-need">'
                + _paragraph(
                    str(comparison.get("choice", ""))
                    + " · "
                    + str(comparison.get("status", "")).replace("_", " ")
                )
                + _paragraph(comparison.get("reason", ""), "muted")
                + "</div>"
            )
        parts.append(
            _code({key: value for key, value in output.items() if key != "option_comparisons"})
        )
        return "".join(parts)
    # The Answer trace is the actual structured output, distinct from its result
    # card: preserve citations, selected option, and limitations without rewriting.
    return _code(output) if output else ""


def _render_flow_entries(entries, *, nested=False, start_index=0) -> str:
    parts = []
    for index, entry in enumerate(entries, start_index):
        stage = entry.get("stage", "Unnamed stage")
        state = entry.get("status", "")
        metadata = [str(state).capitalize()] if state else []
        if entry.get("revision") == 1:
            metadata.append("Explanation refinement 1 of 1"
                            if entry.get("revision_scope") == "explanation"
                            else "Correction 1 of 1")
        latency = entry.get("latency_ms")
        if _number(latency):
            metadata.append(_seconds(latency / 1000))
        attempts = entry.get("attempts", [])
        if entry.get("calls"):
            metadata.append(f"{len(attempts)} model call" + ("s" if len(attempts) != 1 else ""))
        repairs = entry.get("repair_count")
        if not isinstance(repairs, int) or isinstance(repairs, bool) or repairs < 0:
            repairs = (
                sum(max(0, len(call.get("attempts", [])) - 1) for call in entry["calls"])
                if entry.get("calls")
                else max(0, len(attempts) - 1)
            )
        if repairs:
            metadata.append(f"{repairs} validation repair" + ("s" if repairs != 1 else ""))
        opened = " open" if state == "failed" else ""
        parts.append(
            (
                '<div class="evidence-need">'
                if nested
                else f'<details class="flow-step" data-step-key="step-{index}"{opened}>'
            )
            + ('<div class="flow-heading">' if nested else '<summary class="flow-heading">')
            + f'<span class="muted mono">{index + 1:02d}</span> '
            f"<strong>{_text(stage)}</strong> "
            f'<span class="muted">{_text(" · ".join(metadata))}</span>'
            + ("</div>" if nested else "</summary>")
            + '<div class="flow-body">'
        )
        parts.append(_stage_body(entry))
        if entry.get("error"):
            parts.append(_paragraph(entry["error"]))
        call_attempts = {}
        for attempt_index, attempt in enumerate(attempts, 1):
            if "scene" in attempt:
                group = attempt.get("model_call", attempt["scene"])
                call_attempts[group] = call_attempts.get(group, 0) + 1
                label = f"Scene {attempt['scene']} · attempt {call_attempts[group]}"
            else:
                label = f"Attempt {attempt_index}"
            for key in ("validation_error", "error"):
                if attempt.get(key):
                    parts.append(_paragraph(f"{label}: {attempt[key]}", "muted"))
        if entry.get("model"):
            parts.append(_paragraph("Model: " + str(entry["model"]), "muted"))
        if entry.get("embedding_model"):
            parts.append(_paragraph("Embedding model: " + str(entry["embedding_model"]), "muted"))
        if entry.get("index"):
            parts.append(_paragraph("Index: " + str(entry["index"]), "muted"))
        parts.append("</div>" + ("</div>" if nested else "</details>"))
    return "".join(parts)


def render_flow(bundle: dict | None) -> str:
    """Group consecutive captions; preserve real trace order, outputs and errors."""
    entries = (bundle or {}).get("trace", [])
    parts, position, step = [], 0, 0
    if entries:
        parts.append(
            '<div class="flow-controls" role="group" aria-label="Pipeline steps">'
            '<button type="button" data-flow-action="expand">Expand all</button>'
            '<button type="button" data-flow-action="collapse">Collapse all</button>'
            '</div>'
        )
    while position < len(entries):
        end = position
        while end < len(entries) and str(entries[end].get("stage", "")).startswith(
            "Caption scene "
        ):
            end += 1
        if end == position:
            parts.append(_render_flow_entries(entries[position : position + 1], start_index=step))
            position += 1
        else:
            captions = entries[position:end]
            states = {}
            for caption in captions:
                state = caption.get("status")
                if state:
                    states[state] = states.get(state, 0) + 1
            metadata = " · ".join(f"{count} {state}" for state, count in states.items())
            opened = " open" if "failed" in states else ""
            parts.append(
                f'<details class="flow-step" data-step-key="step-{step}"{opened}>'
                '<summary class="flow-heading">'
                f'<span class="muted mono">{step + 1:02d}</span> '
                f"<strong>Scene captioning · {len(captions)} "
                + ("window" if len(captions) == 1 else "windows")
                + f'</strong> <span class="muted">{_text(metadata)}</span>'
                '</summary><div class="flow-body">'
                + _render_flow_entries(captions, nested=True, start_index=position)
                + "</div></details>"
            )
            position = end
        step += 1
    return "".join(parts)
