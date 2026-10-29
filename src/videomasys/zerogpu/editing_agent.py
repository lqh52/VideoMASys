"""Evidence-grounded rough-cut proposals; no QA answers or executable model code."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from videomasys.zerogpu.core import LivePipeline
from videomasys.zerogpu.editing import validate_document
from videomasys.zerogpu.media_sampling import sample_scene_frames
from videomasys.zerogpu.structured_pipeline import inspect_evidence, public_frames
from videomasys.zerogpu.structured_schema import bounded_schema
from videomasys.zerogpu.svp_retrieval import SVPPlan

EDIT_PROMPT_VERSION = "editing-evidence-v2"
BOUNDARY_NOTE = (
    "Proposed cuts use sampled seek times, not frame-perfect event boundaries. "
    "Preview and adjust the trim handles before exporting."
)


class _ClipProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    start_frame_id: str
    end_frame_id: str
    label: str = Field(min_length=1, max_length=80)
    mute: bool
    frame_ids: list[str] = Field(min_length=1, max_length=4)
    observation_ids: list[str] = Field(max_length=3)
    need_ids: list[str] = Field(min_length=1, max_length=3)
    support: str = Field(min_length=1, max_length=220)


class _CutProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    status: Literal["supported", "unresolved"]
    clips: list[_ClipProposal] = Field(max_length=3)
    reason: str = Field(max_length=220)


def _validate_plan(plan):
    if plan.scope == "whole_video" and any(need.depends_on for need in plan.evidence_needs):
        raise ValueError(
            "Whole-video representative needs are independent. For an event-relative edit, "
            "use local_event and preserve genuine temporal dependencies instead of deleting them."
        )


def _validate_cuts(proposal, frames, observations, need_ids, maximum):
    if proposal.status == "unresolved":
        if proposal.clips or not proposal.reason.strip():
            raise ValueError("An unresolved edit needs no clips and a specific reason.")
        return proposal
    if not proposal.clips:
        raise ValueError("A supported edit needs at least one visually supported clip.")
    by_frame = {frame.id: frame for frame in frames}
    by_observation = {observation.id: observation for observation in observations}
    spans, errors, total = [], [], 0.0
    for number, clip in enumerate(proposal.clips, 1):
        prefix = f"Clip {number}"
        if not clip.label.strip() or not clip.support.strip():
            errors.append(f"{prefix}: label and support must be nonempty.")
        for values in (clip.frame_ids, clip.observation_ids, clip.need_ids):
            if len(values) != len(set(values)):
                errors.append(f"{prefix}: repeated evidence references.")
        if any(need not in need_ids for need in clip.need_ids):
            errors.append(f"{prefix}: unknown evidence need.")
        unknown = [frame for frame in [clip.start_frame_id, clip.end_frame_id, *clip.frame_ids]
                   if frame not in by_frame]
        if unknown:
            errors.append(f"{prefix}: {','.join(unknown)} must refer to actual active frames.")
        if clip.start_frame_id not in by_frame or clip.end_frame_id not in by_frame:
            continue
        start, end = by_frame[clip.start_frame_id], by_frame[clip.end_frame_id]
        if start.scene != end.scene or end.seek_seconds - start.seek_seconds < 0.1 - 1e-9:
            errors.append(f"{prefix}: boundaries need at least 0.1s in one inspected window.")
        references = [("frame_ids", frame) for frame in clip.frame_ids]
        for observation_id in clip.observation_ids:
            if observation_id not in by_observation:
                errors.append(f"{prefix}: unknown/superseded observation {observation_id}.")
                continue
            observation = by_observation[observation_id]
            if not set(observation.need_ids) & set(clip.need_ids):
                errors.append(f"{prefix}: {observation_id} does not address its evidence need.")
            references.extend((observation_id, frame) for frame in observation.frame_ids)
        outside = []
        for owner, frame in dict.fromkeys(references):
            value = by_frame.get(frame)
            if (value is None or value.scene != start.scene
                    or not start.seek_seconds <= value.seek_seconds < end.seek_seconds):
                position = f"@{value.seek_seconds:.3f}s" if value else "(inactive)"
                outside.append(f"{owner}:{frame}{position}")
        if outside:
            errors.append(
                f"{prefix} [{start.seek_seconds:.3f},{end.seek_seconds:.3f})s: "
                "outside/exclusive-end "
                f"references {', '.join(outside)}. Omit broad optional observations and cite "
                "accurate interior frames; do not widen only to retain an observation."
            )
        span = (start.seek_seconds, end.seek_seconds)
        if any(span[0] < other[1] and other[0] < span[1] for other in spans):
            errors.append(f"{prefix}: overlaps another proposed source clip.")
        spans.append(span)
        total += max(0.0, span[1] - span[0])
    if total > maximum + 1e-6:
        errors.append(f"Duration exceeds maximum by {total - maximum:.3f}s; shorten or omit clips.")
    if errors:
        raise ValueError(f"Total {total:.3f}s / {maximum:g}s maximum. " + " ".join(errors))
    return proposal


def _cut_schema(frames, observations, need_ids):
    ids = [frame.id for frame in frames]
    schema = bounded_schema(
        _CutProposal, frames=ids, observations=[item.id for item in observations], needs=need_ids,
    )
    for field in ("start_frame_id", "end_frame_id"):
        schema["$defs"]["_ClipProposal"]["properties"][field]["enum"] = ids
    return schema


def propose_edit(
    pipeline: LivePipeline, bundle: dict, instruction: str, target_seconds: float = 30,
):
    """Yield actual retrieval stages, then a validated, non-destructive edit document.

    The caller owns the model/admission session and must index this source first.
    The duration is a maximum, not a quota to fill with unrelated footage.
    No model is called by manual timeline editing or export.
    """
    if not isinstance(instruction, str) or not 2 <= len(instruction.strip()) <= 2000:
        raise ValueError("Describe the edit in 2–2000 characters.")
    if (isinstance(target_seconds, bool) or not isinstance(target_seconds, (int, float))
            or not math.isfinite(target_seconds) or not 1 <= target_seconds <= 60):
        raise ValueError("The AI edit maximum must be between 1 and 60 seconds.")
    if not re.fullmatch(r"[0-9a-f]{64}", str(bundle.get("key", ""))):
        raise ValueError("Editing requires the original video's content hash.")
    if pipeline.prompt_profile != "structured":
        raise ValueError("AI editing requires the structured evidence profile.")
    duration = bundle.get("duration")
    if (isinstance(duration, bool) or not isinstance(duration, (int, float))
            or not math.isfinite(duration) or not 0.1 <= duration <= 180):
        raise ValueError("Editing requires validated source duration between 0.1 and 180 seconds.")
    instruction, maximum = instruction.strip(), float(target_seconds)
    bundle.pop("edit_document", None)
    bundle["edit_proposal"] = {
        "status": "running", "instruction": instruction, "maximum_seconds": maximum,
        "prompt_version": EDIT_PROMPT_VERSION, "boundary_note": BOUNDARY_NOTE,
        "model": pipeline.model_id,
    }
    bundle["edit_caveats"] = [BOUNDARY_NOTE]
    trace = bundle.setdefault("trace", [])
    try:
        context = (
            "Editing task, not QA. Find the visual moments needed for this requested edit.\n"
            f"Instruction: {instruction}\nMaximum final length: {maximum:g} seconds.\n"
        )
        prompt = (
            "Plan evidence for a short video edit; do not answer a question or invent moments. "
            "The instruction is a search request, not video evidence. Use one to three short "
            "visual evidence needs with meaningful IDs. For a named action or object use, "
            "use local_event; matching occurrences may be anywhere in the video. Use real "
            "before/after dependencies only when the requested sequence needs them. "
            "For highlights of the overall activity, use whole_video and independent "
            "representative needs. Retrieval finds candidates, not precise cut boundaries. "
            "Set max_intervals=3 and max_inspected_seconds=60, option_strategy=null. "
            "Stop when matching moments are visible or cannot be established. "
            "Keep fields concise. Return the plan JSON.\n" + context
        )
        plan = pipeline.structured(
            trace, "Edit planner", prompt, SVPPlan, tokens=650, validate=_validate_plan,
        )
        trace[-1]["edit_prompt_version"] = EDIT_PROMPT_VERSION
        yield bundle
        retrieval = yield from pipeline._retrieve_svp(bundle, context, plan)
        if bundle.get("video_path") and bundle.get("frame_directory"):
            bundle["selected"] = [sample_scene_frames(
                Path(bundle["video_path"]), Path(bundle["frame_directory"]), scene, count=6,
            ) for scene in bundle["selected"]]
        frames, observations, graph = inspect_evidence(
            pipeline, bundle, instruction, plan, retrieval,
        )
        if refinement := trace[-1].get("refinement"):
            refinement.pop("same_frames_for_inspection_answer_review", None)
            refinement["same_frames_for_inspection_edit_proposal"] = True
        yield bundle
        if not frames:
            raise ValueError("No inspected frames are available; no edit was proposed.")
        needs = [need.id for need in plan.evidence_needs]
        evidence = {
            "needs": [need.model_dump() for need in plan.evidence_needs],
            "frames_in_image_order": public_frames(frames),
            "observations": graph["observations"], "unresolved": graph["unresolved"],
        }
        prompt = (
            "Propose a rough cut matching the edit instruction from these actual images. "
            "Cached captions and model observations may be wrong. Inspect the pixels; "
            "do not invent unseen actions. Return status=supported only for matching moments. "
            "Otherwise return status=unresolved, clips=[] and a specific reason. "
            "Choose up to three nonoverlapping KEEP clips in requested output order. "
            "For each, choose increasing start_frame_id/end_frame_id from the SAME inspected "
            "scene. Cite at least one frame inside the clip, any accurate observation IDs "
            "(or []), matching need IDs, and concise visual support. All cited observation "
            "frames must also lie inside the clip. The end frame is exclusive: cited support "
            "must occur before it. Prefer frame-specific support for tight cuts. Observation "
            "IDs are optional: omit a broad scene observation whose frames cross the cut, "
            "rather than lengthening the clip to include that entire observation. The cited "
            "interior frames and support statement must still show the requested moment. "
            "Use only the supplied IDs. "
            "Include enough visible lead-in and follow-through to understand the moment, "
            "without unrelated padding. Cuts are approximate sampled positions, not exact "
            "event boundaries. A shorter edit is fine; total duration must not exceed the "
            "maximum. Keep source audio (mute=false) unless the instruction asks to mute. "
            "Labels describe visible content, not unsupported outcomes.\n"
            + context + "Evidence: " + json.dumps(evidence, separators=(",", ":"))
        )
        proposal = pipeline.structured(
            trace, "Edit proposal", prompt, _CutProposal, [frame.path for frame in frames],
            tokens=1100, validate=lambda value: _validate_cuts(
                value, frames, observations, needs, maximum,
            ), response_schema=_cut_schema(frames, observations, needs),
        )
        trace[-1].update(edit_prompt_version=EDIT_PROMPT_VERSION, boundary_note=BOUNDARY_NOTE)
        if proposal.status != "supported":
            raise ValueError(f"No supported edit: {proposal.reason}")
        by_frame = {frame.id: frame for frame in frames}
        document = {"version": "1.0", "source_id": bundle["key"], "title": "", "clips": [
            {"id": f"clip_{index + 1}", "start": by_frame[clip.start_frame_id].seek_seconds,
             "end": by_frame[clip.end_frame_id].seek_seconds, "label": clip.label.strip(),
             "mute": clip.mute}
            for index, clip in enumerate(proposal.clips)
        ]}
        document = validate_document(document, bundle["key"], duration).model_dump()
        bundle["edit_document"] = document
        bundle["edit_proposal"].update(status="complete", evidence=proposal.model_dump())
        trace[-1]["resolved_edit_document"] = document
        trace[-1]["controller_transform"] = "actual frame IDs to sampled seek times; fixed clip IDs"
        yield bundle
    except Exception as exc:
        bundle["edit_proposal"].update(status="failed", error=str(exc))
        raise
