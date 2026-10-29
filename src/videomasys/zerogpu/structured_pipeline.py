"""Typed visual evidence and claim checking in the existing bounded QA stages.

No models, files, network clients or database services are opened at import.
The legacy and hybrid pipelines remain untouched; this path is explicitly opt-in.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from copy import deepcopy
from pathlib import Path

from videomasys.domain.visual_evidence import (
    DirectStructuredReview,
    FrameRef,
    ObservationRecord,
    StructuredAnswer,
    WindowObservation,
    answer_diagnostics,
    answer_need_coverage,
    claim_partition,
    claim_provenance,
    validate_answer,
    validate_choice_change,
    validate_direct_review,
    validate_window,
)
from videomasys.zerogpu.media_sampling import refine_scene_frames, sample_scene_frames
from videomasys.zerogpu.structured_schema import bounded_schema

SCHEMA_VERSION = "visual-evidence-v6"


def frame_registry(scene, video_id):
    """Stable IDs bind scene, seek position and actual JPEG bytes, not model timestamps."""
    paths, times = scene["frames"], scene["frame_times"]
    if not paths or len(paths) != len(times) or len(paths) > 6:
        raise ValueError("Structured inspection requires one to six aligned frames per window.")
    frames = []
    seen_pixels = set()
    for path, timestamp in zip(paths, times, strict=True):
        if (
            isinstance(timestamp, bool) or not math.isfinite(timestamp)
            or not scene["start"] <= timestamp < scene["end"]
        ):
            raise ValueError("Frame seek positions must be finite and inside their scene.")
        source = Path(path)
        if source.is_symlink() or not source.is_file() or not source.stat().st_size:
            raise ValueError("Structured evidence requires nonempty regular frame files.")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        identity = json.dumps([video_id, scene["index"], float(timestamp), digest])
        frame = FrameRef(
            id="f_" + hashlib.sha256(identity.encode()).hexdigest()[:16],
            scene=scene["index"], seek_seconds=float(timestamp),
            path=str(source), sha256=digest,
        )
        if digest not in seen_pixels:
            frames.append(frame)
            seen_pixels.add(digest)
    if list(times) != sorted(times):
        raise ValueError("Frame references must be chronological.")
    return frames


def public_frames(frames):
    return [{"id": item.id, "scene": item.scene, "approx_seconds": round(item.seek_seconds, 3)}
            for item in frames]


def needs_by_scene(plan, retrieval):
    """Preserve need IDs and descriptions; candidate lineage is not visual support."""
    needs = {need.id: {"id": need.id, "description": need.description}
             for need in plan.evidence_needs}
    result = {}
    for selection in retrieval.get("selection_reasons", []):
        ids = list(dict.fromkeys(
            need for reason in selection.get("reasons", [])
            for need in (reason.get("evidence_need_ids")
                         or reason.get("lineage_evidence_need_ids", []))
            if need in needs
        ))
        if ids:
            result[selection["scene"]] = [needs[need] for need in ids]
    return result


def _records(value, scene, phase):
    return [ObservationRecord(
        **fact.model_dump(), id=f"obs_s{scene}_{phase}_{index}", scene=scene,
    ) for index, fact in enumerate(value.facts)]


def _window_view(scene, records, unresolved, frames):
    mapping = {frame.id: frame for frame in frames}
    lines = []
    for record in records:
        times = ", ".join(f"~{mapping[frame].seek_seconds:.2f}s" for frame in record.frame_ids)
        fact = " ".join(part for part in (record.subject, record.predicate, record.object) if part)
        lines.append(f"{record.id} · {times}: {fact}")
    uncertainty = [item.reason for item in unresolved]
    uncertainty += [item.uncertainty for item in records if item.uncertainty]
    return {
        "scene": scene, "observation": "\n".join(lines), "relevant": bool(records),
        "uncertainty": " ".join(dict.fromkeys(uncertainty)),
        "observations": [item.model_dump() for item in records],
        "unresolved": [item.model_dump() for item in unresolved],
    }


def inspect_evidence(pipeline, bundle, question, plan, retrieval):
    """Same window calls and at most one refinement; all final citations see the same pixels."""
    selected = bundle["selected"]
    sought = needs_by_scene(plan, retrieval)
    all_needs = [{"id": need.id, "description": need.description} for need in plan.evidence_needs]
    entry = {
        "stage": "Evidence inspection", "status": "running", "calls": [], "attempts": [],
        "repair_count": 0, "schema_version": SCHEMA_VERSION,
        "prompt_profile": pipeline.prompt_profile, "prompt_version": pipeline.prompt_version,
        "model": pipeline.model_id,
    }
    bundle["trace"].append(entry)
    started = time.monotonic()
    windows = []

    def inspect(scene, *, refining=False):
        frames = frame_registry(scene, bundle["key"])
        needs = sought.get(scene["index"], all_needs)
        local_trace = []
        try:
            value = pipeline.structured(
                local_trace, f"{'Refine' if refining else 'Inspect'} scene {scene['index']}",
                pipeline.role_prompts.inspection_prompt(
                    question, needs, public_frames(frames), refining=refining,
                ),
                WindowObservation, [frame.path for frame in frames], tokens=650,
                validate=lambda result: validate_window(
                    result, frames, [need["id"] for need in needs], allow_refine=not refining,
                ),
                response_schema=bounded_schema(
                    WindowObservation, frames=[frame.id for frame in frames],
                    needs=[need["id"] for need in needs],
                ),
            )
            records = _records(value, scene["index"], "refined" if refining else "initial")
            local_trace[-1]["controller_records"] = [record.model_dump() for record in records]
            return {"frames": frames, "records": records, "value": value}
        finally:
            for call in local_trace:
                call.update(scene=scene["index"], evidence_sought=needs,
                            frame_registry=[frame.model_dump() for frame in frames])
                entry["calls"].append(call)
                entry["attempts"].extend(call["attempts"])
                entry["repair_count"] += max(0, len(call["attempts"]) - 1)

    def publish():
        entry["output"] = {"schema_version": SCHEMA_VERSION, "evidence": [
            _window_view(scene["index"], window["records"], window["value"].unresolved,
                         window["frames"])
            for scene, window in zip(selected, windows, strict=False)
        ]}

    try:
        for scene in selected:
            windows.append(inspect(scene))
            publish()
        requested = next((index for index, window in enumerate(windows)
                          if window["value"].refine_frame_id is not None), None)
        if requested is not None and bundle.get("video_path") and bundle.get("frame_directory"):
            old = windows[requested]
            target = next(frame for frame in old["frames"]
                          if frame.id == old["value"].refine_frame_id)
            # Inspect the exact final six files, not a different 12-frame grid.
            refined_scene = refine_scene_frames(
                Path(bundle["video_path"]), Path(bundle["frame_directory"]), selected[requested],
                target.seek_seconds, count=6, retain_context=True,
            )
            windows[requested] = inspect(refined_scene, refining=True)
            selected[requested] = refined_scene
            entry["refinement"] = {
                "scene": target.scene, "requested_frame_id": target.id,
                "requested_time": target.seek_seconds, "max_refinements": 1,
                "sample_count": len(windows[requested]["frames"]),
                "detail_span": refined_scene["detail_span"],
                "superseded_observation_ids": [record.id for record in old["records"]],
                "same_frames_for_inspection_answer_review": True,
            }
        frames = [frame for window in windows for frame in window["frames"]]
        records = [record for window in windows for record in window["records"]]
        # Keep the UI's final scene frame lists aligned with the actual model pack.
        for scene, window in zip(selected, windows, strict=True):
            scene["frames"] = [frame.path for frame in window["frames"]]
            scene["frame_times"] = [frame.seek_seconds for frame in window["frames"]]
        graph = {
            "schema_version": SCHEMA_VERSION, "video_id": bundle["key"],
            "plan": plan.model_dump(), "frames": [frame.model_dump() for frame in frames],
            "observations": [record.model_dump() for record in records],
            "retrieval_selection": retrieval.get("selection_reasons", []),
            "retrieval_coverage_scope": "candidate_lineage_not_visual_confirmation",
            "unresolved": [dict(item.model_dump(), scene=scene["index"])
                           for scene, window in zip(selected, windows, strict=True)
                           for item in window["value"].unresolved],
        }
        bundle["evidence_graph"] = graph
        publish()
        entry["status"] = "complete"
        return frames, records, graph
    except Exception:
        entry["status"] = "failed"
        raise
    finally:
        entry["latency_ms"] = round((time.monotonic() - started) * 1000)


def answer_view(draft, frames, observations):
    """Derive UI text/citations from reviewed claims; no independent summary generation."""
    claims = {claim.id: claim for claim in draft.claims}
    provenance = claim_provenance(draft, observations)
    selected = [claims[key] for key in draft.answer_claim_ids]
    conclusion = " ".join(claim.statement for claim in selected)
    decision_support = next((item.claim_ids for item in draft.option_comparisons
                             if item.choice == draft.selected_choice), draft.answer_claim_ids)
    used_frames = {
        frame for key in dict.fromkeys(
            draft.answer_claim_ids + [claim.id for claim in draft.claims
                                      if claim.kind == "limitation"]
        ) for frame in provenance[key]["frame_ids"]
    }
    return {
        **draft.model_dump(),
        **claim_partition(draft),
        "answer": conclusion,
        # One explicit pair for review: the selected option and the displayed
        # conclusion must express the same decision, not merely share frame IDs.
        "decision": {"selected_choice": draft.selected_choice, "conclusion": conclusion,
                     "support_claim_ids": list(decision_support)},
        "inference": " ".join(claim.rationale for claim in selected
                               if claim.kind == "inferred" and claim.rationale),
        "supporting_scenes": sorted({frame.scene for frame in frames if frame.id in used_frames}),
        "limitations": [claim.statement for claim in draft.claims
                        if claim.kind == "limitation" and claim.id not in draft.answer_claim_ids],
        "option_comparisons": [dict(item.model_dump(), reason=" ".join(
            claims[key].statement for key in item.claim_ids
        ) or "Not established by the sampled evidence.") for item in draft.option_comparisons],
        "claim_provenance": provenance,
    }


def need_checklist(plan, graph):
    """Index recorded observations and gaps without claiming a need is satisfied."""
    result = []
    for need in plan.evidence_needs:
        facts = [record for record in graph["observations"] if need.id in record["need_ids"]]
        gaps = [record for record in graph["unresolved"] if record["need_id"] == need.id]
        result.append({
            "need_id": need.id, "description": need.description,
            "observation_ids": [record["id"] for record in facts],
            "uncertainties": [record["uncertainty"] for record in facts if record["uncertainty"]],
            "unresolved": gaps,
            "inspection_state": "unresolved" if gaps else "observed" if facts else "uninspected",
        })
    return result


def review_verdict(review, draft, *, revision):
    """Separate answer decisions, explanation repair, and nonblocking review notes.

    A claim wording problem is not itself permission to change the answer. The
    reviewer's answer judgment must identify any effect on the actual decision.
    References and positive evidence paths have already passed strict validation.
    """
    answer_lineage = set(claim_partition(draft)["answer_lineage_ids"])
    checks = [dict(check.model_dump(), scope=(
        "explanation" if check.target_id in answer_lineage else "advisory"
    )) for check in review.claim_checks]
    for index, (assessment, option) in enumerate(zip(
        review.option_assessments, draft.option_comparisons, strict=True,
    )):
        checks.append({
            "target_id": f"option_{index}",
            "status": "pass" if assessment.status == option.status else "revise",
            "reason": assessment.reason,
            "expected_status": assessment.status,
            "draft_status": option.status,
            "scope": "answer" if option.choice == draft.selected_choice else "advisory",
        })
    judgment = review.answer_judgment.model_dump()
    if judgment["selected_choice"] != draft.selected_choice:
        judgment["status"] = "revise"
    checks.append({"target_id": "selection", **judgment, "scope": "answer"})

    def concerns(scope):
        return [f"{check['target_id']}: {check['reason']}" for check in checks
                if check["status"] != "pass" and check["scope"] == scope]

    issues, explanation_issues, advisories = (
        concerns("answer"), concerns("explanation"), concerns("advisory")
    )
    scope = "answer" if issues else "explanation" if explanation_issues else "none"
    return {
        "accepted": not issues, "issues": issues,
        "explanation_accepted": not explanation_issues, "explanation_issues": explanation_issues,
        "advisories": advisories, "review_scope": "answer_and_explanation",
        "revision_scope": scope, "needs_revision": scope != "none" and not revision,
        "checks": checks,
        "direct_assessments": review.model_dump(), "review_complete": True,
        "verdict_source": "controller_comparison_of_direct_model_assessments",
    }


def answer_retrieved(pipeline, bundle, question, options, plan, retrieval):
    bundle.update(final_answer_complete=False, review_complete=False,
                  retained_previous_draft=False, retained_reviewed_answer=False,
                  explanation_repair_failed=False)
    for key in ("answer", "verification", "resolution", "reviewed_pair", "completion_issue"):
        bundle.pop(key, None)
    if bundle.get("video_path") and bundle.get("frame_directory"):
        bundle["selected"] = [sample_scene_frames(
            Path(bundle["video_path"]), Path(bundle["frame_directory"]), scene, count=6,
        ) for scene in bundle["selected"]]
    frames, observations, graph = inspect_evidence(pipeline, bundle, question, plan, retrieval)
    yield bundle
    images = [frame.path for frame in frames]
    candidate = (
        pipeline.choose_option(bundle["trace"], question, options, images) if options else None
    )
    if options:
        bundle["trace"][-1]["candidate_is_provisional"] = True
        yield bundle
    evidence = {
        "needs": [need.model_dump() for need in plan.evidence_needs],
        "frames_in_image_order": public_frames(frames),
        "observations": graph["observations"], "unresolved": graph["unresolved"],
        "need_checklist": need_checklist(plan, graph),
    }
    graph["need_checklist"] = evidence["need_checklist"]
    previous, feedback = None, None
    last_reviewed = None
    need_ids = [need.id for need in plan.evidence_needs]
    try:
        for revision in range(2):
            bundle["review_complete"] = False
            previous_choice = previous["selected_choice"] if previous is not None else candidate
            explanation_only = bool(feedback and feedback.get("revision_scope") == "explanation")

            def validate_draft(value, *, explanation_only=explanation_only,
                               previous_choice=previous_choice):
                value = validate_answer(value, frames, observations, need_ids, options)
                if options:
                    if explanation_only and value.selected_choice != previous_choice:
                        raise ValueError(
                            "Explanation-only repair must preserve selected_choice exactly; "
                            "repair the explanation, not the answer decision."
                        )
                    value = validate_choice_change(value, previous_choice)
                return value

            schema = bounded_schema(
                StructuredAnswer, frames=[frame.id for frame in frames],
                observations=[record.id for record in observations],
                needs=need_ids, options=options,
            )
            if options and explanation_only:
                # Enforce the same decision in both native schema and runtime,
                # retaining all alternatives for explanation/option assessments.
                schema["properties"]["selected_choice"] = (
                    {"type": "null"} if previous_choice is None
                    else {"type": "string", "enum": [previous_choice]}
                )
            draft = pipeline.structured(
                bundle["trace"], "Answer",
                pipeline.role_prompts.answer_prompt(
                    question, options, candidate, evidence, draft=previous, feedback=feedback,
                ),
                StructuredAnswer, images, tokens=1400,
                validate=validate_draft,
                response_schema=schema,
                repair_diagnostics=lambda value: answer_diagnostics(
                    value, frames, observations, need_ids, options,
                ),
            )
            value = answer_view(draft, frames, observations)
            value["plan_coverage"] = answer_need_coverage(draft, need_ids)
            transition = {
                "previous_choice": previous_choice, "selected_choice": draft.selected_choice,
                "changed": bool(options) and draft.selected_choice != previous_choice,
                "reason": draft.choice_change_reason,
                "previous_conclusion": (
                    " ".join(next(claim["statement"] for claim in previous["claims"]
                                  if claim["id"] == identifier)
                             for identifier in previous["answer_claim_ids"])
                    if previous is not None else None
                ),
                "revision_scope": feedback["revision_scope"] if feedback else "initial",
            }
            value["selected_option"] = (
                options.index(draft.selected_choice) if draft.selected_choice else None
            )
            entry = bundle["trace"][-1]
            answer_trace_index = len(bundle["trace"]) - 1
            entry.update(
                structured_output=draft.model_dump(), output=value, revision=revision,
                revision_scope=feedback["revision_scope"] if feedback else "initial",
                decision_transition=transition,
                controller_transform={"operation": "claims_to_answer_text_and_frame_citations",
                                      "provisional_candidate": candidate,
                                      "final_choice": draft.selected_choice,
                                      "claim_partition": claim_partition(draft)},
            )
            bundle["answer"] = value
            bundle.pop("verification", None)
            bundle["status"] = "unverified"
            only_limitation = all(claim.kind == "limitation" for claim in draft.claims
                                  if claim.id in draft.answer_claim_ids)
            insufficient = only_limitation or (bool(options) and draft.selected_choice is None)
            bundle["resolution"] = "insufficient_evidence" if insufficient else "answer_proposed"
            graph.pop("review", None)
            graph["reasoning"] = draft.model_dump()
            graph["claim_partition"] = claim_partition(draft)
            graph["claim_provenance"] = value["claim_provenance"]
            graph["plan_coverage"] = value["plan_coverage"]
            yield bundle
            targets = [claim.id for claim in draft.claims]
            review = pipeline.structured(
                bundle["trace"], "Verifier",
                pipeline.role_prompts.review_prompt(
                    question, evidence,
                    {**draft.model_dump(), "resolved_provenance": value["claim_provenance"],
                     "claim_partition": claim_partition(draft), "decision": value["decision"],
                     "decision_transition": transition},
                    targets, final=bool(revision),
                ),
                DirectStructuredReview, images, tokens=1000,
                validate=lambda result, draft=draft: validate_direct_review(result, draft, options),
                response_schema=bounded_schema(
                    DirectStructuredReview, options=options, targets=targets,
                ),
            )
            verdict = review_verdict(review, draft, revision=revision)
            bundle["trace"][-1].update(
                output=verdict, structured_output=review.model_dump(), revision=revision,
                revision_scope=feedback["revision_scope"] if feedback else "initial",
                max_answer_revisions=1,
            )
            bundle["verification"] = verdict
            bundle["review_complete"] = True
            bundle["status"] = (
                "verified" if verdict["accepted"] and verdict["explanation_accepted"]
                and not insufficient else "unverified"
            )
            graph["review"] = verdict
            bundle["reviewed_pair"] = {
                "answer_trace_index": answer_trace_index,
                "review_trace_index": len(bundle["trace"]) - 1,
            }
            last_reviewed = deepcopy({
                "answer": value, "verification": verdict, "resolution": bundle["resolution"],
                "reasoning": graph["reasoning"], "claim_partition": graph["claim_partition"],
                "claim_provenance": graph["claim_provenance"],
                "plan_coverage": graph["plan_coverage"],
                "reviewed_pair": bundle["reviewed_pair"],
            })
            yield bundle
            if not verdict["needs_revision"]:
                break
            previous, feedback = draft.model_dump(), verdict
        bundle["final_answer_complete"] = True
        yield bundle
    except Exception:
        if (explanation_only and last_reviewed is not None
                and last_reviewed["verification"]["accepted"]):
            # Optional explanation editing must not erase an already reviewed
            # answer. Restore the *matching* answer/review pair, not a candidate
            # or an unreviewed revision. The failed attempt stays in the trace.
            for key in ("answer", "verification", "resolution"):
                bundle[key] = last_reviewed[key]
            bundle["reviewed_pair"] = last_reviewed["reviewed_pair"]
            for key in ("reasoning", "claim_partition", "claim_provenance", "plan_coverage"):
                graph[key] = last_reviewed[key]
            note = ("Explanation refinement did not complete; "
                    "the previously reviewed answer is unchanged.")
            bundle["verification"].update(
                needs_revision=False, explanation_repair_failed=True,
                advisories=[*bundle["verification"]["advisories"], note],
            )
            graph["review"] = bundle["verification"]
            bundle.update(status="unverified", final_answer_complete=True, review_complete=True,
                          retained_previous_draft=False, retained_reviewed_answer=True,
                          explanation_repair_failed=True)
            yield bundle
            return
        # Only an already validated draft can survive optional review/revision
        # failure. It remains explicitly incomplete and ineligible for release
        # scoring; the failed stage and raw outputs remain in the trace.
        if not bundle.get("answer"):
            raise
        if last_reviewed is not None:
            for key in ("answer", "verification", "resolution"):
                bundle[key] = last_reviewed[key]
            bundle["reviewed_pair"] = last_reviewed["reviewed_pair"]
            for key in ("reasoning", "claim_partition", "claim_provenance", "plan_coverage"):
                graph[key] = last_reviewed[key]
        issue = "Review or answer revision did not complete; this retained draft is unverified."
        previous_verdict = bundle.get("verification", {})
        bundle["verification"] = {
            **previous_verdict, "accepted": False, "needs_revision": False,
            "review_complete": False,
            "issues": list(previous_verdict.get("issues", [])) + [issue],
            "verdict_source": "controller_retained_valid_draft_after_incomplete_stage",
        }
        graph["review"] = bundle["verification"]
        bundle.update(status="unverified", final_answer_complete=False, review_complete=False,
                      retained_previous_draft=True, completion_issue=issue)
        yield bundle
