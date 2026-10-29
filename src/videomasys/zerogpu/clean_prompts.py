"""Focused, example-free role prompts. Data, schemas and budgets stay controller-owned."""

import json

PROMPT_VERSION = "clean-roles-v2"
EVIDENCE_RULES = (
    "Use only the supplied images as visual evidence; descriptions can be wrong. "
    "Separate observations from inference. An unobserved action is not a contradicted action. "
    "Do not invent actions between samples or beyond their time span."
)


def caption_prompt(scene):
    return (
        "Describe the visible actor/object actions and state changes across these chronological "
        "frames for later search. Name concrete interactions, not just the setting. "
        "Do not infer intent or unseen actions. Use at most 60 words. "
        f"Window: {scene['start']:.1f}–{scene['end']:.1f}s; "
        f"sample times: {scene.get('frame_times', [])}."
    )


def planner_prompt(context):
    return (
        "Plan the visual evidence needed to answer the question, not the answer. "
        "You have no video evidence yet; choices are hypotheses. "
        "For whole_video scope, use exactly three independent context needs for beginning, "
        "middle and end, each with depends_on=[]. For local_event, request only distinguishing "
        "facts. When purpose depends on an action's outcome, locate the stated anchor and "
        "request its UNKNOWN subsequent same-object interaction; arrival alone is not the outcome. "
        "For a preceding cause, the before need depends on the anchor. Independent attributes "
        "need no temporal chain. Use descriptive lowercase IDs and dependencies on earlier IDs. "
        "Use at most three needs; set max_intervals=3 and max_inspected_seconds=60 "
        "(at most 60 inspected seconds). State an evidence-based "
        "stop condition or unresolved uncertainty. Keep text fields under 25 words.\n" + context,
        None,
    )


def tool_prompt(context, plan):
    return (
        "Compile the supplied plan into a ToolProgram, not an answer or executed result. "
        'Use version="2.0", compiler="model", locate/relate arrays and one inspect object. '
        "Respect the plan's budgets. Give EVERY need its own direct locate query using its "
        "description and exact ID, even when relate also targets it. Queries search fallible "
        "cached captions; do not invent actions. Use unique l1,l2,... and r1,r2,... IDs, "
        "limits 2–4. "
        "For whole_video, use three independent locates with temporal_position beginning, middle, "
        "end, drawing hints from their matching timeline groups; no relations. For local_event "
        "use any unless a temporal position is established. Each dependency requires a relate "
        "from the dependency's source output to the dependent need: before for temporal_role "
        "before, after for after/outcome. relation places the TARGET relative to SOURCE: "
        "if A precedes B, retrieving B from source A uses after, not before. "
        "No relations for independent needs. References must "
        "point to earlier outputs. Overlapping windows do not prove event order. "
        "inspect must include ALL locate and relate outputs in sources and ALL need IDs in "
        "evidence_need_ids; no dead outputs or discarded dependencies.\n"
        + context
        + "\nPlan to compile: "
        + json.dumps(plan, separators=(",", ":")),
        None,
    )


def inspect_prompt(scene, question, needs):
    return (
        "Describe the visible action sequence relevant to the requested evidence; do not answer "
        "the question. Follow actor/object continuity and physical state changes. "
        + EVIDENCE_RULES
        + " Set relevant from these images. Put unresolved details in uncertainty. If a brief "
        "relevant interaction needs a closer look, set refine_at to its sampled time; otherwise "
        "null. Observation: at most 80 words; uncertainty: at most 20 words.\n"
        f"Question (not a visual fact): {question}\nEvidence sought: {needs}\n"
        f"These {len(scene['frames'])} images are ALL from scene {scene['index']}, "
        f"spanning {scene['start']:.2f}–{scene['end']:.2f}s. Approximate sample times: "
        f"{[round(t, 2) for t in scene.get('frame_times', [])]}."
    )


def refine_prompt(scene, question, needs):
    return (
        "Inspect this closer chronological sequence. Name the physical interaction and visible "
        "changes before/during/after it; report unresolved details rather than guessing. "
        + EVIDENCE_RULES
        + " Do not select an answer. No further sampling is available; set refine_at=null. "
        "Observation: at most 80 words; uncertainty: at most 20 words.\n"
        f"Question: {question}\nEvidence sought: {needs}\nScene {scene['index']}; "
        f"detail span: {scene['detail_span']}; sample times: "
        f"{[round(t, 2) for t in scene['frame_times']]}."
    )


def choice_prompt(question, options):
    return (
        "Select the most specific answer supported by these chronological images. "
        "Do not invent a cause or outcome. Return the exact choice text only.\n"
        f"Question: {question}\nChoices: {json.dumps(options)}\n"
    )


def answer_prompt(context, mapping, evidence, scene_ids, options, candidate):
    rules = (
        "Answer the question directly using the selected images. "
        + EVIDENCE_RULES
        + " A visible same-object sequence may support a qualified purpose inference; intent "
        "need not be directly visible. State specific visual limits, not generic doubt. "
        "Cite only scenes that support your claims. Answer: at most 80 words; inference: 45 words. "
    )
    if options:
        rules += (
            f"Audit this fixed candidate: {candidate!r}; it is not evidence. Compare every "
            "supplied choice exactly once with a short visual reason. Only the candidate may "
            "be best_supported. If unsupported, mark it unclear and state limitations; do not "
            "silently change it. selected_choice and prose must refer to the same candidate. "
        )
    else:
        rules += (
            "This is open-ended: give a substantive answer or a specific evidence limitation. "
            "Do not ask for choices or return a placeholder. "
        )
    return (
        rules
        + "\n"
        + context
        + "Image mapping: "
        + json.dumps(mapping)
        + "\nObservations: "
        + evidence
        + f"\nsupporting_scenes may only contain IDs {sorted(scene_ids)}, never image numbers."
    )


def verifier_prompt(context, mapping, evidence, retrieval, options, candidate):
    rules = (
        "Review the draft against the actual images, not agreement with its observations. "
        + EVIDENCE_RULES
        + " Check actor/object continuity, timing, citations, and factual claims. Accept a "
        "grounded, qualified inference or an honest specific visual limitation. Reject invented "
        "facts, unsupported certainty or a non-answer. "
    )
    if options:
        rules += (
            f"The draft selects EXACTLY this option: {candidate!r}. Check its support and every "
            "alternative's visual justification; reject unresolved or mismatched choice claims. "
        )
    else:
        rules += "Review this open-ended answer without requiring choices or a candidate. "
    return (
        rules
        + "\n"
        + context
        + "Image mapping: "
        + json.dumps(mapping)
        + "\nUncovered retrieval needs: "
        + json.dumps(retrieval.get("uncovered_evidence_needs", []))
        + "\nUncovered temporal outputs: "
        + json.dumps(retrieval.get("uncovered_temporal_outputs", []))
        + "\nObservations: "
        + evidence
        + "\nDraft: "
    )


def review_rules():
    return (
        "\nIf accepted, return issues=[] and needs_revision=false. Otherwise give specific "
        "evidence issues. Request one revision only if the same images can support a corrected "
        "or qualified answer; the review is not new evidence."
    )


def revision_prompt(base, draft, issues):
    return (
        base
        + "\nCorrect the draft using the same images and the review issues. Remove unsupported "
        "details, correct citations, and qualify uncertain claims. "
        "Do not treat feedback as evidence."
        + "\nPrevious draft: "
        + draft
        + "\nReview issues: "
        + json.dumps(issues)
    )


FINAL_REVIEW_RULE = (
    "\nThis is the final review; set needs_revision=false. Report unresolved issues rather than "
    "accepting because the correction budget is exhausted."
)
