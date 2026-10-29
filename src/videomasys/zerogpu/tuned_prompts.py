"""Grounding candidate v2: data first, short rules last, explicit output examples.

Inspectors still guide bounded refinement. Answers/reviews independently inspect
the same images without recycling the inspector's descriptions as evidence.
The v1 candidate and clean-v2 baseline remain separately available.
"""

import json

from videomasys.zerogpu.clean_prompts import (
    FINAL_REVIEW_RULE,
    caption_prompt,
    choice_prompt,
    planner_prompt,
    revision_prompt,
    tool_prompt,
)

__all__ = [
    "PROMPT_VERSION", "FINAL_REVIEW_RULE", "caption_prompt", "choice_prompt",
    "planner_prompt", "tool_prompt", "inspect_prompt", "refine_prompt", "answer_prompt",
    "verifier_prompt", "review_rules", "revision_prompt",
]
PROMPT_VERSION = "grounded-roles-v2"

VISUAL_RULES = (
    "Use only what these images show. The question is a request, not a sequence to narrate. "
    "Describe the actual starting state, visible contact/motion, and last visible state. "
    "Do not fill gaps between them. Do not merge different objects into one sequence. "
    "Use a broad name for an uncertain tool or object. A completed result needs visible "
    "before/after states of the same object, not merely a familiar goal. "
)
COMPARISON_RULES = (
    "For each choice: best_supported means the candidate explains the visible action; "
    "compatible means possible but not distinguished; unclear means insufficient evidence; "
    "contradicted requires a positively observed incompatible fact. "
    "Example with unrelated choices: "
    '[{"choice":"ran earlier","status":"unclear","reason":"Earlier movement is not sampled."},'
    '{"choice":"door stays closed","status":"contradicted",'
    '"reason":"The door is visibly open in the sampled interval."}] '
    "Missing evidence must NEVER be labeled contradicted. "
)


def inspect_prompt(scene, question, needs):
    return (
        f"Question (not a visual fact): {question}\nEvidence sought: {needs}\n"
        f"Scene {scene['index']}; interval {scene['start']:.2f}–{scene['end']:.2f}s; "
        f"chronological sample times: {[round(t, 2) for t in scene.get('frame_times', [])]}.\n"
        "Task: describe this window, not the answer. "
        + VISUAL_RULES
        + "Example: if the first frame already shows a runner beyond a gate, say "
        "'Already beyond the gate, the runner slows down'; do not invent crossing the gate. "
        "Give up to three short timestamped facts in observation (at most 70 words). "
        "Put unresolved identity or missing transitions in uncertainty (at most 20 words). "
        "Set relevant based on the requested evidence. Set refine_at to one sampled time "
        "if closer frames could resolve an important interaction, otherwise null."
    )


def refine_prompt(scene, question, needs):
    return (
        f"Question (not a visual fact): {question}\nEvidence sought: {needs}\n"
        f"Scene {scene['index']}; detail span: {scene['detail_span']}; sample times: "
        f"{[round(t, 2) for t in scene['frame_times']]}.\n"
        "Task: report visible contact and state changes, not a guessed purpose. "
        + VISUAL_RULES
        + "Use at most three timestamped facts, 70 words. Put unresolved details in uncertainty, "
        "at most 20 words. No further sampling is available; set refine_at=null."
    )


def answer_prompt(context, mapping, evidence, scene_ids, options, candidate):
    # Deliberately do not repeat fallible inspection prose. Images remain unchanged.
    prompt = context + "Image mapping: " + json.dumps(mapping) + "\n"
    prompt += (
        "Task: answer directly from these images. "
        + VISUAL_RULES
        + "Give the conclusion and its strongest visible support in at most 45 words. "
        "Do not retell the question's setup. In inference, connect the visible support to the "
        "conclusion in at most 30 words. Purpose may be inferred from a supported interaction; "
        "do not demand a statement of mental intent. Name only material uncertainty "
        "in limitations. "
        "Do not list absent distractors as factual limitations. "
        f"supporting_scenes must use IDs {sorted(scene_ids)}, never image numbers, and cover "
        "every scene needed for the facts you actually assert. "
    )
    if options:
        prompt += (
            f"Keep the fixed candidate {candidate!r}. It is a hypothesis, not evidence. "
            "Compare every choice exactly once. Only the candidate may be best_supported. "
            "If it lacks support, use unclear and explain the missing evidence; do not change it. "
            + COMPARISON_RULES
        )
    else:
        prompt += (
            "This is open-ended: give a substantive answer or a specific visual limitation. "
            "Do not ask for choices or use a placeholder."
        )
    return prompt


def verifier_prompt(context, mapping, evidence, retrieval, options, candidate):
    return (
        context + "Image mapping: " + json.dumps(mapping)
        + "\nUncovered retrieval needs (not proof of absence): "
        + json.dumps(retrieval.get("uncovered_evidence_needs", []))
        + "\nUncovered temporal outputs: "
        + json.dumps(retrieval.get("uncovered_temporal_outputs", []))
        + "\nDraft to audit independently against the images: "
    )


def review_rules():
    # This comes AFTER the draft in both initial and final reviews.
    return (
        "\nTask: audit ALL fields of this draft, including inference, limitations and "
        "choice reasons. "
        + VISUAL_RULES
        + "Check wrong object/tool names, invented starting/ending events, unsupported completed "
        "results, and missing supporting scene citations. A correct headline does not excuse "
        "false supporting details. Reject unsupported factual claims even in a limitation. "
        "For choice comparisons, 'not seen' or 'no evidence' cannot justify contradicted. "
        "Example unrelated draft: a choice is contradicted because earlier running is not shown. "
        'Correct review: {"accepted":false,"issues":["Not sampled does not contradict earlier '
        'running; label this alternative unclear."],"needs_revision":true}. '
        "For a real issue, name the specific claim and the sampled scene/time or missing evidence. "
        "Do not copy the example issue unless it actually occurs. "
        "Accept a supported purpose inference without requiring visible mental intent. "
        "Review open-ended answers without requiring choices. "
        "If accepted, return issues=[] and needs_revision=false. Otherwise give specific issues "
        "and request one revision only if the same images can support a corrected or "
        "qualified answer."
    )
