"""Compact grounding-focused candidate; clean-v2 remains an immutable baseline.

Retrieval, schemas, sampling, output budgets and the number of model calls are unchanged.
The reviewer deliberately does not receive the inspector's fallible descriptions.
"""

import json

from videomasys.zerogpu.clean_prompts import (
    FINAL_REVIEW_RULE,
    caption_prompt,
    choice_prompt,
    planner_prompt,
    review_rules,
    revision_prompt,
    tool_prompt,
)

__all__ = [
    "PROMPT_VERSION", "FINAL_REVIEW_RULE", "caption_prompt", "choice_prompt",
    "planner_prompt", "tool_prompt", "inspect_prompt", "refine_prompt", "answer_prompt",
    "verifier_prompt", "review_rules", "revision_prompt",
]

PROMPT_VERSION = "grounded-roles-v1"
VISUAL_RULES = (
    "Images are the evidence; the question and descriptions are not. "
    "Describe visible contact or motion before inferring its purpose. "
    "Use a broad object/tool name when its identity is unclear. "
    "Do not merge different objects into one action sequence. "
    "Do not add an arrival, departure, repetition or completed result that the samples omit. "
    "A state change requires visible before/after states of the same object. "
)
COMPARISON_RULES = (
    "Choice statuses: best_supported = the candidate explains the visible interaction; "
    "compatible = fits but the images do not distinguish this purpose; "
    "unclear = insufficient evidence; contradicted = a visible fact is incompatible with it. "
    "Not seen is unclear, NOT contradicted. Absence of a person/action in sparse samples "
    "does not rule out that purpose. "
    "Example (unrelated): no sampled running -> 'ran earlier' is unclear; "
    "a visibly open door -> 'door stays closed in these images' is contradicted. "
)


def inspect_prompt(scene, question, needs):
    return (
        "Report only visible actions relevant to the evidence sought; do not answer the question. "
        + VISUAL_RULES
        + "Give a short chronological observation with sampled timestamps for key interactions. "
        "If an action is already underway in the first image, describe it as underway, "
        "not starting. "
        "Put uncertain identity or unseen transitions in uncertainty, not in observation as fact. "
        "Set relevant from the images. Set refine_at to one sampled time only if closer frames "
        "could resolve a relevant physical interaction; otherwise null. "
        "Observation: at most 80 words; uncertainty: at most 20 words.\n"
        f"Question (not a visual fact): {question}\nEvidence sought: {needs}\n"
        f"These {len(scene['frames'])} images are ALL from scene {scene['index']}, "
        f"spanning {scene['start']:.2f}–{scene['end']:.2f}s. Approximate sample times: "
        f"{[round(t, 2) for t in scene.get('frame_times', [])]}."
    )


def refine_prompt(scene, question, needs):
    return (
        "Describe the visible contact and any supported state change in this closer sequence. "
        + VISUAL_RULES
        + "Use sampled timestamps for key interactions. Do not answer the question. "
        "Put unresolved details in uncertainty. No further sampling is available; "
        "set refine_at=null. "
        "Observation: at most 80 words; uncertainty: at most 20 words.\n"
        f"Question (not a visual fact): {question}\nEvidence sought: {needs}\n"
        f"Scene {scene['index']}; detail span: {scene['detail_span']}; sample times: "
        f"{[round(t, 2) for t in scene['frame_times']]}."
    )


def answer_prompt(context, mapping, evidence, scene_ids, options, candidate):
    rules = (
        "Give a direct, concise answer supported by the selected images. "
        + VISUAL_RULES
        + "Do not retell the question's setup as an observed fact. "
        "In answer, state the main conclusion and its strongest visible support. "
        "In inference, connect that support to the conclusion; visible actions can establish "
        "a likely purpose without showing mental intent. Do not hedge an action whose "
        "physical effect is visible. Cite each scene needed to support your factual claims. "
        "State only specific, relevant gaps in limitations; do not assert that an unseen "
        "event never happened. Omit nonessential uncertain details. "
        "Answer: at most 60 words; inference: at most 35 words. "
    )
    if options:
        rules += (
            f"Audit the fixed candidate {candidate!r}; it is not evidence. "
            "Compare every supplied choice exactly once; only the candidate may be best_supported. "
            "If unsupported, mark it unclear and state the missing evidence; do not change it. "
            "selected_choice and prose must refer to the same candidate. "
            + COMPARISON_RULES
        )
    else:
        rules += (
            "This is open-ended: give a substantive answer or a specific evidence limitation. "
            "Do not ask for choices or return a placeholder. "
        )
    return (
        rules + "\n" + context + "Image mapping: " + json.dumps(mapping)
        + "\nFallible inspection notes (verify against images before using): " + evidence
        + f"\nsupporting_scenes may only contain IDs {sorted(scene_ids)}, never image numbers."
    )


def verifier_prompt(context, mapping, evidence, retrieval, options, candidate):
    # Do not feed the inspector's account back as corroboration of the answer.
    rules = (
        "Audit the draft independently against the actual images. "
        + VISUAL_RULES
        + "Check factual claims in ALL fields, including inference, limitations and "
        "choice reasons. "
        "Reject a wrong tool/object name, an unseen action stated as fact, an unsupported "
        "completed result, or a claim whose supporting scene is missing. "
        "For a failure, name the claim and the sampled time/scene or missing visual evidence. "
        "Accept a grounded main conclusion even when intent is inferred; do not require a "
        "verbal declaration of purpose. Do not demand details the draft does not assert. "
    )
    if options:
        rules += (
            f"The fixed candidate is {candidate!r}. Reject a mismatch or unsupported selection. "
            + COMPARISON_RULES
            + "Reject any contradicted label justified only by 'not shown', 'no evidence' or "
            "an alternative being less likely. "
        )
    else:
        rules += "Review this open-ended answer without requiring choices or a candidate. "
    return (
        rules + "\n" + context + "Image mapping: " + json.dumps(mapping)
        + "\nUncovered retrieval needs (not proof of absence): "
        + json.dumps(retrieval.get("uncovered_evidence_needs", []))
        + "\nUncovered temporal outputs: "
        + json.dumps(retrieval.get("uncovered_temporal_outputs", []))
        + "\nDraft: "
    )
