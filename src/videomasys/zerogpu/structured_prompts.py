"""Plain role instructions with shared SVP retrieval examples and linked evidence."""

import json

from videomasys.zerogpu import svp_prompts
from videomasys.zerogpu.clean_prompts import caption_prompt

__all__ = ["PROMPT_VERSION", "caption_prompt", "choice_prompt", "planner_prompt", "tool_prompt",
           "inspection_prompt", "answer_prompt", "review_prompt"]

PROMPT_VERSION = "structured-evidence-v7"

PLANNER_RULES = """\
Plan the evidence to look for, not the answer. You have no video evidence yet;
question wording and choices are things to investigate, not observations.
Use one to three short visual questions with meaningful lowercase IDs.
Prioritize evidence essential to distinguish the answer; extra context is exploratory,
not an additional question the final answer must resolve.
For a local event, find the stated actor/action, then any missing cause or outcome.
Treat pickup/carry/arrival as one anchor unless an intermediate event matters.
Ask what happens to the same object afterward; do not guess the outcome.
Add background people or objects only when they help distinguish plausible answers,
not merely because a distractor mentions them. For object function, seek how it is
used in the activity; do not require a visible before/after change in every object.
For repeated use of an object, seek representative interactions and task context.
Keep genuine before/after dependencies when nearby actions distinguish its use.
Choose scope by the question's target: an object's use is local_event, even when
the wording mentions the overall process. local_event can search occurrences
anywhere in the video; it does not mean only one occurrence.
For an overall objective, use whole_video and independent needs for representative
actions and distinguishing details, not a mandatory beginning/middle/end chain.
Use local_event otherwise. Add depends_on only for event-relative evidence, using
earlier need IDs. temporal_role says where the needed evidence lies relative to
its anchor: before, after, outcome or context; use anchor for the event itself.
A preceding cause depends on its later anchor. For object function, seek contact
and nearby changes. Needs name visible events or context, not the final comparison
or explanation. For phase comparisons, retrieve each phase's actions independently;
the Answer agent compares them. Use outcome only for a later observable event,
not an analytical difference in purpose. Stop when evidence answers the question
or the missing detail is clear. Set max_intervals=3 and max_inspected_seconds=60.
Keep text fields under 25 words. Return the plan JSON.
"""

TOOL_RULES = """\
Turn the plan into a retrieval program, not an answer or invented tool results.
Use version="3.0", compiler="model", at most eight ordered steps and one inspect.
Give steps unique lowercase IDs. source is video or an earlier step's output;
only locate and overview may start from video. Limits are 1..12.
locate searches cached captions for an actor/action/object. Search identifiable
events directly; use scoped locate to narrow earlier results. Captions suggest
search queries, not established facts.
Use distinctive action/object wording from relevant cached captions, preserving
the question's meaning. Query one anchor event, not a conjunction of several
events that could match different moments. For carrying, anchor arrival/placement
of the identifiable object at the destination rather than later generic handling.
For an object's repeated role, retrieve up to three distinct matching windows;
search from video, not only from a sampled overview. Do not anchor a single late
occurrence and search only after it. Honor every
declared plan dependency; do not collapse a multi-hop chain into one relation.
relate searches before/after/around the top source hit. Use a precise anchor with
limit=1; for transport, anchor arrival/placement. Unknown later outcomes normally
use horizon_seconds=60 and limit=4; shorten the horizon for immediate transitions.
Use scoped locate after relate when a target query is available; it ranks candidates
inside the horizon before limiting them. Locate the actual intermediate event before
another temporal hop. Window overlap does not establish event order.
overview samples diverse windows for whole_video: include overview(source="video", limit>=3)
directly in inspect, optionally adding a targeted locate. Do not chain independent
aspects with relate. This samples the video; it does not view every moment.
Tag needs on relevant steps; intermediate steps may have evidence_need_ids=[].
For a dependency, tag its anchor in the relation's source ancestry and its target
on the relation or a later scoped step. A directionless context dependency may
use locate scoped directly to its anchor; it does not need an artificial time hop.
inspect lists useful final outputs and all
need IDs. Every step must feed inspect, directly or through another step. Respect
the plan's budgets; the search horizon is not inspected video time. Return JSON.
"""


def planner_prompt(context):
    return svp_prompts._with_examples(
        PLANNER_RULES, svp_prompts.planner_example_ids(context), "planner", context,
    )


def tool_prompt(context, plan):
    return svp_prompts._with_examples(
        TOOL_RULES, svp_prompts.tool_example_ids(plan), "tool",
        context + "\nPlan to compile: " + json.dumps(plan, separators=(",", ":")),
    )


GROUNDING_RULES = """\
Use images as visual evidence; questions, captions and model observations may be
wrong. Follow the same actor and object without filling gaps between samples.
Use broad names for unclear objects. If something is not visible, say what is
missing, not that it never happened. Separate what is visible from what it suggests.
Purpose or cause may be inferred from linked observations, without inventing actions.
Sample times are approximate, not exact event boundaries.
"""


PURPOSE_REASONING = """\
For why/purpose/function questions, explain the salient object-directed activity,
not just a visible posture or location. A likely purpose can be inferred from
actions and task context without directly seeing the intention or its completed
result. Mark it inferred and cite the observations; never describe an unseen
action as observed. A missing incidental detail is not the same as missing the
essential basis for the answer. Abstain when the available evidence cannot
distinguish a defensible answer, not merely because an inferred purpose is not
literally depicted.
Fictional example: carrying a watering can to a plant and tilting its spout over
the pot supports the purpose of watering. It does not establish that a water
stream was visible. Merely looking at the plant describes posture, not the
purpose of bringing and using the can.
"""


OPTION_REASONING = """\
Compare the closest plausible choices using the same evidence standard. Context
that fits both does not distinguish them. Do not infer one from the setup while
requiring direct action proof for the other. Prefer the function supported by
the object's actual use and nearby actions over a generic possible use; greater
specificity alone is not evidence. State any remaining ambiguity honestly.
"""


def choice_prompt(question, options):
    return (
        "Choose the answer that best explains the visible activity and answers the question. "
        "Use the chronological frames, following the same actor and object. Reasonable "
        "purpose or cause can follow from the actions; do not require a literal depiction "
        "of an intention. Do not invent a missing action or use question wording as evidence.\n"
        + OPTION_REASONING
        + "Question: " + question + "\nChoices: " + json.dumps(options)
    )


def inspection_prompt(question, needs, frames, *, refining=False):
    return (
        "Inspect this window for the requested evidence. Return up to three facts, not an answer.\n"
        + GROUNDING_RULES
        + "For tool use, follow the working end and its contact point, not just the hand "
        "or surrounding setup. Check each frame for a brief interaction before calling "
        "it unseen.\n"
        + "Name subject, predicate and object; choose state, action or change. "
        "Link each fact to supplied frame_ids and need_ids. A change needs at least two "
        "distinct chronological frames. Put a specific ambiguity in uncertainty, otherwise ''. "
        "Each requested need must have a fact or an unresolved entry explaining what is missing.\n"
        "Fictional example: hand / is beside / handle is a state, not evidence of pulling. "
        "Cite its frame; put the unknown pulling action in unresolved.\n"
        + (
            "This is the final close look. Set refine_frame_id=null.\n"
            if refining else
            "If an important interaction is brief or ambiguous in even one frame, set "
            "refine_frame_id to that frame ID for closer sampling. Prioritize possible "
            "contact over static nearby objects; otherwise null.\n"
        )
        + f"Question (not evidence): {question}\n"
        + "Requested needs: " + json.dumps(needs, separators=(",", ":"))
        + "\nFrames in image order: " + json.dumps(frames, separators=(",", ":"))
    )


def answer_prompt(question, options, candidate, evidence, *, draft=None, feedback=None):
    prompt = (
        "Build concise evidence claims and a conclusion. Only statements selected by "
        "answer_claim_ids form the displayed answer.\n"
        + GROUNDING_RULES
        + PURPOSE_REASONING
        + (OPTION_REASONING if options else "")
        + "Use IDs c1, c2, etc. An observed claim says what is visible; an inferred claim "
        "explains it; a limitation says what cannot be established. Link each claim to need_ids. "
        "Cite accurate observation_ids or frame_ids; no need to copy an observation's frames. "
        "An inference may instead depend_on earlier premises, not timeline order, "
        "with a short rationale. "
        "Other claims use depends_on=[] and rationale=''. Unused reference lists are []. "
        "Use direct images if an observation is wrong. Every claim needs direct or inherited "
        "frame support. Prefer claims needed by the answer, its options or limitations. "
        "Extra contextual claims remain visible to review but do not become answer premises "
        "unless an inference explicitly depends on them. "
        "Build the essential reasoning for this question, not a report on every planned need. "
        "Exploratory context and unresolved checks remain in the evidence trace; include them "
        "in the answer only when they support, challenge or limit its conclusion. Do not invent "
        "a premise or a limitation just to cover a need. "
        "answer_claim_ids lists conclusions in order, not every premise. If an existing "
        "conclusion depends on evidence claims, select that conclusion: it already inherits "
        "its premises. Three roots is a maximum, not a target. Do not fill those slots with "
        "observations and omit the actual conclusion. "
        "All limitation claims are displayed separately; use them for meaningful uncertainty.\n"
        "Fictional format example, not evidence: obs_contact shows switch contact and "
        "obs_change shows the indicator changing color afterward:\n"
        '{"claims":[{"id":"c1","kind":"observed",'
        '"statement":"The hand touches the switch and the indicator changes color.",'
        '"frame_ids":[],"observation_ids":["obs_contact","obs_change"],"depends_on":[],'
        '"need_ids":["anchor"],"rationale":""},{"id":"c2","kind":"inferred",'
        '"statement":"The switch appears to activate the indicator.","frame_ids":[], '
        '"observation_ids":[],"depends_on":["c1"],"need_ids":["effect"],'
        '"rationale":"Contact followed by the color change supports activation."}],'
        '"answer_claim_ids":["c2"],"selected_choice":null,"choice_change_reason":"",'
        '"option_comparisons":[]}\n'
        "Use the current evidence IDs below, not the example's observation or need IDs.\n"
        "Here c2 is the conclusion and inherits its evidence through c1. The need checklist "
        "records available observations and uncertainties, not required answer premises.\n"
    )
    if options:
        prompt += (
            "Assess every choice once, in supplied order: supported if cited claims give "
            "evidence for it; unsupported if the available evidence does not support it. "
            "Unsupported does not mean impossible; it needs no proof of absence and may "
            "use claim_ids=[]. Supported choices need observed or inferred claims, not "
            "limitations. More than one choice may be supported. Set selected_choice to "
            "the one that best answers the question, citing its support in the answer. "
            "Do not label another supported choice unsupported just because it did not win. "
            "If no choice can be selected, use null and select a specific limitation claim "
            "in answer_claim_ids; listing only observed facts is not an abstention answer. "
            "Check the preliminary candidate, or the prior selection during revision. Neither "
            "is ground truth or an immutable choice. If selected_choice changes, including "
            "to or from null, give a concrete choice_change_reason citing existing support IDs: "
            "what evidence favors the new answer or makes the prior answer insufficient? "
            "Minor wording, missing incidental details or an unseen intention alone do not "
            "justify a change. If the choice is unchanged, use choice_change_reason=''. "
            "The displayed conclusion must answer the question and agree with selected_choice.\n"
            + "Choices: " + json.dumps(options)
            + "\nPreliminary candidate: " + json.dumps(candidate) + "\n"
        )
    else:
        prompt += (
            "For this open-ended question, set selected_choice=null, choice_change_reason='' "
            "and option_comparisons=[]. "
            "Give the supported answer, or say specifically what the evidence cannot establish.\n"
        )
    prompt += (
        "Use one to five short claims. Answer under 70 words; rationale under "
        "30 words per inferred claim.\nQuestion: " + question
        + "\nEvidence context (model observations are fallible): "
        + json.dumps(evidence, separators=(",", ":"))
    )
    if draft is not None:
        scope = feedback.get("revision_scope", "answer") if isinstance(feedback, dict) else "answer"
        prompt += (
            "\nThis is the only revision. Use the same evidence and address the requested "
            "correction. Advisory context or alternative-option comments do not require "
            "changing the answer; consider them if they reveal a real contradiction. "
            "Feedback is not new evidence.\nPrevious draft: "
            + json.dumps(draft, separators=(",", ":"))
            + "\nReview feedback: " + json.dumps(feedback, separators=(",", ":"))
            + (
                "\nExplanation-only revision: fix the flagged evidence statements, inference "
                "wording or citations while preserving the main conclusion's meaning. "
                "Keep selected_choice exactly as in the previous draft, including null. "
                "Use choice_change_reason=''; do not switch to another choice or abstain."
                if scope == "explanation" else
                "\nAnswer revision: fix the substantive answer issue. Change selected_choice "
                "only when existing evidence justifies it, with choice_change_reason naming "
                "that evidence. Keep the choice and use '' when no change is justified."
            )
        )
    return prompt


def review_prompt(question, evidence, draft, targets, *, final=False):
    return (
        "Review the claims, assess the choices, then judge the selected answer.\n"
        + GROUNDING_RULES
        + PURPOSE_REASONING
        + OPTION_REASONING
        + "Return claim_checks for every supplied claim target_id: pass if justified as "
        "written, otherwise revise. Check the statement, rationale and linked evidence. "
        "resolved_provenance includes inherited frames; empty direct lists do not mean "
        "missing support. A reasonable inference or an honest limitation can pass. "
        "Context claims remain evidence to consider, including anything that challenges "
        "the conclusion; they are not automatically answer premises. "
        "For option_assessments, return every choice in supplied order with YOUR assessment: "
        "supported when its cited claims and frames support it, otherwise unsupported. "
        "No evidence of sharing means status=unsupported, not that sharing was disproved. "
        "Do not return pass/revise for option assessments. Several choices can have support. "
        "For answer_judgment, check the exact decision.selected_choice, displayed conclusion "
        "and support_claim_ids together. They must answer the question and agree with the "
        "essential evidence; do not pass a choice when the displayed conclusion answers "
        "something else. Check any decision_transition: a changed choice needs a concrete "
        "evidence-based reason, not wording preferences or a missing incidental detail. "
        "For revision_scope='explanation', compare decision_transition.previous_conclusion "
        "with the current conclusion: use answer_judgment revise if the substantive answer "
        "meaning changed, not merely its wording. "
        "Separate answer adequacy from explanation edits: flag unsupported wording or "
        "citations in claim_checks, but pass answer_judgment when the same main answer "
        "remains justified. Context or alternative-option comments alone are not answer "
        "failures; if they undermine the conclusion, explain that in answer_judgment. "
        "Use pass and repeat the draft's selected_choice when justified; otherwise revise "
        "and specify the essential defect. Propose a supplied choice or null only when "
        "the evidence warrants changing it. Any non-null proposed choice must be supported "
        "in your option assessments. For open-ended questions use selected_choice=null "
        "and judge the displayed conclusion. An honest abstention can pass when essential "
        "evidence really is insufficient. "
        "Keep reasons under 25 words.\n"
        + ("Final review: report remaining problems even though no revision remains.\n"
           if final else "At most one targeted revision using the same images is available.\n")
        + "Question: " + question
        + "\nClaim targets: " + json.dumps(targets)
        + "\nEvidence context: " + json.dumps(evidence, separators=(",", ":"))
        + "\nDraft: " + json.dumps(draft, separators=(",", ":"))
    )
