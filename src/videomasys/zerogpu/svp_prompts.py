"""Experimental SVP-style planning and source-scoped retrieval demonstrations.

Examples are authored fiction, not benchmark answers or retrieved observations.
The compiler emits one bounded typed program; it is not an observation-driven
agent loop. Unchanged answer/review prompts keep this experiment retrieval-led.
"""

import json
import re

from videomasys.zerogpu.clean_prompts import (
    FINAL_REVIEW_RULE,
    answer_prompt,
    caption_prompt,
    choice_prompt,
    review_rules,
    revision_prompt,
    verifier_prompt,
)

__all__ = [
    "PROMPT_VERSION", "FINAL_REVIEW_RULE", "caption_prompt", "choice_prompt",
    "planner_prompt", "tool_prompt", "inspect_prompt", "refine_prompt", "answer_prompt",
    "verifier_prompt", "review_rules", "revision_prompt",
]

PROMPT_VERSION = "svp-roles-v3"

PLANNER_RULES = """\
Plan which visible facts would answer the question, not the answer or executable code.
You have no video evidence. The question describes what to investigate; choices are
hypotheses, not observations. Use one to three question-specific evidence needs.
Choose scope=whole_video for an overall objective or summary, otherwise local_event.
A global question needs diverse representative actions and distinguishing support,
as independent needs, not a mandatory beginning/middle/end or event chain. Add a
dependency only when the question actually requires event-relative evidence.
Supporting aspects of an overall activity are not a before/after dependency.
A local question can need just one fact.
Name the stated actor/event first when later facts depend on identifying it. For an
unknown subsequent action, request what happens after that anchor without guessing
what happens. A preceding cause depends on the anchor but lies before it in time.
For purpose after stated object transport, combine pickup/carry/arrival into ONE
anchor need. Add an outcome need for the UNKNOWN same-object interaction at the
destination. Do not split the question's stated setup into gratuitous yes/no checks.
Arrival alone does not explain purpose; seek what the person does with the object.
Keep a separate intermediate event only if it distinguishes the requested answer.
For object function, request actual contact and changes around the same interaction.
Use temporal_role anchor|before|after|outcome|context; independent needs have no
dependencies. depends_on contains only earlier need IDs. Use meaningful lowercase
IDs, short descriptions phrased as visual subquestions, and an evidence-based stop
condition. Set max_intervals=3 and max_inspected_seconds=60. Keep other text fields
under 25 words. Return only the plan JSON; no extended reasoning.
"""

TOOL_RULES = """\
Compile the supplied plan into one typed retrieval program, not an answer or result.
Use version="3.0", compiler="model", ordered steps and one terminal inspect object.
Each step has op, a meaningful unique lowercase id, source, limit (1..12), and
evidence_need_ids. Only locate and overview may start from source="video"; otherwise
source must name an earlier step. No arbitrary Python or invented tool observations.
locate: search cached captions using a short actor/action/object query. Start directly
from video when the event is identifiable. locate -> inspect is a valid complete
program; do not add a broad actor search first unless it disambiguates the event.
Source is a real restriction: scoped locate searches only an earlier output.
Cached action descriptions may supply concrete queries as SEARCH HYPOTHESES, never
observations or answers. Search the requested interaction, not just an actor/location.
relate: retrieve temporal candidates from the top-ranked source hit. Retrieving B
after A uses source=A, relation=after; before/cause uses before, contact context around.
Use a precise event anchor with limit=1. For transport, anchor the arrival/placement,
not a query mixing every pickup, carrying and arrival action. For an unknown outcome
across coarse windows, use horizon_seconds=60 and limit=4; use a shorter horizon only
when the question requires an immediate transition. This is cached search distance,
NOT inspected video seconds. Follow relate with scoped locate when target semantics
are available. That target query ranks eligible windows BEFORE the relation limit;
it never widens the horizon. Before another temporal hop, locate its actual event
within the first relation output. Overlapping windows do not prove event ordering.
overview: select diverse cached windows for overall activity. For whole_video include
overview(source="video", limit>=3) directly in inspect, optionally with a targeted
locate for distinguishing support. Do not add relate just to connect supporting
aspects of an overall objective. Representative sampling is not full-video viewing.
Tag each need on a relevant step. Intermediate searches may have evidence_need_ids=[];
a need does NOT need a separate global locate. For a dependency, tag the earlier need
in the relation's source ancestry and the target on the relation or its descendant.
inspect names the useful final evidence outputs and all plan need IDs. Every step
must reach inspect, but intermediate candidates need not be viewed. Use at most eight
steps and respect the plan's budgets. Return only program JSON.
"""


def _plan(scope, objective, needs, stop, reasoning):
    return {
        "scope": scope,
        "objective": objective,
        "evidence_needs": [
            {"id": name, "description": question, "temporal_role": role, "depends_on": deps}
            for name, question, role, deps in needs
        ],
        "temporal_constraints": [],
        "option_strategy": None,
        "max_intervals": 3,
        "max_inspected_seconds": 60,
        "stop_condition": stop,
        "reasoning": reasoning,
    }


def _locate(name, source, query, limit, needs=()):
    return {
        "op": "locate", "id": name, "source": source, "query": query,
        "limit": limit, "evidence_need_ids": list(needs),
    }


def _relate(name, source, relation, horizon, limit, needs=()):
    return {
        "op": "relate", "id": name, "source": source, "relation": relation,
        "horizon_seconds": horizon, "limit": limit, "evidence_need_ids": list(needs),
    }


def _program(rationale, steps, sources, needs):
    return {
        "version": "3.0", "compiler": "model", "rationale": rationale,
        "max_intervals": 3, "max_inspected_seconds": 60,
        "steps": steps,
        "inspect": {"sources": sources, "evidence_need_ids": needs},
    }


# Questions, requested evidence, unexecuted programs and explicitly fictional
# cache hints demonstrate retrieval. There are no answers, frames or tool results.
DEMONSTRATIONS = {
    "direct_event": {
        "question": "What color is the container the worker places on the cart?",
        "plan": _plan(
            "local_event", "Identify the container color at the stated placement.",
            [("container_color", "What color is the container placed on the cart?",
              "context", [])],
            "The container color is visible at placement, or remains unclear.",
            "One identifiable interaction supplies the requested attribute.",
        ),
        "program": _program(
            "Search the stated interaction directly and inspect the best matching window.",
            [_locate("container_placement", "video", "worker places container on cart", 1,
                     ["container_color"])],
            ["container_placement"], ["container_color"],
        ),
    },
    "scoped_event": {
        "question": "What color is the worker's cap when they lift the orange cone?",
        "plan": _plan(
            "local_event", "Identify the cap color at the stated interaction.",
            [("cap_at_lift", "What color is the worker's cap during the cone lift?",
              "context", [])],
            "The worker and cap are visible at that interaction, or the color remains unclear.",
            "Identify the correct actor and event before checking the attribute.",
        ),
        "program": _program(
            "Narrow worker candidates to the stated interaction; inspect only that subset.",
            [
                _locate("worker_candidates", "video", "worker", 8),
                _locate("cone_lift", "worker_candidates", "worker lifts orange cone", 2,
                        ["cap_at_lift"]),
            ],
            ["cone_lift"], ["cap_at_lift"],
        ),
    },
    "after_event": {
        "question": "What does the athlete do after touching the yellow marker?",
        "plan": _plan(
            "local_event", "Find the athlete's action following the marker contact.",
            [
                ("marker_contact", "When does the athlete touch the yellow marker?", "anchor", []),
                ("following_action", "What does the same athlete do afterward?", "after",
                 ["marker_contact"]),
            ],
            "The anchor and a subsequent same-athlete action are observed, "
            "or their order remains unclear.",
            "Use the known contact to locate the unknown later action.",
        ),
        "program": _program(
            "Find the exact contact, move forward, then narrow the follow-up to the same athlete.",
            [
                _locate("marker_touch", "video", "athlete touches yellow marker", 1,
                        ["marker_contact"]),
                _relate("after_touch", "marker_touch", "after", 60, 4),
                _locate("athlete_followup", "after_touch", "athlete actions", 2,
                        ["following_action"]),
            ],
            ["marker_touch", "athlete_followup"], ["marker_contact", "following_action"],
        ),
    },
    "transport_purpose": {
        "question": (
            "Why does the technician lift the control panel and carry it to the workbench?"
        ),
        "plan": _plan(
            "local_event", "Explain the stated transport through the subsequent panel interaction.",
            [
                ("panel_arrival",
                 "When does the technician lift the panel and bring it to the workbench?",
                 "anchor", []),
                ("panel_outcome", "What does the technician then do with that panel there?",
                 "outcome", ["panel_arrival"]),
            ],
            "The subsequent same-panel interaction explains its transport, "
            "or the purpose remains unresolved.",
            "Treat the stated transport as one anchor; investigate the unknown interaction.",
        ),
        "cached_caption_hints": [
            "Technician carries a control panel to the workbench.",
            "Technician rotates several fasteners on the control panel.",
        ],
        "program": _program(
            "Narrow the stated arrival, then use the fallible fastener-motion hint "
            "to search a later interaction.",
            [
                _locate("panel_at_workbench", "video",
                        "technician places control panel on workbench", 1, ["panel_arrival"]),
                _relate("after_panel_arrival", "panel_at_workbench", "after", 60, 4),
                _locate("panel_interaction", "after_panel_arrival",
                        "technician rotates fasteners on control panel", 2, ["panel_outcome"]),
            ],
            ["panel_at_workbench", "panel_interaction"], ["panel_arrival", "panel_outcome"],
        ),
    },
    "interleaved_events": {
        "question": (
            "What does the attendant do after the visitor shows a pass following the gate opening?"
        ),
        "plan": _plan(
            "local_event",
            "Find the attendant's response to the correctly ordered pass interaction.",
            [
                ("gate_opening", "When does the gate open?", "anchor", []),
                ("pass_interaction", "When does the visitor show a pass after that opening?",
                 "after", ["gate_opening"]),
                ("attendant_response", "What does the attendant do after that pass interaction?",
                 "after", ["pass_interaction"]),
            ],
            "The ordered pass interaction and attendant response are observed, "
            "or their relation remains unclear.",
            "Identify the intermediate event before seeking the action relative to it.",
        ),
        "program": _program(
            "Re-localize the pass event inside temporal candidates before traversing from it.",
            [
                _locate("gate_opens", "video", "gate opens", 1, ["gate_opening"]),
                _relate("after_gate", "gate_opens", "after", 60, 5),
                _locate("pass_shown", "after_gate", "visitor shows pass to attendant", 1,
                        ["pass_interaction"]),
                _relate("after_pass", "pass_shown", "after", 60, 4),
                _locate("attendant_followup", "after_pass", "attendant actions", 2,
                        ["attendant_response"]),
            ],
            ["gate_opens", "pass_shown", "attendant_followup"],
            ["gate_opening", "pass_interaction", "attendant_response"],
        ),
    },
    "before_cause": {
        "question": "Why does the cyclist stop near the gate?",
        "plan": _plan(
            "local_event", "Find visible context that could explain the cyclist's stop.",
            [
                ("cyclist_stop", "When and where does the cyclist stop near the gate?",
                 "anchor", []),
                ("preceding_context", "What visible condition or event precedes that stop?",
                 "before", ["cyclist_stop"]),
            ],
            "A visible preceding condition explains the stop, or the cause cannot be established.",
            "Search backward from the stated event without assuming its cause.",
        ),
        "program": _program(
            "Anchor the stop, then inspect its immediate preceding context.",
            [
                _locate("stop_at_gate", "video", "cyclist stops near gate", 1, ["cyclist_stop"]),
                _relate("before_stop", "stop_at_gate", "before", 30, 3, ["preceding_context"]),
            ],
            ["stop_at_gate", "before_stop"], ["cyclist_stop", "preceding_context"],
        ),
    },
    "object_function": {
        "question": "How does the person use the small lever during this operation?",
        "plan": _plan(
            "local_event", "Establish the lever's function from its actual interaction.",
            [
                ("lever_contact", "How does the hand contact or move the lever?", "anchor", []),
                ("contact_changes",
                 "What changes in the same lever or mechanism around that contact?",
                 "context", ["lever_contact"]),
            ],
            "The contact and corresponding state change support a function, "
            "or the effect remains uncertain.",
            "Inspect contact and nearby states rather than assuming a familiar function.",
        ),
        "program": _program(
            "Find lever contact and inspect nearby same-object changes.",
            [
                _locate("lever_use", "video", "person uses small lever", 1, ["lever_contact"]),
                _relate("contact_context", "lever_use", "around", 20, 4),
                _locate("mechanism_detail", "contact_context", "lever position and mechanism", 3,
                        ["contact_changes"]),
            ],
            ["lever_use", "mechanism_detail"], ["lever_contact", "contact_changes"],
        ),
    },
    "global_objective": {
        "question": (
            "Across the clip, how do the person's interactions with the components "
            "contribute to the main objective?"
        ),
        "plan": _plan(
            "whole_video",
            "Identify the overall objective from representative activity and object changes.",
            [
                ("activity_pattern",
                 "What representative activities recur or change across the clip?",
                 "context", []),
                ("component_changes",
                 "What visible component interactions or state changes distinguish the objective?",
                 "context", []),
            ],
            "Diverse activity and a distinguishing interaction support the objective, "
            "or its scope remains uncertain.",
            "Combine broad coverage with specific corroboration without presuming a task.",
        ),
        "program": _program(
            "Retrieve diverse cached activity and corroborate a component interaction "
            "within that coverage.",
            [
                {"op": "overview", "id": "activity_overview", "source": "video", "limit": 6,
                 "evidence_need_ids": ["activity_pattern"]},
                _locate("component_detail", "activity_overview",
                        "person interacting with components", 2, ["component_changes"]),
            ],
            ["activity_overview", "component_detail"], ["activity_pattern", "component_changes"],
        ),
    },
}


def _question_text(context):
    """Choose teaching patterns from the question only, never cached hints or choices."""
    question = context.partition("Question:")[2] if "Question:" in context else context
    boundary = r"\n(?:Answer choices|Response mode|Cached|Plan to compile)\b"
    return re.split(boundary, question)[0].lower()


def planner_example_ids(context):
    question = _question_text(context)
    purpose = re.search(r"\bwhy\b|\bpurpose\b|\breason\b", question)
    transport = re.search(
        r"\b(?:carry|carries|carrying|carried|bring|brings|bringing|brought)\b"
        r"|\bpick(?:s|ed|ing)? up\b|\bmove(?:s|d|ing)?\b.*\bto\b", question,
    )
    if purpose and transport:
        return ["transport_purpose", "before_cause", "global_objective"]
    if re.search(r"\bafter\b", question) and re.search(r"\bfollowing\b", question):
        return ["interleaved_events", "after_event", "global_objective"]
    if re.search(r"\bbefore\b|\bwhy\b|\bcause", question):
        return ["before_cause", "after_event", "global_objective"]
    if re.search(r"\bfunction\b|\bpurpose\b|\buse[ds]?\b|\bcontribut", question):
        return ["object_function", "global_objective", "scoped_event"]
    return ["direct_event", "after_event", "global_objective"]


def tool_example_ids(plan):
    """Select examples by validated structure, not benchmark matching or another model."""
    if plan.get("scope") == "whole_video":
        return ["global_objective", "scoped_event"]
    dependencies = [need for need in plan.get("evidence_needs", []) if need.get("depends_on")]
    dependent_ids = {need["id"] for need in dependencies}
    if any(
        dependency in dependent_ids for need in dependencies for dependency in need["depends_on"]
    ):
        return ["interleaved_events", "after_event"]
    if any(need["temporal_role"] == "before" for need in dependencies):
        return ["before_cause", "after_event"]
    if any(need["temporal_role"] == "outcome" for need in dependencies):
        return ["transport_purpose", "after_event"]
    if any(need["temporal_role"] == "after" for need in dependencies):
        return ["after_event", "transport_purpose"]
    if dependencies:
        return ["object_function", "scoped_event"]
    return ["direct_event", "scoped_event"]


def _with_examples(rules, ids, role, current):
    parts = [
        rules,
        "BEGIN_FICTIONAL_EXAMPLES\nThese unexecuted pairs teach structure only. They contain no "
        "verified observations or correct answers. Any cached hints are fictional and fallible. "
        "Do not copy their entities, IDs, hints or queries into the current task. "
        "Do not choose a pattern just because it appears in an example.",
    ]
    for example_id in ids:
        example = DEMONSTRATIONS[example_id]
        inputs = {"question": example["question"]}
        if role == "tool":
            inputs["plan"] = example["plan"]
            if "cached_caption_hints" in example:
                inputs["cached_caption_hints"] = example["cached_caption_hints"]
        output = example["program" if role == "tool" else "plan"]
        parts.append(
            f"Example {example_id}\nINPUT: " + json.dumps(inputs, separators=(",", ":"))
            + "\nOUTPUT: " + json.dumps(output, separators=(",", ":"))
        )
    parts.append("END_FICTIONAL_EXAMPLES\nCURRENT TASK:\n" + current)
    return "\n\n".join(parts), list(ids)


def planner_prompt(context):
    return _with_examples(PLANNER_RULES, planner_example_ids(context), "planner", context)


def tool_prompt(context, plan):
    return _with_examples(
        TOOL_RULES, tool_example_ids(plan), "tool",
        context + "\nPlan to compile: " + json.dumps(plan, separators=(",", ":")),
    )


INSPECTION_RULES = (
    "Treat each requested evidence need as a visual subquestion about these frames only. "
    "Report up to three short timestamped facts in observation, linking them to the relevant "
    "need ID when supplied; do not answer the overall question or select a choice. "
    "Describe the actual starting state, visible contact or motion, and last visible state. "
    "Do not narrate a transition that happened before the first sample or fill gaps. "
    "Track the same actor and object; do not merge different objects. If identity or a tool "
    "subtype is unclear, use a broad name and state the uncertainty. "
    "Only images establish visual facts: the question and cached descriptions can be wrong. "
    "Unobserved is not contradicted. Report an unsupported subquestion as unresolved, not absent. "
    "Keep observation under 80 words and uncertainty under 20 words. "
)


def inspect_prompt(scene, question, needs):
    return (
        INSPECTION_RULES
        + "Set relevant from the requested evidence actually visible. For one important brief "
        "interaction that closer sampling could clarify, set refine_at to its sampled time; "
        "otherwise null.\n"
        f"Question (not evidence): {question}\nVisual subquestions for this scene: {needs}\n"
        f"These {len(scene['frames'])} chronological images are all from scene {scene['index']}, "
        f"spanning {scene['start']:.2f}–{scene['end']:.2f}s. Sample times: "
        f"{[round(t, 2) for t in scene.get('frame_times', [])]}."
    )


def refine_prompt(scene, question, needs):
    return (
        INSPECTION_RULES
        + "This is a closer view of the same scene. Check the requested contact and visible "
        "before/after states; do not guess a purpose. No further sampling is available; "
        "set refine_at=null.\n"
        f"Question (not evidence): {question}\nVisual subquestions for this scene: {needs}\n"
        f"Scene {scene['index']}; detail span: {scene['detail_span']}; sample times: "
        f"{[round(t, 2) for t in scene['frame_times']]}."
    )
