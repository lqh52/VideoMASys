"""Preserved legacy profile: paired planner/compiler demonstrations and evidence rules.

The examples are independently authored fiction, not benchmark answers or visual
observations. Keep them out of the image-backed inspector/answerer/reviewer calls.
"""

import json

from videomasys.agents.tool_sequence import TOOL_DATAFLOW_RULES

PROMPT_VERSION = "legacy-paired-synthetic-v3"

PLANNER_RULES = """\
Return one plan JSON object matching the schema, including scope. Plan visual
checks, not an answer. You have NO images or current-video captions. Define what
must be checked from the question; do not guess an observed sequence. Choices
are competing hypotheses, not facts.

FIRST classify scope from the QUESTION, not the most salient cached action:
- whole_video: overall/primary/overarching objective, summary, or all actions across
  the video. Exactly THREE independent context needs: early, middle, late. All have
  depends_on=[]. Ask for representative actions and state changes, not a predicted
  activity or a local anchor/outcome pair.
- local_event: a particular action, object, interaction, cause, or return episode.

For local_event choose the appropriate pattern:
- Transport purpose, ONLY when the question explicitly states carrying an object:
  exactly TWO needs. Anchor combines stated setup, handling, movement, and arrival.
  Outcome asks what the person DOES WITH THAT OBJECT AT THE DESTINATION AFTER
  ARRIVING. Repeating the movement/arrival is not an explanatory outcome.
  Describe the UNKNOWN subsequent interaction to inspect, not a predicted
  placement, opening, or use. Do not stop just because arrival/placement is seen.
- Return/again: anchor is the LATER return after the stated preceding activity.
  Outcome is the interaction at the return location. Do not invent a carried object,
  placement, or posture; do not substitute the initial visit for the return.
- Object function: cover BEFORE, CONTACT, and AFTER for the same hand/tool/object.
  Seek what enters, leaves, or changes and compare competing physical functions.
  Do not invent rules such as 'only one function requires repeating an action'.
- Cause: anchor the event, then add a before need depending on that anchor to check
  its preceding condition. This retrieval dependency points backward in video time.
  Other local questions need
  only facts that distinguish the alternatives.

Use descriptive lowercase IDs and dependencies on earlier IDs. At most three needs;
max_intervals=3, max_inspected_seconds=60. Keep text fields under 25 words.
Independent local facts (for example, an object's color) do NOT need an invented
before/after chain. Use one context need if it can answer the question.
"""


TOOL_RULES = (
    """\
Return one ToolProgram JSON object matching the schema; no code or answer.
Use compiler="model", version="2.0". locate and relate are arrays; inspect is ONE
object. Respect the plan's interval and seconds budgets.

locate searches cached captions. EVERY need needs its own direct locate, INCLUDING
needs also reached by relate. Use short actor/object/action queries, not 'purpose'
alone. Translate EACH evidence need's description into a query that actually
searches for that need. Do not replace a need with a more salient cached action.
Copy each need ID exactly; check coverage against the complete plan before
returning. Use supplied entities/actions; choices suggest alternatives, never facts.
Use sequential l1,l2,l3, limits 2–4. For scope=whole_video, compile three independent
locates for early/middle/late, with temporal_position="beginning","middle","end"
respectively; no relations. For local_event use "any" unless the question or timeline
establishes a position. A return query must distinguish the later visit without
inventing carrying, placement, or posture.
For whole_video each locate's scene MIDPOINT must fall in its assigned video third.
Use broad actor/object/action search terms from that period's fallible captions;
include alternatives when useful. Do not assume a later action finished a task.
The current whole-video timeline is grouped by beginning/middle/end. Choose each
query's hints from its MATCHING GROUP, not the first three scene entries. If a group
has no hints, keep the query neutral rather than copying an action from another third.

relate searches windows around the best source hit and reranks using the target
need's locate query. Use sequential r1,r2,... and the plan's dependency direction.
The anchor window can contain the subsequent action; overlap does not prove ordering.
Use relate ONLY for a dependency in the actual plan; independent needs have no
relations. Include ALL required relation outputs in inspect. Do not omit a direct
locate or delete dependencies to repair an unused-output error.
"""
    + TOOL_DATAFLOW_RULES
)


DEMONSTRATION_VERSION = "paired-synthetic-v3"


def _plan(scope, objective, needs, stop, reasoning):
    return {
        "scope": scope,
        "objective": objective,
        "evidence_needs": [
            {"id": name, "description": description, "temporal_role": role, "depends_on": deps}
            for name, description, role, deps in needs
        ],
        "max_intervals": 3,
        "max_inspected_seconds": 60,
        "stop_condition": stop,
        "reasoning": reasoning,
    }


def _program(rationale, locates, relations=()):
    calls = [
        {
            "id": f"l{i}",
            "query": query,
            "limit": 4,
            "evidence_need_ids": [need],
            "temporal_position": position,
        }
        for i, (need, query, position) in enumerate(locates, 1)
    ]
    return {
        "version": "2.0",
        "compiler": "model",
        "rationale": rationale,
        "max_intervals": 3,
        "max_inspected_seconds": 60,
        "locate": calls,
        "relate": list(relations),
        "inspect": {
            "sources": [call["id"] for call in [*calls, *relations]],
            "evidence_need_ids": [need for need, _, _ in locates],
        },
    }


# Question -> plan -> program pairs, deliberately unrelated to the demo library.
# There are no answer fields, example images, or pretend retrieved tool results.
DEMONSTRATIONS = {
    "global_workbench": {
        "question": "Across the clip, what is the person mainly working on?",
        "cached_caption_hints": [
            "Early: person holds a ruler against boards.",
            "Middle: person shifts boards on a bench.",
            "Late: person handles clamps beside the boards.",
        ],
        "plan": _plan(
            "whole_video",
            "Identify the main activity across the clip.",
            [
                (
                    "early",
                    "Representative actions and object state changes near the beginning.",
                    "context",
                    [],
                ),
                (
                    "middle",
                    "Representative actions and object state changes near the midpoint.",
                    "context",
                    [],
                ),
                (
                    "late",
                    "Representative actions and object state changes near the end.",
                    "context",
                    [],
                ),
            ],
            "Three periods support an overall activity, or show why it remains uncertain.",
            "Compare independent temporal samples without assuming the overall task.",
        ),
        "program": _program(
            "Inspect independent representatives from each video third; no event chain.",
            [
                ("early", "Person handles boards and ruler", "beginning"),
                ("middle", "Person moves or aligns boards on bench", "middle"),
                ("late", "Person handles boards and clamps", "end"),
            ],
        ),
    },
    "local_mat_purpose": {
        "question": "Why does the person carry a folded mat from the shelf to the doorway?",
        "cached_caption_hints": [
            "Person carries a folded mat toward a doorway.",
            "Later, person kneels near the mat; hand interaction unclear.",
        ],
        "plan": _plan(
            "local_event",
            "Explain the stated movement of the mat.",
            [
                (
                    "mat_arrival",
                    "The person carries the folded mat from the shelf to the doorway.",
                    "anchor",
                    [],
                ),
                (
                    "mat_interaction",
                    "What the person does with the mat at the doorway after arrival.",
                    "outcome",
                    ["mat_arrival"],
                ),
            ],
            "The subsequent same-object interaction supports a purpose or exposes a visual limit.",
            "Arrival is setup; inspect what happens to the mat without guessing the action.",
        ),
        "program": _program(
            "Find the mat's arrival and inspect the subsequent interaction with that same mat.",
            [
                ("mat_arrival", "Person carries folded mat from shelf to doorway", "any"),
                ("mat_interaction", "Person handles mat at doorway after arrival", "any"),
            ],
            [
                {
                    "id": "r1",
                    "source": "l1",
                    "relation": "after",
                    "horizon_seconds": 60,
                    "limit": 4,
                    "evidence_need_ids": ["mat_interaction"],
                }
            ],
        ),
    },
    "local_helmet_attribute": {
        "question": "What color is the helmet held by the person beside the scooter?",
        "cached_caption_hints": ["Person stands beside a scooter holding a helmet."],
        "plan": _plan(
            "local_event",
            "Identify the held helmet's visible color.",
            [
                (
                    "helmet_color",
                    "The helmet held by the person beside the scooter, with color visible.",
                    "context",
                    [],
                )
            ],
            "The specified helmet's color is visible or explicitly unclear.",
            "One object attribute is needed; there is no temporal dependency.",
        ),
        "program": _program(
            "Locate the held helmet; independent attribute inspection needs no temporal traversal.",
            [("helmet_color", "Person beside scooter holding helmet", "any")],
        ),
    },
    "local_trolley_cause": {
        "question": "Why did the person stop pulling the trolley?",
        "cached_caption_hints": [
            "Person pulls a trolley across the floor.",
            "Trolley stops beside a cable; wheel contact is unclear.",
        ],
        "plan": _plan(
            "local_event",
            "Find the visible circumstances preceding the trolley stop.",
            [
                ("trolley_stop", "The person stops pulling the trolley.", "anchor", []),
                (
                    "preceding_condition",
                    "The trolley, floor and person's actions just before the stop.",
                    "before",
                    ["trolley_stop"],
                ),
            ],
            "Preceding observations support a cause or show that it cannot be determined.",
            "Locate the stop first, then look backward without assuming the cable caused it.",
        ),
        "program": _program(
            "Inspect the trolley stop and preceding conditions, not its later outcome.",
            [
                ("trolley_stop", "Person stops pulling trolley", "any"),
                ("preceding_condition", "Trolley wheels floor and person before stopping", "any"),
            ],
            [
                {
                    "id": "r1",
                    "source": "l1",
                    "relation": "before",
                    "horizon_seconds": 60,
                    "limit": 4,
                    "evidence_need_ids": ["preceding_condition"],
                }
            ],
        ),
    },
}

# Scope balance without loading the entire compiler demonstration bank into each
# planner call. The before/cause contract is explicit in PLANNER_RULES above.
PLANNER_DEMONSTRATION_IDS = (
    "global_workbench",
    "local_mat_purpose",
    "local_helmet_attribute",
)


def _prompt_with_examples(rules, ids, role, current):
    parts = [
        rules,
        "BEGIN_SYNTHETIC_EXAMPLES\nThese fictional input/output pairs teach structure only. "
        "Their captions are fallible hints, not verified observations. Never copy their actors, "
        "objects, queries or IDs into a different task. Use only the CURRENT TASK below.",
    ]
    for example_id in ids:
        example = DEMONSTRATIONS[example_id]
        inputs = {"question": example["question"]}
        if role == "tool":
            inputs["cached_caption_hints"] = example["cached_caption_hints"]
            inputs["plan"] = example["plan"]
        output = example["plan" if role == "planner" else "program"]
        parts.append(
            f"Synthetic example {example_id}\nINPUT: "
            + json.dumps(inputs, separators=(",", ":"))
            + "\nOUTPUT: "
            + json.dumps(output, separators=(",", ":"))
        )
    parts.append("END_SYNTHETIC_EXAMPLES\nCURRENT TASK (not an example):\n" + current)
    return "\n\n".join(parts), {
        "version": DEMONSTRATION_VERSION,
        "ids": list(ids),
        "synthetic": True,
    }


def planner_prompt(search_context):
    """Balanced scope demonstrations; the model still classifies the actual question."""
    return _prompt_with_examples(
        PLANNER_RULES, PLANNER_DEMONSTRATION_IDS, "planner", search_context
    )


def tool_prompt(search_context, plan):
    """Select by the validated plan, never dataset/answer similarity or an extra LLM call."""
    if plan["scope"] == "whole_video":
        ids = ("global_workbench",)
    else:
        dependencies = [need for need in plan["evidence_needs"] if need.get("depends_on")]
        ids = tuple(
            example_id
            for role, example_id in (
                ("before", "local_trolley_cause"),
                ("after", "local_mat_purpose"),
            )
            if any(
                (need["temporal_role"] == "before") == (role == "before") for need in dependencies
            )
        ) or ("local_helmet_attribute",)
    return _prompt_with_examples(
        TOOL_RULES,
        ids,
        "tool",
        search_context + "\nPlan to compile: " + json.dumps(plan, separators=(",", ":")),
    )


EVIDENCE_CALIBRATION_RULES = """\
Keep every justification within what the selected images show. Missing evidence
is not contradiction: do not invent off-camera space, weather, actions, or object
states to rule something out. Daylight does not establish temperature or direct
sun exposure. Multiple frames of one contact do not establish repeated events.
Do not claim a defect, replacement, or successful completion unless shown.
Qualify an inferred motive as plausible/probable, and state the particular visual
limit when the cause cannot be established. A caption or inspector's claim is
not additional evidence if the images do not support it.
"""


OPEN_ANSWER_RULES = """\
Answer this open-ended video question directly in natural language inside the
answer field. Infer the most specific task or explanation supported by the
visible actions; connect observations across time when the question requires it.
You are not selecting from a list. The user has supplied everything required to
ask a free-form question. Do not request choices, invent candidates, or return a
placeholder such as None/null/N/A.

The selected images are authoritative; the observations below can be wrong.
Separate visible facts from the reasoning bridge in inference. Use limitations
for a specific missing visual fact, not for missing answer choices. For a
why/purpose question, a visible same-object action sequence can support a
reasonable purpose inference. If the reason remains underdetermined, explain
what is visible and which conclusion cannot be established; a calibrated
uncertainty statement is a valid response. Do not fabricate a cause just to give
a definite answer. Cite only actual scene IDs that support your response.
Use at most 80 words in answer and 45 words in inference.
"""


OPEN_VERIFIER_RULES = """\
Review an open-ended response to the question against the actual selected images.
Decide whether it answers the question at a supported level of detail or explains
a genuine visual limitation. No answer choices, candidate, or exact-choice match
are required. Do not reject a free-form answer for lacking those fields.
The images take precedence over fallible captions and inspector descriptions.
Check actor/object continuity, timing, scene citations, and all factual claims.
Reject a placeholder/non-answer, an invented observation, unsupported completion,
or a confident cause that the sampled evidence cannot establish. Accept a
grounded inference that is clearly qualified, or an honest explanation that a
particular cause cannot be determined from these images. Review support and
calibration, not agreement with an unavailable benchmark answer.
For accepted=true return issues=[]; otherwise name the specific answer/evidence
problem. Do not manufacture an issue solely because the question is open-ended.
"""
