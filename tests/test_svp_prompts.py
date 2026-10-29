"""Offline prompt/dataflow contracts, not evidence of model answer accuracy."""

import copy
import json

import pytest

from videomasys.zerogpu import clean_prompts, structured_prompts, svp_prompts
from videomasys.zerogpu.core import tool_search_context
from videomasys.zerogpu.svp_retrieval import SVPPlan, SVPProgram, SVPProgramValidator


@pytest.mark.parametrize("example_id", svp_prompts.DEMONSTRATIONS)
def test_paired_examples_validate_against_real_plan_and_program_contract(example_id):
    example = svp_prompts.DEMONSTRATIONS[example_id]
    plan = SVPPlan.model_validate(example["plan"])
    program = SVPProgram.model_validate(example["program"])
    assert SVPProgramValidator().validate(program, plan) is program
    assert {"question", "plan", "program"} <= set(example)
    assert set(example) <= {"question", "plan", "program", "cached_caption_hints"}
    assert not {"answer", "observations", "results", "frames"} & set(example)
    assert len(plan.evidence_needs) <= 3
    assert program.max_intervals == 3
    assert program.max_inspected_seconds == 60


def test_scoped_example_teaches_parent_restriction_without_inspecting_broad_parent():
    program = svp_prompts.DEMONSTRATIONS["scoped_event"]["program"]
    broad, narrow = program["steps"]
    assert broad["source"] == "video"
    assert broad["evidence_need_ids"] == []
    assert narrow["source"] == broad["id"]
    assert program["inspect"]["sources"] == [narrow["id"]]
    assert broad["limit"] > narrow["limit"]


def test_direct_example_does_not_force_a_chain_for_one_identifiable_event():
    program = svp_prompts.DEMONSTRATIONS["direct_event"]["program"]
    step, = program["steps"]
    assert step["op"] == "locate" and step["source"] == "video"
    assert step["limit"] == 1
    assert program["inspect"]["sources"] == [step["id"]]


def test_temporal_example_teaches_locate_relate_then_source_scoped_locate():
    program = svp_prompts.DEMONSTRATIONS["after_event"]["program"]
    anchor, relation, followup = program["steps"]
    assert anchor["source"] == "video"
    assert relation["source"] == anchor["id"]
    assert relation["relation"] == "after"
    assert followup["source"] == relation["id"]
    assert followup["evidence_need_ids"] == ["following_action"]
    assert relation["evidence_need_ids"] == []
    assert program["inspect"]["sources"] == [anchor["id"], followup["id"]]


def test_before_dependency_points_backward_from_the_anchor():
    example = svp_prompts.DEMONSTRATIONS["before_cause"]
    anchor_need, cause_need = example["plan"]["evidence_needs"]
    anchor, preceding = example["program"]["steps"]
    assert cause_need["depends_on"] == [anchor_need["id"]]
    assert cause_need["temporal_role"] == "before"
    assert preceding["source"] == anchor["id"]
    assert preceding["relation"] == "before"
    assert cause_need["id"] in preceding["evidence_need_ids"]


def test_transport_purpose_combines_setup_and_targets_unknown_outcome():
    example = svp_prompts.DEMONSTRATIONS["transport_purpose"]
    anchor, outcome = example["plan"]["evidence_needs"]
    assert anchor["temporal_role"] == "anchor"
    assert outcome["temporal_role"] == "outcome"
    assert outcome["depends_on"] == [anchor["id"]]
    assert "lift the panel and bring it" in anchor["description"]
    assert "What does the technician then do" in outcome["description"]
    event, relation, target = example["program"]["steps"]
    assert event["source"] == "video"
    assert event["limit"] == 1
    assert "places control panel on workbench" == event["query"].removeprefix("technician ")
    assert relation["source"] == event["id"]
    assert relation["relation"] == "after"
    assert relation["horizon_seconds"] == 60
    assert target["source"] == relation["id"]
    assert example["program"]["inspect"]["sources"] == [event["id"], target["id"]]


def test_compiler_may_use_fictional_cache_action_as_hypothesis_but_planner_cannot_see_it():
    example = svp_prompts.DEMONSTRATIONS["transport_purpose"]
    context = "Question: " + example["question"]
    planner, planner_ids = svp_prompts.planner_prompt(context)
    compiler, compiler_ids = svp_prompts.tool_prompt(context, example["plan"])
    assert "transport_purpose" in planner_ids and "transport_purpose" in compiler_ids
    assert "fasteners" not in planner
    assert "fasteners" not in json.dumps(example["plan"])
    assert "technician rotates fasteners" in compiler
    assert "SEARCH HYPOTHESES" in compiler
    assert "fictional and fallible" in compiler


def test_intermediate_event_is_relocated_before_second_temporal_hop():
    program = svp_prompts.DEMONSTRATIONS["interleaved_events"]["program"]
    anchor, relation, intermediate, later, final = program["steps"]
    assert relation["source"] == anchor["id"]
    assert intermediate["op"] == "locate"
    assert intermediate["source"] == relation["id"]
    assert later["source"] == intermediate["id"]
    assert final["source"] == later["id"]
    assert program["inspect"]["sources"] == [anchor["id"], intermediate["id"], final["id"]]


def test_global_example_uses_diversity_and_corroboration_without_forced_thirds():
    example = svp_prompts.DEMONSTRATIONS["global_objective"]
    assert example["plan"]["scope"] == "whole_video"
    assert len(example["plan"]["evidence_needs"]) == 2
    assert all(not need["depends_on"] for need in example["plan"]["evidence_needs"])
    overview, detail = example["program"]["steps"]
    assert overview["op"] == "overview"
    assert overview["source"] == "video"
    assert overview["limit"] >= 3
    assert detail["source"] == overview["id"]
    assert set(example["program"]["inspect"]["sources"]) == {overview["id"], detail["id"]}
    assert "temporal_position" not in json.dumps(example)


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Why does the worker stop?", "before_cause"),
        ("What happened before the signal?", "before_cause"),
        ("How is the handle used?", "object_function"),
        ("What color is the sign?", "direct_event"),
        ("What is the overall objective?", "global_objective"),
        ("Why does the worker carry the cylinder to the stand?", "transport_purpose"),
        ("What happened after the visitor arrived following the gate opening?",
         "interleaved_events"),
    ],
)
def test_planner_teaching_selection_is_question_structural(question, expected):
    context = f"Question: {question}\nResponse mode: open-ended.\n"
    ids = svp_prompts.planner_example_ids(context)
    assert expected in ids
    assert len(ids) == len(set(ids)) == 3
    # Answer hypotheses and caption hints must not choose the demonstration pattern.
    contaminated = (
        f"Question: {question}\nAnswer choices (hypotheses): [\"why before use\"]\n"
        "Cached hints: something happens before another action; use a tool."
    )
    assert svp_prompts.planner_example_ids(contaminated) == ids


@pytest.mark.parametrize("example_id", svp_prompts.DEMONSTRATIONS)
def test_compiler_teaching_selection_uses_plan_shape_not_context_or_names(example_id):
    original = svp_prompts.DEMONSTRATIONS[example_id]["plan"]
    modified = copy.deepcopy(original)
    modified["objective"] = "Unrelated objective wording"
    for need in modified["evidence_needs"]:
        need["description"] = "Unrelated description wording"
    ids = svp_prompts.tool_example_ids(original)
    assert svp_prompts.tool_example_ids(modified) == ids
    assert len(ids) == len(set(ids)) == 2
    assert example_id in ids


@pytest.mark.parametrize("example_id", svp_prompts.DEMONSTRATIONS)
def test_builder_keeps_paired_json_and_current_input_separate_and_bounded(example_id):
    example = svp_prompts.DEMONSTRATIONS[example_id]
    context = f"Question: {example['question']}\nResponse mode: open-ended.\n"
    plan_before = copy.deepcopy(example["plan"])
    for prompt, ids in (
        svp_prompts.planner_prompt(context),
        svp_prompts.tool_prompt(context, example["plan"]),
    ):
        assert isinstance(ids, list)
        assert 2 <= len(ids) <= 3
        assert len(prompt) < 15000
        assert prompt.count("\nOUTPUT: ") == len(ids)
        current = prompt.split("END_FICTIONAL_EXAMPLES\nCURRENT TASK:\n")[1]
        assert current.startswith(context)
        assert "no verified observations or correct answers" in prompt
        for line in prompt.splitlines():
            if line.startswith("OUTPUT: "):
                json.loads(line.removeprefix("OUTPUT: "))
    assert example["plan"] == plan_before


def test_rules_teach_typed_dataflow_not_fixed_direct_locate_or_global_thirds():
    assert "one to three question-specific evidence needs" in svp_prompts.PLANNER_RULES
    assert "not a mandatory beginning/middle/end or event chain" in svp_prompts.PLANNER_RULES
    assert "combine pickup/carry/arrival into ONE" in svp_prompts.PLANNER_RULES
    assert "UNKNOWN same-object interaction" in svp_prompts.PLANNER_RULES
    rules = svp_prompts.TOOL_RULES
    for required in (
        'version="3.0"', "ordered steps", "Source is a real restriction",
        "evidence_need_ids=[]", "does NOT need a separate global",
        "intermediate candidates need", "source=A", "relation=after",
        "No arbitrary Python", "Representative sampling is not full-video viewing",
        'overview(source="video", limit>=3) directly in inspect',
        "NOT inspected video seconds", "horizon_seconds=60",
        "locate its actual event\nwithin the first relation output", "SEARCH HYPOTHESES",
        "locate -> inspect is a valid complete", "BEFORE the relation limit",
        "it never widens the horizon", "anchor the arrival/placement",
        "Do not add relate just to connect supporting",
    ):
        assert required in rules


def _scene():
    return {
        "index": 7, "start": 10.0, "end": 20.0,
        "frames": ["fixture_a.jpg", "fixture_b.jpg"], "frame_times": [12.25, 18.5],
        "detail_span": [11.0, 15.0],
    }


def test_inspector_uses_targeted_subquestions_and_real_time_mapping_without_examples():
    scene = _scene()
    needs = [{"id": "visible_contact", "description": "Which object does the hand contact?"}]
    initial = svp_prompts.inspect_prompt(scene, "User question", needs)
    refinement = svp_prompts.refine_prompt(scene, "User question", needs)
    for prompt in (initial, refinement):
        assert isinstance(prompt, str)
        assert "visible_contact" in prompt
        assert "Which object does the hand contact?" in prompt
        assert "timestamped facts" in prompt
        assert "Unobserved is not contradicted" in prompt
        assert "do not answer the overall question" in prompt
        assert "do not merge different objects" in prompt
        assert "[12.25, 18.5]" in prompt
        assert "FICTIONAL_EXAMPLES" not in prompt
        assert len(prompt) < 2200
    assert "2 chronological images" in initial
    assert "scene 7" in initial
    assert "refine_at=null" in refinement


def test_unchanged_roles_reuse_clean_to_isolate_retrieval_experiment():
    for name in (
        "caption_prompt", "choice_prompt", "answer_prompt", "verifier_prompt",
        "review_rules", "revision_prompt",
    ):
        assert getattr(svp_prompts, name) is getattr(clean_prompts, name)
    assert svp_prompts.FINAL_REVIEW_RULE == clean_prompts.FINAL_REVIEW_RULE
    assert svp_prompts.PROMPT_VERSION != clean_prompts.PROMPT_VERSION
    assert svp_prompts.PROMPT_VERSION == "svp-roles-v3"


def test_authored_demonstrations_and_new_rules_have_no_library_answer_leakage():
    texts = [
        json.dumps(svp_prompts.DEMONSTRATIONS), svp_prompts.PLANNER_RULES,
        svp_prompts.TOOL_RULES, svp_prompts.INSPECTION_RULES,
    ]
    for text in texts:
        for forbidden in (
            "nextqa", "egoschema", "unwrap", "paintbrush", "dishwashing", "sofa",
            "sponge", "gift", "4882821564", "0074f737", "00b9a0de",
        ):
            assert forbidden not in text.lower()


def test_structured_answer_prioritizes_essential_reasoning_without_hiding_exploration():
    evidence = {
        "needs": [{"id": "contact"}, {"id": "background"}],
        "observations": [{"id": "obs_contact", "frame_ids": ["frame_one"]}],
        "unresolved": [{"need_id": "background", "reason": "Outside the sampled view"}],
    }
    prompt = structured_prompts.answer_prompt("Why use the switch?", [], None, evidence)
    assert structured_prompts.PROMPT_VERSION == "structured-evidence-v7"
    assert "extra context is exploratory" in structured_prompts.PLANNER_RULES
    assert "not a report on every planned need" in prompt
    assert "not required answer premises" in prompt
    assert "Every planned need must reach" not in prompt
    assert "for each current need" not in prompt
    supplied = prompt.split("Evidence context (model observations are fallible): ")[1]
    assert json.loads(supplied) == evidence
    assert "Every claim needs direct or inherited frame support" in prompt
    assert '"answer_claim_ids":["c2"]' in prompt
    assert "choice_change_reason=''" in prompt


def test_structured_object_use_keeps_event_scope_and_brief_contact_attention():
    rules = structured_prompts.PLANNER_RULES
    assert "an object's use is local_event" in rules
    assert "Keep genuine before/after dependencies" in rules
    assert "anywhere in the video" in rules
    assert "search from video, not only from a sampled overview" in structured_prompts.TOOL_RULES
    prompt = structured_prompts.inspection_prompt("How is the tool used?", [], [])
    assert "follow the working end and its contact point" in prompt
    assert "brief or ambiguous in even one frame" in prompt
    refined = structured_prompts.inspection_prompt("How is the tool used?", [], [], refining=True)
    assert "Set refine_frame_id=null" in refined


def test_structured_choice_answer_and_review_share_symmetric_comparison():
    for prompt in (
        structured_prompts.choice_prompt("What is its use?", ["A", "B"]),
        structured_prompts.answer_prompt("What is its use?", ["A", "B"], "A", {}),
        structured_prompts.review_prompt("What is its use?", {}, {}, []),
    ):
        assert structured_prompts.OPTION_REASONING in prompt
        assert "greater\nspecificity alone is not evidence" in prompt


def test_bounded_compiler_hint_can_preserve_later_actions_without_changing_default():
    scene = {"index": 0, "start": 0., "end": 20.,
             "caption": "Background activity. " * 15 + "The tool contacts the lever."}
    full = tool_search_context("Question", [scene], "local_event", caption_limit=600)
    timeline = json.loads(full.split("not verified evidence: ")[1])
    assert timeline[0]["search_hint"] == scene["caption"]
    assert "The tool contacts the lever." not in tool_search_context(
        "Question", [scene], "local_event",
    )


def test_structured_choice_changes_and_explanation_revisions_have_distinct_instructions():
    options = ["Activate the indicator", "Disconnect the cable"]
    draft = {"selected_choice": options[0], "claims": [{"id": "c1"}]}
    for candidate in (options[0], None):
        prompt = structured_prompts.answer_prompt("Why use the switch?", options, candidate, {})
        assert "Neither is ground truth or an immutable choice" in prompt
        assert "including to or from null" in prompt
        assert "choice_change_reason citing existing support IDs" in prompt
        assert "Minor wording, missing incidental details" in prompt
        assert "Preliminary candidate: " + json.dumps(candidate) in prompt
    for scope in ("answer", "explanation"):
        feedback = {"revision_scope": scope, "issues": ["Correct the stated contact"]}
        prompt = structured_prompts.answer_prompt(
            "Why use the switch?", options, options[0], {}, draft=draft, feedback=feedback,
        )
        assert json.dumps(draft, separators=(",", ":")) in prompt
        assert json.dumps(feedback, separators=(",", ":")) in prompt
        assert "Fix each target marked revise" not in prompt
        if scope == "explanation":
            assert "preserving the main conclusion's meaning" in prompt
            assert "Keep selected_choice exactly as in the previous draft, including null" in prompt
            assert prompt.endswith("do not switch to another choice or abstain.")
        else:
            assert "Answer revision: fix the substantive answer issue" in prompt
            assert "Keep selected_choice exactly" not in prompt


def test_structured_review_checks_exact_decision_without_promoting_advisory_comments():
    draft = {
        "decision": {"selected_choice": "Activate the indicator", "conclusion": "Activation",
                     "support_claim_ids": ["c2"]},
        "decision_transition": {"previous_choice": None,
                                "selected_choice": "Activate the indicator",
                                "changed": True, "reason": "c2 supports activation"},
    }
    prompt = structured_prompts.review_prompt("Why use the switch?", {}, draft, ["c1", "c2"])
    assert "exact decision.selected_choice, displayed conclusion and support_claim_ids" in prompt
    assert "do not pass a choice when the displayed conclusion answers something else" in prompt
    assert "Check any decision_transition" in prompt
    assert (
        "compare decision_transition.previous_conclusion with the current conclusion: "
        "use answer_judgment revise if the substantive answer meaning changed, "
        "not merely its wording"
    ) in prompt
    assert "Separate answer adequacy from explanation edits" in prompt
    assert "Context or alternative-option comments alone are not answer failures" in prompt
    assert "if they undermine the conclusion" in prompt
    assert "every supplied claim target_id" in prompt
    assert prompt.endswith(json.dumps(draft, separators=(",", ":")))
