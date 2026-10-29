"""Offline profile tests; synthetic fixtures establish contracts, not model accuracy."""

import json

import numpy as np
import pytest

from videomasys.zerogpu import clean_prompts, prompts, tuned_prompts
from videomasys.zerogpu.core import CACHE_VERSION, LivePipeline, SceneCache


@pytest.mark.parametrize("profile", [None, "", "other", True, []])
def test_unknown_profile_rejected(profile):
    with pytest.raises(ValueError, match="prompt_profile"):
        LivePipeline(None, None, "fixture", "fixture", prompt_profile=profile)


def test_clean_default_and_exact_legacy_cache_identity():
    clean = LivePipeline(None, None, "fixture", "embed")
    legacy = LivePipeline(None, None, "fixture", "embed", prompt_profile="legacy")
    assert clean.prompt_profile == "clean"
    assert clean.prompt_version == clean_prompts.PROMPT_VERSION
    assert legacy.signature == f"{CACHE_VERSION}:fixture:embed"
    assert clean.signature != legacy.signature
    assert clean.prompt_version in clean.signature


def test_clean_planner_and_compiler_are_shorter_without_examples():
    context = "Question: What is the person doing?\nResponse mode: open-ended.\n"
    plan = prompts.DEMONSTRATIONS["global_workbench"]["plan"]  # Test-only typed fixture.
    for clean, legacy in (
        (clean_prompts.planner_prompt(context), prompts.planner_prompt(context)),
        (clean_prompts.tool_prompt(context, plan), prompts.tool_prompt(context, plan)),
    ):
        assert len(clean[0]) < len(legacy[0]) * 0.6
        assert clean[1] is None
        assert "SYNTHETIC_EXAMPLES" not in clean[0]
        assert "OUTPUT:" not in clean[0]
    planner = clean_prompts.planner_prompt(context)[0]
    compiler = clean_prompts.tool_prompt(context, {})[0]
    for text in (
        "whole_video",
        "three independent context",
        "depends_on=[]",
        "UNKNOWN",
        "before need depends on the anchor",
        "60 inspected seconds",
        "max_intervals=3",
        "max_inspected_seconds=60",
    ):
        assert text in planner
    for text in (
        "EVERY need",
        "direct locate",
        "matching timeline groups",
        "no relations",
        "Each dependency requires a relate",
        "TARGET relative to SOURCE",
        "retrieving B from source A uses after, not before",
        "ALL locate and relate outputs",
        "ALL need IDs",
        "do not prove event order",
    ):
        assert text in compiler


@pytest.mark.parametrize("profile", ["clean", "tuned", "legacy"])
@pytest.mark.parametrize("multiple_choice", [False, True])
def test_all_live_roles_route_profile_and_preserve_schema_and_trace(
    tmp_path,
    profile,
    multiple_choice,
):
    plan = {
        "scope": "local_event",
        "objective": "Inspect a visible object",
        "evidence_needs": [
            {
                "id": "object",
                "description": "Visible object interaction",
                "temporal_role": "context",
                "depends_on": [],
            }
        ],
        "max_intervals": 1,
        "max_inspected_seconds": 20,
        "stop_condition": "The visible action is identified",
        "reasoning": "Inspect the object",
    }
    program = {
        "version": "2.0",
        "compiler": "model",
        "rationale": "Find the object",
        "max_intervals": 1,
        "max_inspected_seconds": 20,
        "locate": [
            {
                "id": "l1",
                "query": "object",
                "limit": 2,
                "evidence_need_ids": ["object"],
                "temporal_position": "any",
            }
        ],
        "relate": [],
        "inspect": {"sources": ["l1"], "evidence_need_ids": ["object"]},
    }
    options = ["Object moves", "Object stays"] if multiple_choice else []
    answer = {
        "answer": "The visible object moves.",
        "supporting_scenes": [0],
        "inference": "Movement is visible.",
        "limitations": [],
    }
    if multiple_choice:
        answer.update(
            selected_choice=options[0],
            option_comparisons=[
                {"choice": options[0], "status": "best_supported", "reason": "Visible movement"},
                {"choice": options[1], "status": "contradicted", "reason": "Visible movement"},
            ],
        )
    outputs = [
        {"caption": "A visible object moves."},
        plan,
        program,
        {
            "scene": 0,
            "observation": "A visible object moves.",
            "relevant": True,
            "uncertainty": "",
            "refine_at": None,
        },
    ]
    if multiple_choice:
        outputs.append(options[0])
    outputs += [answer, {"accepted": True, "issues": [], "needs_revision": False}]
    pending = iter(outputs)
    calls = []

    def generate(prompt, images, tokens):
        calls.append((prompt, images, tokens))
        value = next(pending)
        return value if isinstance(value, str) else json.dumps(value)

    pipeline = LivePipeline(
        generate, lambda _: np.asarray([[1.0, 0.0]]), "fixture", "fixture", prompt_profile=profile
    )
    scene = {
        "index": 0,
        "start": 0.0,
        "end": 20.0,
        "frames": ["fixture.jpg"],
        "frame_times": [10.0],
        "caption": "",
    }
    bundle = {"scenes": [scene], "database": str(tmp_path / "cache.sqlite"), "trace": []}
    pipeline.index(bundle)
    pipeline.save_index(bundle)
    assert SceneCache(bundle["database"], pipeline.signature).load([scene])
    list(pipeline.qa(bundle, "What is visible?", options))
    assert bundle["status"] == "verified"
    model_steps = [step for step in bundle["trace"] if "model" in step]
    assert all(step["prompt_profile"] == profile for step in model_steps)
    assert all(step["prompt_version"] == pipeline.prompt_version for step in model_steps)
    assert all(step["attempts"][0]["raw_response"] for step in model_steps)
    planner = next(step for step in model_steps if step["stage"] == "Planner")
    if profile != "legacy":
        assert "prompt_examples" not in planner
        assert all("SYNTHETIC_EXAMPLES" not in prompt for prompt, *_ in calls)
        assert calls[0][0].startswith(clean_prompts.caption_prompt(scene))
        assert "Daylight" not in calls[-1][0]  # No case-specific legacy calibration.
    else:
        assert planner["prompt_examples"]["version"] == prompts.DEMONSTRATION_VERSION
        assert calls[0][0].startswith("Create a searchable action caption")
        assert "Daylight" in calls[-1][0]


def test_role_rules_do_not_inject_benchmark_answers_or_predict_missing_actions():
    scene = {
        "index": 7,
        "start": 10,
        "end": 20,
        "frames": ["fixture.jpg"],
        "frame_times": [12.5],
        "detail_span": [11, 14],
    }
    role_prompts = [
        clean_prompts.caption_prompt(scene),
        clean_prompts.inspect_prompt(scene, "Question", []),
        clean_prompts.refine_prompt(scene, "Question", []),
        clean_prompts.choice_prompt("Question", ["A", "B"]),
        clean_prompts.answer_prompt("Question", [], "{}", [7], [], None),
        clean_prompts.verifier_prompt("Question", [], "{}", {}, [], None),
        clean_prompts.revision_prompt("Question", "{}", []),
        clean_prompts.review_rules(),
    ]
    for text in role_prompts:
        assert not any(
            word in text.lower()
            for word in (
                "unwrap",
                "paintbrush",
                "dishwashing",
                "nextqa",
                "egoschema",
                "benchmark answer",
            )
        )
    assert "refine_at=null" in role_prompts[2]
    assert "not a contradicted action" in role_prompts[1]
    assert "do not invent" in role_prompts[1].lower()
    assert "Do not ask for choices" in role_prompts[4]
    assert "without requiring choices" in role_prompts[5]


def test_tuned_profile_keeps_retrieval_baseline_and_uses_separate_cache():
    clean = LivePipeline(None, None, "fixture", "embed", prompt_profile="clean")
    tuned = LivePipeline(None, None, "fixture", "embed", prompt_profile="tuned")
    assert clean.prompt_version == "clean-roles-v2"
    assert tuned.prompt_version == "grounded-roles-v2"
    assert tuned.signature != clean.signature
    assert tuned.prompt_version in tuned.signature
    assert tuned.role_prompts is tuned_prompts
    for role in ("caption_prompt", "planner_prompt", "tool_prompt", "choice_prompt"):
        assert getattr(tuned_prompts, role) is getattr(clean_prompts, role)


def test_tuned_review_is_independent_and_checks_all_fields():
    evidence = "DO_NOT_PRIME_THE_REVIEWER_WITH_THIS_DESCRIPTION"
    review = tuned_prompts.verifier_prompt("Question", [], evidence, {}, ["A", "B"], "A")
    review += "ACTUAL_DRAFT" + tuned_prompts.review_rules()
    assert evidence not in review
    assert "ALL fields" in review
    assert "limitations and choice reasons" in review
    assert "'no evidence' cannot justify contradicted" in review
    assert "sampled scene/time" in review
    assert review.index("Task:") > review.index("ACTUAL_DRAFT")
    assert len(review) < 2500


def test_tuned_visual_rules_and_example_do_not_encode_evaluation_answers():
    scene = {
        "index": 2, "start": 10, "end": 20, "frames": ["fixture.jpg"],
        "frame_times": [12.5], "detail_span": [11, 14],
    }
    role_texts = [
        tuned_prompts.inspect_prompt(scene, "Question", []),
        tuned_prompts.refine_prompt(scene, "Question", []),
        tuned_prompts.answer_prompt("Question", [], "{}", [2], ["A", "B"], "A"),
        tuned_prompts.review_rules(),
    ]
    for text in role_texts:
        for forbidden in ("unwrap", "paintbrush", "dishwashing", "nextqa", "egoschema", "sofa",
                          "sponge", "brush", "cleaner", "present"):
            assert forbidden not in text.lower()
        assert "Do not merge different objects" in text
        assert "before/after states" in text
    assert "timestamped facts" in role_texts[0]
    assert "Already beyond the gate" in role_texts[0]
    assert "refine_at=null" in role_texts[1]
    assert "Example with unrelated choices" in role_texts[2]
    assert "supporting_scenes" in role_texts[2]


def test_tuned_data_precedes_instructions_and_answer_omits_fallible_notes():
    answer = tuned_prompts.answer_prompt("Question: data\n", [], "FALLIBLE_NOTES", [2], [], None)
    assert "FALLIBLE_NOTES" not in answer
    assert answer.index("Task:") > answer.index("Image mapping:")
