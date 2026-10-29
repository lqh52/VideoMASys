"""Synthetic prompt/controller contract tests, not model accuracy measurements."""

import copy
import json

import numpy as np
import pytest

from videomasys.agents.tool_sequence import ToolProgramValidator
from videomasys.domain.schemas import ToolProgram
from videomasys.zerogpu import core
from videomasys.zerogpu.prompts import (
    DEMONSTRATION_VERSION,
    DEMONSTRATIONS,
    PLANNER_DEMONSTRATION_IDS,
    planner_prompt,
    tool_prompt,
)


@pytest.mark.parametrize("example_id", DEMONSTRATIONS)
def test_every_paired_example_passes_production_plan_and_program_contracts(example_id):
    example = DEMONSTRATIONS[example_id]
    assert set(example) == {"question", "cached_caption_hints", "plan", "program"}
    plan = core.QAPlan.model_validate(example["plan"])
    program = ToolProgram.model_validate(example["program"])
    ToolProgramValidator().validate(program, plan)
    assert plan.max_intervals == program.max_intervals == 3
    assert plan.max_inspected_seconds == program.max_inspected_seconds == 60
    assert all(len(need.description.split()) < 25 for need in plan.evidence_needs)


def test_planner_examples_are_actual_question_to_plan_pairs_not_output_only_templates():
    current = "Question: Isolated current task without an answer."
    prompt, metadata = planner_prompt(current)
    assert metadata == {
        "version": DEMONSTRATION_VERSION,
        "ids": list(PLANNER_DEMONSTRATION_IDS),
        "synthetic": True,
    }
    pairs = prompt.split("Synthetic example ")[1:]
    assert len(pairs) == 3
    for pair in pairs:
        example_id = pair.splitlines()[0]
        inputs = json.loads(pair.split("\nINPUT: ")[1].splitlines()[0])
        output = json.loads(pair.split("\nOUTPUT: ")[1].splitlines()[0])
        assert inputs["question"] == DEMONSTRATIONS[example_id]["question"]
        assert "cached_caption_hints" not in inputs
        assert output == DEMONSTRATIONS[example_id]["plan"]
        assert "program" not in inputs and "plan" not in inputs
    assert prompt.endswith("CURRENT TASK (not an example):\n" + current)


@pytest.mark.parametrize("example_id", DEMONSTRATIONS)
def test_compiler_selects_one_plan_matched_example_and_keeps_real_task_last(example_id):
    plan = copy.deepcopy(DEMONSTRATIONS[example_id]["plan"])
    plan["objective"] = "An isolated current objective, not the demonstration."
    prompt, metadata = tool_prompt("Question: isolated current input", plan)
    assert metadata["ids"] == [example_id]
    assert prompt.count("\nINPUT: ") == prompt.count("\nOUTPUT: ") == 1
    inputs = json.loads(prompt.split("\nINPUT: ")[1].splitlines()[0])
    output = json.loads(prompt.split("\nOUTPUT: ")[1].splitlines()[0])
    assert inputs["plan"] == DEMONSTRATIONS[example_id]["plan"]
    assert output == DEMONSTRATIONS[example_id]["program"]
    assert json.loads(prompt.split("\nPlan to compile: ")[1]) == plan
    if plan["scope"] == "whole_video":
        assert "local_mat_purpose" not in prompt and output["relate"] == []


def test_example_metadata_and_real_validation_repair_remain_in_trace():
    example = copy.deepcopy(DEMONSTRATIONS["global_workbench"])
    invalid = copy.deepcopy(example["program"])
    invalid["relate"] = [
        {
            "id": "r1",
            "source": "l1",
            "relation": "after",
            "horizon_seconds": 60,
            "limit": 4,
            "evidence_need_ids": ["middle"],
        },
        {
            "id": "r2",
            "source": "l2",
            "relation": "after",
            "horizon_seconds": 60,
            "limit": 4,
            "evidence_need_ids": ["late"],
        },
    ]
    invalid["inspect"]["sources"] += ["r1", "r2"]
    replies = iter([example["plan"], invalid, example["program"]])
    pipeline = core.LivePipeline(
        lambda *args: json.dumps(next(replies)), None, "fixture", "fixture", prompt_profile="legacy"
    )
    caption = "An isolated cached-caption marker that must not prime the planner."
    bundle = {
        "trace": [],
        "scenes": [
            {
                "index": 0,
                "start": 0,
                "end": 90,
                "caption": caption,
                "frames": [],
            }
        ],
    }
    updates = pipeline.qa(bundle, "What is the overall isolated activity?", [])
    next(updates)
    next(updates)
    updates.close()
    planner, compiler = bundle["trace"]
    assert planner["prompt_examples"]["ids"] == list(PLANNER_DEMONSTRATION_IDS)
    assert compiler["prompt_examples"]["ids"] == ["global_workbench"]
    assert compiler["output"]["relate"] == [] and len(compiler["attempts"]) == 2
    assert caption not in planner["attempts"][0]["prompt"]
    assert "Fallible cached timeline" not in planner["attempts"][0]["prompt"]
    assert all(caption in attempt["prompt"] for attempt in compiler["attempts"])
    assert json.loads(compiler["attempts"][0]["raw_response"]) == invalid
    assert "whole_video forbids relate" in compiler["attempts"][0]["validation_error"]
    assert "whole_video forbids relate" in compiler["attempts"][1]["prompt"]


def test_mixed_temporal_plan_gets_both_directions_without_global_example():
    plan = copy.deepcopy(DEMONSTRATIONS["local_trolley_cause"]["plan"])
    plan["evidence_needs"].append(
        {
            "id": "following_action",
            "description": "What happens after the trolley stops.",
            "temporal_role": "after",
            "depends_on": ["trolley_stop"],
        }
    )
    core.QAPlan.model_validate(plan)
    prompt, metadata = tool_prompt("Question: Isolated before and after comparison.", plan)
    assert metadata["ids"] == ["local_trolley_cause", "local_mat_purpose"]
    assert "Synthetic example global_workbench" not in prompt
    assert '"relation":"before"' in prompt and '"relation":"after"' in prompt


def test_compiler_timeline_groups_match_executor_midpoints_not_first_three_windows():
    scenes = [
        {"index": i, "start": i * 20, "end": (i + 1) * 20, "caption": f"Isolated action {i}."}
        for i in range(9)
    ]
    current = "Question: An isolated overall objective?"
    context = core.tool_search_context(current, scenes, "whole_video")
    timeline = json.loads(context.split("not verified evidence: ")[1])
    assert timeline["duration_seconds"] == 180
    groups = timeline["groups_by_window_midpoint"]
    for position, expected in (("beginning", [0, 1, 2]), ("middle", [3, 4, 5]), ("end", [6, 7, 8])):
        assert [scene["scene"] for scene in groups[position]["scenes"]] == expected
    # Exact boundary is assigned to the later third; a straddler is never in both.
    scenes[0].update(start=50, end=70)
    context = core.tool_search_context(current, scenes, "whole_video")
    groups = json.loads(context.split("not verified evidence: ")[1])["groups_by_window_midpoint"]
    assert 0 not in [scene["scene"] for scene in groups["beginning"]["scenes"]]
    assert 0 in [scene["scene"] for scene in groups["middle"]["scenes"]]
    local = core.tool_search_context(current, scenes, "local_event")
    assert isinstance(json.loads(local.split("not verified evidence: ")[1]), list)


def test_failed_global_coverage_is_visible_and_never_calls_inspector_or_answerer(tmp_path):
    example = DEMONSTRATIONS["global_workbench"]
    replies = iter([example["plan"], example["program"]])
    model_calls = []

    def generate(prompt, images, tokens):
        model_calls.append(prompt)
        return json.dumps(next(replies))

    pipeline = core.LivePipeline(generate, lambda texts: np.ones((len(texts), 2)), "test", "test")
    bundle = {
        "trace": [],
        "duration": 40,
        "database": str(tmp_path / "index.sqlite"),
        "scenes": [
            {
                "index": i,
                "start": i * 20,
                "end": (i + 1) * 20,
                "frames": [],
                "caption": "Isolated caption.",
            }
            for i in range(2)
        ],
    }
    core.SceneCache(bundle["database"], pipeline.signature).save(bundle["scenes"], np.ones((2, 2)))
    with pytest.raises(ValueError, match="coverage is incomplete.*middle"):
        list(pipeline.qa(bundle, "What is the overall activity?", []))
    assert len(model_calls) == 2 and "answer" not in bundle
    retrieval = bundle["trace"][-1]
    assert retrieval["stage"] == "Retrieval" and retrieval["status"] == "failed"
    assert retrieval["output"]["temporal_coverage"]["complete"] is False
    assert retrieval["output"]["temporal_coverage"]["missing_positions"] == ["middle"]


@pytest.mark.parametrize("duration,count", [(1, 1), (3, 3), (20, 3), (40, 3), (90, 5), (180, 9)])
def test_short_uploads_prepare_distinct_representatives_without_reindexing_longer_ones(
    tmp_path,
    monkeypatch,
    duration,
    count,
):
    source = tmp_path / "fixture.mp4"
    source.write_bytes(b"Nonplayable synthetic fixture; metadata and sampling are isolated.")
    monkeypatch.setattr(
        core,
        "checked_command",
        lambda args: json.dumps(
            {
                "streams": [{"codec_type": "video", "width": 640, "height": 480}],
                "format": {"duration": str(duration)},
            }
        ).encode(),
    )
    monkeypatch.setattr(core, "sample_scene_frames", lambda path, directory, scene, count: scene)
    bundle = core.prepare_video(source, tmp_path / "cache")
    assert len(bundle["scenes"]) == count
    if duration >= 3:
        positions = {
            min(2, int(3 * (scene["start"] + scene["end"]) / 2 / duration))
            for scene in bundle["scenes"]
        }
        assert positions == {0, 1, 2}
    if duration in {20, 40}:
        cache = core.SceneCache(bundle["database"], "unchanged-caption-model")
        old = bundle["scenes"][: duration // 20]
        cache.save(old, np.ones((len(old), 2)))
        assert cache.load(bundle["scenes"]) is None
