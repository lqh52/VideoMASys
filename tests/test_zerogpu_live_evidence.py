"""Live-controller integration with explicitly isolated model/sampler fixtures.

No expected benchmark answer is fed to the runtime. These tests check data flow,
sampling budgets, cache invalidation, and preservation of actual model inputs.
"""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from videomasys.zerogpu import core  # noqa: E402
from videomasys.zerogpu.media_sampling import sample_timestamps  # noqa: E402


def fixture_raw(value):
    return value if isinstance(value, str) else json.dumps(value)


@pytest.mark.parametrize(
    "invalid",
    ["0", "1", "test a", "Test A because it fits.", '{"selected_choice":"Test A"}', "```Test A```"],
)
def test_choice_reader_repairs_only_to_an_exact_choice_and_preserves_raw_attempts(invalid):
    replies = iter([invalid, "Test B"])
    calls = []

    def generate(prompt, images, tokens):
        calls.append((prompt, list(images), tokens))
        return next(replies)

    pipeline = core.LivePipeline(generate, None, "test", "test")
    trace = []
    choice = pipeline.choose_option(trace, "Isolated choice?", ["Test A", "Test B"], ["frame.jpg"])
    assert choice == "Test B"
    assert len(calls) == 2 and all(call[1] == ["frame.jpg"] and call[2] == 120 for call in calls)
    entry = trace[0]
    assert entry["stage"] == "Choice selection" and entry["status"] == "complete"
    assert entry["output"] == {"selected_choice": "Test B"}
    assert entry["attempts"][0]["raw_response"] == invalid
    assert "validation_error" in entry["attempts"][0]
    assert entry["attempts"][1]["raw_response"] == "Test B"
    assert "No explanation or index" in calls[1][0]


def test_choice_reader_decodes_a_json_string_without_rewriting_choice_text():
    raw = json.dumps("Test B")
    pipeline = core.LivePipeline(lambda *args: raw, None, "test", "test")
    trace = []
    assert pipeline.choose_option(trace, "Isolated choice?", ["Test A", "Test B"], []) == "Test B"
    attempt = trace[0]["attempts"][0]
    assert attempt["raw_response"] == raw
    assert attempt["normalization"] == "decode JSON string"
    assert len(trace[0]["attempts"]) == 1


def test_choice_reader_never_interprets_repeated_numeric_answers_as_an_index():
    pipeline = core.LivePipeline(lambda *args: "0", None, "test", "test")
    trace = []
    with pytest.raises(ValueError, match="Choice selection failed validation after two attempts"):
        pipeline.choose_option(trace, "Isolated choice?", ["Test A", "Test B"], ["frame.jpg"])
    assert trace[0]["status"] == "failed" and "output" not in trace[0]
    assert len(trace[0]["attempts"]) == 2
    assert all(item["raw_response"] == "0" for item in trace[0]["attempts"])


def test_choice_reader_runtime_failure_remains_visible_without_fallback():
    def fail(*args):
        raise RuntimeError("Isolated GPU unavailable")

    pipeline = core.LivePipeline(fail, None, "test", "test")
    trace = []
    with pytest.raises(RuntimeError, match="Isolated GPU unavailable"):
        pipeline.choose_option(trace, "Isolated choice?", ["Test A", "Test B"], ["frame.jpg"])
    assert trace[0]["status"] == "failed" and "output" not in trace[0]
    assert trace[0]["attempts"][0]["error"] == "Isolated GPU unavailable"
    assert "raw_response" not in trace[0]["attempts"][0]


def fixture_scenes():
    return [
        {
            "index": i,
            "start": i * 20.0,
            "end": (i + 1) * 20.0,
            "caption": f"Isolated cache caption {i}.",
            "frames": [f"cached_{i}_{j}.jpg" for j in range(4)],
            "frame_times": sample_timestamps(i * 20.0, (i + 1) * 20.0, 4),
        }
        for i in range(9)
    ]


def install_sampler_fixture(monkeypatch):
    calls = []

    def sample(video_path, directory, scene, count):
        calls.append((video_path, directory, copy.deepcopy(scene), count))
        return dict(
            scene,
            frames=[f"fresh_{scene['index']}_{j}.jpg" for j in range(count)],
            frame_times=sample_timestamps(scene["start"], scene["end"], count),
        )

    monkeypatch.setattr(core, "sample_scene_frames", sample)
    return calls


def fixture_comparisons(options, best="Test A"):
    return [
        {
            "choice": choice,
            "status": "best_supported" if choice == best else "compatible",
            "reason": f"Isolated comparison fixture for {choice}.",
        }
        for choice in options
    ]


def fixture_whole_video_plan():
    return {
        "scope": "whole_video",
        "objective": "Review the isolated activity across time.",
        "evidence_needs": [
            {
                "id": name,
                "description": f"Observe {name} test activity.",
                "temporal_role": "context",
            }
            for name in ["early", "middle", "late"]
        ],
        "max_intervals": 3,
        "max_inspected_seconds": 60,
        "stop_condition": "Representative fixture windows observed.",
        "reasoning": "Compare independent temporal samples.",
    }


def test_open_ended_qa_skips_choice_selection_and_keeps_null_choice(tmp_path):
    plan = fixture_whole_video_plan()
    plan.update(
        scope="local_event",
        evidence_needs=plan["evidence_needs"][:1],
        max_intervals=1,
        max_inspected_seconds=20,
    )
    program = {
        "rationale": "Inspect an isolated open-ended fixture.",
        "max_intervals": 1,
        "max_inspected_seconds": 20,
        "locate": [
            {"id": "l1", "query": "test activity", "limit": 2, "evidence_need_ids": ["early"]}
        ],
        "relate": [],
        "inspect": {"sources": ["l1"], "evidence_need_ids": ["early"]},
    }
    replies = iter(
        [
            plan,
            program,
            {"scene": 0, "observation": "An isolated open-ended observation.", "relevant": True},
            {
                "answer": "An isolated open-ended answer.",
                "selected_choice": None,
                "supporting_scenes": [0],
                "option_comparisons": [],
            },
            {"accepted": True, "issues": []},
        ]
    )
    pipeline = core.LivePipeline(
        lambda *args: fixture_raw(next(replies)), lambda _: np.asarray([[1.0, 0.0]]), "test", "test"
    )
    bundle = {
        "scenes": fixture_scenes()[:1],
        "trace": [],
        "duration": 20,
        "database": str(tmp_path / "cache.sqlite"),
    }
    core.SceneCache(bundle["database"], pipeline.signature).save(
        bundle["scenes"], np.asarray([[1.0, 0.0]])
    )
    updates = list(pipeline.qa(bundle, "Describe the isolated action.", []))
    assert len(updates) == 6
    assert [entry["stage"] for entry in bundle["trace"]] == [
        "Planner",
        "Tool agent",
        "Retrieval",
        "Evidence inspection",
        "Answer",
        "Verifier",
    ]
    assert bundle["answer"]["selected_choice"] is None
    assert bundle["answer"]["selected_option"] is None
    assert bundle["answer"]["option_comparisons"] == []


@pytest.mark.parametrize("problem", ["dependency", "wrong_role", "missing_need", "missing_scope"])
def test_whole_video_plan_rejects_event_chains_and_requires_explicit_scope(problem):
    plan = fixture_whole_video_plan()
    assert core.QAPlan.model_validate(plan).scope == "whole_video"
    if problem == "dependency":
        plan["evidence_needs"][1]["depends_on"] = ["early"]
    elif problem == "wrong_role":
        plan["evidence_needs"][1]["temporal_role"] = "anchor"
    elif problem == "missing_need":
        plan["evidence_needs"].pop()
    else:
        del plan["scope"]
    with pytest.raises(ValueError, match="scope|whole_video"):
        core.QAPlan.model_validate(plan)


@pytest.mark.parametrize("problem", [None, "missing_beginning", "wrong_middle"])
def test_whole_video_compiler_must_cover_beginning_middle_and_end(problem):
    plan = fixture_whole_video_plan()
    calls = [
        {
            "id": f"l{index}",
            "query": f"Observe {need['id']} test activity",
            "limit": 2,
            "evidence_need_ids": [need["id"]],
            "temporal_position": position,
        }
        for index, (need, position) in enumerate(
            zip(plan["evidence_needs"], ["beginning", "middle", "end"], strict=True), 1
        )
    ]
    if problem == "missing_beginning":
        calls[0]["temporal_position"] = "any"
    elif problem == "wrong_middle":
        calls[1]["temporal_position"] = "end"
    program = {
        "rationale": "Independent fixture coverage.",
        "max_intervals": 3,
        "max_inspected_seconds": 60,
        "locate": calls,
        "relate": [],
        "inspect": {
            "sources": ["l1", "l2", "l3"],
            "evidence_need_ids": ["early", "middle", "late"],
        },
    }
    responses = iter([plan, program, program])
    pipeline = core.LivePipeline(lambda *args: json.dumps(next(responses)), None, "test", "test")
    bundle = {"scenes": fixture_scenes(), "trace": [], "duration": 180}
    updates = pipeline.qa(bundle, "What is the overall isolated activity?", ["Test A", "Test B"])
    next(updates)
    if problem:
        with pytest.raises(ValueError, match="Tool agent failed validation"):
            next(updates)
        assert bundle["trace"][-1]["status"] == "failed"
        assert all(
            "temporal_position=" in attempt["validation_error"]
            for attempt in bundle["trace"][-1]["attempts"]
        )
    else:
        next(updates)
        assert bundle["trace"][-1]["status"] == "complete"
    updates.close()


@pytest.mark.parametrize("repair_verdict", [False, True])
def test_unresolved_choice_cannot_receive_an_accepted_verdict(tmp_path, repair_verdict):
    from videomasys.zerogpu.presentation import render_answer

    plan = fixture_whole_video_plan()
    plan.update(
        scope="local_event",
        evidence_needs=plan["evidence_needs"][:1],
        max_intervals=1,
        max_inspected_seconds=20,
    )
    program = {
        "rationale": "Inspect one isolated fixture window.",
        "max_intervals": 1,
        "max_inspected_seconds": 20,
        "locate": [
            {"id": "l1", "query": "test activity", "limit": 2, "evidence_need_ids": ["early"]}
        ],
        "relate": [],
        "inspect": {"sources": ["l1"], "evidence_need_ids": ["early"]},
    }
    comparisons = [
        dict(item, status="unclear") for item in fixture_comparisons(["Test A", "Test B"])
    ]
    responses = iter(
        [
            plan,
            program,
            {
                "scene": 0,
                "observation": "An ambiguous isolated fixture interaction.",
                "relevant": False,
            },
            "Test A",
            {
                "answer": "Test A is tentative.",
                "selected_choice": "Test A",
                "supporting_scenes": [0],
                "option_comparisons": comparisons,
                "limitations": ["The discriminating action is not visible."],
            },
            {"accepted": True, "issues": []},
            {"accepted": False, "issues": ["The selected option remains unresolved."]}
            if repair_verdict
            else {"accepted": True, "issues": []},
        ]
    )
    pipeline = core.LivePipeline(
        lambda *args: fixture_raw(next(responses)),
        lambda _: np.asarray([[1.0, 0.0]]),
        "test",
        "test",
    )
    bundle = {
        "scenes": fixture_scenes()[:1],
        "duration": 20,
        "trace": [],
        "database": str(tmp_path / "cache.sqlite"),
    }
    core.SceneCache(bundle["database"], pipeline.signature).save(
        bundle["scenes"], np.asarray([[1.0, 0.0]])
    )
    updates = pipeline.qa(bundle, "Which isolated test activity?", ["Test A", "Test B"])
    if repair_verdict:
        list(updates)
        assert bundle["status"] == "unverified"
        assert bundle["verification"]["accepted"] is False
        assert "The selected option remains unresolved." in render_answer(bundle)
    else:
        with pytest.raises(ValueError, match="Verifier failed validation"):
            list(updates)
        assert "verification" not in bundle
        assert bundle["trace"][-1]["status"] == "failed"
    verdict = bundle["trace"][-1]
    assert verdict["stage"] == "Verifier" and len(verdict["attempts"]) == 2
    assert "unresolved" in verdict["attempts"][0]["validation_error"]
    assert json.loads(verdict["attempts"][0]["raw_response"])["accepted"] is True
    assert "Draft · not verified" in render_answer(bundle)
    assert "Model-checked" not in render_answer(bundle)
    choice = next(item for item in bundle["trace"] if item["stage"] == "Choice selection")
    assert choice["output"] == {"selected_choice": "Test A"}
    assert bundle["answer"]["selected_choice"] == "Test A"
    assert all(item["status"] == "unclear" for item in bundle["answer"]["option_comparisons"])


def test_one_relevant_refinement_replaces_only_its_six_final_frames(tmp_path, monkeypatch):
    plan = fixture_whole_video_plan()
    plan.update(scope="local_event", evidence_needs=plan["evidence_needs"][:1])
    program = {
        "rationale": "Retrieve isolated fixture windows.",
        "max_intervals": 3,
        "max_inspected_seconds": 60,
        "locate": [
            {"id": "l1", "query": "test activity", "limit": 8, "evidence_need_ids": ["early"]}
        ],
        "relate": [],
        "inspect": {"sources": ["l1"], "evidence_need_ids": ["early"]},
    }
    observations = [
        {
            "scene": i,
            "observation": f"Original isolated observation {i}.",
            "relevant": i != 2,
            "refine_at": i * 20 + 8,
        }
        for i in [2, 5, 8]
    ]
    responses = iter(
        [
            plan,
            program,
            *observations,
            {
                "scene": 5,
                "observation": "Refined isolated contact sequence.",
                "relevant": True,
                "refine_at": None,
            },
            "Test A",
            {
                "answer": "An isolated answer from refined images.",
                "selected_choice": "Test A",
                "supporting_scenes": [5],
                "option_comparisons": fixture_comparisons(["Test A", "Test B"]),
            },
            {"accepted": True, "issues": []},
        ]
    )
    model_calls, refinement_calls = [], []

    def generate(prompt, images, tokens):
        model_calls.append({"prompt": prompt, "images": list(images)})
        return fixture_raw(next(responses))

    def refine(video, directory, scene, focus_time, count, *, retain_context=False):
        refinement_calls.append((scene["index"], focus_time, count, retain_context))
        lower, upper = max(scene["start"], focus_time - 2), min(scene["end"], focus_time + 2)
        focused_count = 4 if retain_context else count
        result = dict(
            scene,
            detail_span=[lower, upper],
            frames=[
                f"detail_{scene['index']}_n{focused_count}_{j}.jpg" for j in range(focused_count)
            ],
            frame_times=sample_timestamps(lower, upper, focused_count),
        )
        if retain_context:
            pairs = list(zip(result["frame_times"], result["frames"], strict=True))
            pairs += [(scene["frame_times"][i], scene["frames"][i]) for i in [0, -1]]
            pairs.sort(key=lambda pair: pair[0])
            result.update(
                frame_times=[pair[0] for pair in pairs],
                frames=[pair[1] for pair in pairs],
                retained_context_times=[scene["frame_times"][0], scene["frame_times"][-1]],
            )
        return result

    coarse_calls = install_sampler_fixture(monkeypatch)
    monkeypatch.setattr(core, "refine_scene_frames", refine)
    pipeline = core.LivePipeline(generate, lambda _: np.asarray([[1.0, 0.0]]), "test", "test")
    bundle = {
        "scenes": fixture_scenes(),
        "trace": [],
        "duration": 180,
        "video_path": str(tmp_path / "video.mp4"),
        "frame_directory": str(tmp_path / "frames"),
        "database": str(tmp_path / "cache.sqlite"),
    }
    original_scenes = copy.deepcopy(bundle["scenes"])
    scores = [0, 0.1, 1, 0.2, 0.3, 0.9, 0.4, 0.5, 0.8]
    vectors = np.asarray([[score, (1 - score**2) ** 0.5] for score in scores])
    core.SceneCache(bundle["database"], pipeline.signature).save(bundle["scenes"], vectors)
    list(pipeline.qa(bundle, "What is the isolated interaction?", ["Test A", "Test B"]))
    assert len(coarse_calls) == 3
    assert refinement_calls == [(5, 108, 12, False), (5, 108, 6, True)]
    assert bundle["scenes"] == original_scenes
    selected = bundle["selected"]
    assert [scene["index"] for scene in selected] == [2, 5, 8]
    assert [selected[1]["start"], selected[1]["end"]] == [100, 120]
    assert selected[1]["detail_span"] == [106, 110]
    assert selected[1]["retained_context_times"] == [100.25, 119.75]
    assert selected[1]["frame_times"][0] == 100.25 and selected[1]["frame_times"][-1] == 119.75
    assert all(106 < timestamp < 110 for timestamp in selected[1]["frame_times"][1:-1])
    assert sum(len(scene["frames"]) for scene in selected) == 18
    assert sum(scene["end"] - scene["start"] for scene in selected) == 60
    assert [len(call["images"]) for call in model_calls] == [0, 0, 6, 6, 6, 12, 18, 18, 18]
    final_images = [path for scene in selected for path in scene["frames"]]
    assert (
        model_calls[-3]["images"]
        == model_calls[-2]["images"]
        == model_calls[-1]["images"]
        == final_images
    )
    assert [path for path in final_images if path.startswith("fresh_5_")] == [
        "fresh_5_0.jpg",
        "fresh_5_5.jpg",
    ]
    assert len([path for path in final_images if path.startswith("detail_5_n4_")]) == 4
    final_times = [round(timestamp, 2) for timestamp in selected[1]["frame_times"]]
    assert json.dumps(final_times) in model_calls[-2]["prompt"]
    assert json.dumps(final_times) in model_calls[-1]["prompt"]
    entry = next(item for item in bundle["trace"] if item["stage"] == "Evidence inspection")
    assert entry["refinement"] == {
        "scene": 5,
        "requested_time": 108,
        "detail_span": [106, 110],
        "sample_count": 12,
        "max_refinements": 1,
        "retained_context_times": [100.25, 119.75],
    }
    assert len(entry["calls"]) == len(entry["attempts"]) == 4 and entry["repair_count"] == 0
    assert entry["calls"][1]["output"]["observation"] == "Original isolated observation 5."
    assert entry["calls"][-1]["stage"] == "Refine scene 5"
    assert len(entry["calls"][-1]["attempts"][0]["images"]) == 12
    assert entry["output"]["evidence"][1]["observation"] == "Refined isolated contact sequence."


@pytest.mark.parametrize("focus", [30, 60.1, float("nan"), float("inf")])
def test_invalid_refinement_request_fails_visibly_without_sampling(tmp_path, monkeypatch, focus):
    selected = [fixture_scenes()[2]]
    response = {
        "scene": 2,
        "observation": "Isolated unclear interaction.",
        "relevant": True,
        "refine_at": focus,
    }

    def forbidden_sampling(*args, **kwargs):
        raise AssertionError("Invalid focus must never reach media sampling")

    monkeypatch.setattr(core, "refine_scene_frames", forbidden_sampling)
    pipeline = core.LivePipeline(lambda *args: json.dumps(response), None, "test", "test")
    trace = []
    with pytest.raises(ValueError, match="Inspect scene 2 failed validation"):
        pipeline.inspect_windows(
            trace,
            selected,
            "Isolated question?",
            ["Test action"],
            video_path=tmp_path / "video.mp4",
            directory=tmp_path / "frames",
        )
    entry = trace[0]
    assert entry["status"] == "failed" and entry["repair_count"] == 1
    assert len(entry["calls"]) == 1 and len(entry["attempts"]) == 2
    assert all(
        "refine_at must be a time inside" in attempt["validation_error"]
        for attempt in entry["attempts"]
    )
    assert all(attempt["raw_response"] == json.dumps(response) for attempt in entry["attempts"])
    assert "refinement" not in entry


@pytest.mark.parametrize("max_intervals", [1, 2, 3])
@pytest.mark.parametrize(
    "repair_kind",
    [
        "none",
        "citation",
        "unlisted_choice",
        "candidate_switch",
        "missing_comparison",
        "duplicate_comparison",
        "competing_best",
        "unresolved_without_limit",
    ],
)
@pytest.mark.parametrize("reverse_options", [False, True])
def test_only_selected_windows_get_dense_frames_and_answer_sees_same_evidence(
    tmp_path, monkeypatch, max_intervals, repair_kind, reverse_options
):
    selected_ids = [2, 5, 8][:max_intervals]
    options = ["Test A", "Test B"]
    if reverse_options:
        options.reverse()
    plan = {
        "scope": "local_event",
        "objective": "Inspect the test activity.",
        "evidence_needs": [
            {
                "id": "activity",
                "description": "Observe the test activity.",
                "temporal_role": "context",
            }
        ],
        "max_intervals": max_intervals,
        "max_inspected_seconds": max_intervals * 20,
        "stop_condition": "Requested fixture evidence available.",
        "reasoning": "This is a controller fixture, not a benchmark answer.",
    }
    program = {
        "rationale": "Select the strongest isolated fixture matches.",
        "max_intervals": max_intervals,
        "max_inspected_seconds": max_intervals * 20,
        "locate": [
            {"id": "l1", "query": "test activity", "limit": 8, "evidence_need_ids": ["activity"]}
        ],
        "relate": [],
        "inspect": {"sources": ["l1"], "evidence_need_ids": ["activity"]},
    }
    inspection = {
        "evidence": [
            {
                "scene": i,
                "observation": f"Isolated visible test action in scene {i}.",
                "relevant": i != selected_ids[0],
                "uncertainty": "Fixture deliberately omits an off-frame detail.",
            }
            for i in selected_ids
        ]
    }
    invalid_comparisons = fixture_comparisons(options)
    if repair_kind == "missing_comparison":
        invalid_comparisons = invalid_comparisons[:1]
    elif repair_kind == "duplicate_comparison":
        invalid_comparisons = [invalid_comparisons[0], invalid_comparisons[0]]
    elif repair_kind == "competing_best":
        invalid_comparisons = [dict(item, status="best_supported") for item in invalid_comparisons]
    elif repair_kind == "unresolved_without_limit":
        invalid_comparisons = [dict(item, status="unclear") for item in invalid_comparisons]
    responses = iter(
        [
            plan,
            program,
            *inspection["evidence"],
            "Test A",
            *(
                [
                    {
                        "answer": "Invalid structured answer fixture.",
                        "selected_choice": "Unlisted choice fixture"
                        if repair_kind == "unlisted_choice"
                        else "Test B"
                        if repair_kind == "candidate_switch"
                        else "Test A",
                        "supporting_scenes": [999] if repair_kind == "citation" else selected_ids,
                        "option_comparisons": invalid_comparisons,
                    }
                ]
                if repair_kind != "none"
                else []
            ),
            {
                "answer": "Isolated fixture answer.",
                "selected_choice": "Test A",
                "supporting_scenes": selected_ids,
                "inference": "Isolated inference from test actions.",
                "option_comparisons": fixture_comparisons(options),
            },
            {"accepted": True, "issues": []},
        ]
    )
    model_calls = []

    def generate(prompt, images, tokens):
        model_calls.append({"prompt": prompt, "images": list(images), "tokens": tokens})
        return fixture_raw(next(responses))

    pipeline = core.LivePipeline(
        generate, lambda _: np.asarray([[1.0, 0.0]]), "test-model", "test-embed"
    )
    bundle = {
        "scenes": fixture_scenes(),
        "trace": [],
        "duration": 180.0,
        "database": str(tmp_path / "index.sqlite"),
        "video_path": str(tmp_path / "source.mp4"),
        "frame_directory": str(tmp_path / "frames"),
    }
    original_scenes = copy.deepcopy(bundle["scenes"])
    vectors = np.asarray(
        [[score, (1 - score**2) ** 0.5] for score in [0, 0.1, 1, 0.2, 0.3, 0.9, 0.4, 0.5, 0.8]]
    )
    core.SceneCache(bundle["database"], pipeline.signature).save(bundle["scenes"], vectors)
    sampling_calls = install_sampler_fixture(monkeypatch)
    updates = pipeline.qa(bundle, "What happens in this isolated test?", options)
    for _ in range(3):
        next(updates)  # Planner, compiler, and actual vector retrieval finish first.
    assert sampling_calls == []
    assert [scene["index"] for scene in bundle["selected"]] == selected_ids
    next(updates)  # Inspection triggers targeted sampling, not a full-video rescan.
    assert len(sampling_calls) == max_intervals <= 3
    assert [call[2]["index"] for call in sampling_calls] == selected_ids
    assert all(call[3] == 6 for call in sampling_calls)
    assert all(call[0] == Path(bundle["video_path"]) for call in sampling_calls)
    assert all(call[1] == Path(bundle["frame_directory"]) for call in sampling_calls)
    assert bundle["scenes"] == original_scenes
    assert sum(len(scene["frames"]) for scene in bundle["selected"]) == max_intervals * 6 <= 18
    assert sum(scene["end"] - scene["start"] for scene in bundle["selected"]) <= 60
    for scene in bundle["selected"]:
        assert scene["frame_times"][0] == scene["start"] + 0.25
        assert scene["frame_times"][-1] == scene["end"] - 0.25
    list(updates)
    inspection_calls = model_calls[2 : 2 + max_intervals]
    choice_call = model_calls[2 + max_intervals]
    answer_call, verifier_call = model_calls[-2:]
    expected_images = [path for scene in bundle["selected"] for path in scene["frames"]]
    assert [path for call in inspection_calls for path in call["images"]] == expected_images
    assert (
        choice_call["images"] == answer_call["images"] == verifier_call["images"] == expected_images
    )
    assert json.dumps(options) in choice_call["prompt"]
    for scene, call in zip(bundle["selected"], inspection_calls, strict=True):
        assert call["images"] == scene["frames"]
        assert f"ALL from scene {scene['index']}" in call["prompt"]
        assert str([round(timestamp, 2) for timestamp in scene["frame_times"]]) in call["prompt"]
        assert "Zero-based choices" not in call["prompt"]
        contract = json.loads(call["prompt"].split("The required JSON schema is:\n", 1)[1])
        assert contract["properties"]["scene"]["const"] == scene["index"]
    assert all(path.startswith("fresh_") for path in expected_images)
    mapping = [
        {
            "scene": scene["index"],
            "span": [scene["start"], scene["end"]],
            "image_numbers": list(range(position * 6 + 1, position * 6 + 7)),
            "sample_times": [round(timestamp, 2) for timestamp in scene["frame_times"]],
        }
        for position, scene in enumerate(bundle["selected"])
    ]
    for call in (answer_call, verifier_call):
        assert json.dumps(mapping) in call["prompt"]
    assert all(not model_calls[i]["images"] for i in [0, 1])
    attempts = {
        entry["stage"]: entry["attempts"] for entry in bundle["trace"] if "attempts" in entry
    }
    assert [
        path for attempt in attempts["Evidence inspection"] for path in attempt["images"]
    ] == expected_images
    assert (
        attempts["Choice selection"][0]["images"]
        == attempts["Answer"][-1]["images"]
        == attempts["Verifier"][-1]["images"]
        == expected_images
    )
    answer_contract = json.loads(
        attempts["Answer"][0]["prompt"].split("The required JSON schema is:\n", 1)[1]
    )
    citation_schema = answer_contract["properties"]["supporting_scenes"]["items"]
    assert citation_schema.get("enum", [citation_schema.get("const")]) == selected_ids
    assert len(attempts["Answer"]) == 1 + int(repair_kind != "none")
    if repair_kind == "citation":
        assert "999" in attempts["Answer"][0]["raw_response"]
        assert "validation_error" in attempts["Answer"][0]
    if repair_kind == "unlisted_choice":
        assert "Unlisted choice fixture" in attempts["Answer"][0]["raw_response"]
        assert "literal_error" in attempts["Answer"][0]["validation_error"]
    if repair_kind == "candidate_switch":
        assert json.loads(attempts["Answer"][0]["raw_response"])["selected_choice"] == "Test B"
        assert "literal_error" in attempts["Answer"][0]["validation_error"]
    if repair_kind in {
        "missing_comparison",
        "duplicate_comparison",
        "competing_best",
        "unresolved_without_limit",
    }:
        assert "validation_error" in attempts["Answer"][0]
    comparison_schema = answer_contract["properties"]["option_comparisons"]
    assert comparison_schema["minItems"] == comparison_schema["maxItems"] == len(options)
    assert set(item["choice"] for item in bundle["answer"]["option_comparisons"]) == set(options)
    assert answer_contract["properties"]["selected_choice"]["const"] == "Test A"
    assert attempts["Choice selection"][0]["raw_response"] == "Test A"
    assert "selected_option" not in answer_contract["properties"]
    raw_answer = json.loads(attempts["Answer"][-1]["raw_response"])
    assert raw_answer["selected_choice"] == "Test A" and "selected_option" not in raw_answer
    answer_entry = next(entry for entry in bundle["trace"] if entry["stage"] == "Answer")
    expected_index = options.index("Test A")
    assert answer_entry["controller_transform"] == {
        "operation": "exact_choice_text_to_zero_based_index",
        "selected_choice": "Test A",
        "selected_option": expected_index,
    }
    assert answer_entry["output"] == bundle["answer"]
    assert bundle["answer"]["selected_option"] == expected_index
    assert bundle["answer"]["selected_choice"] == "Test A"
    assert "The draft selects EXACTLY this option: 'Test A'" in verifier_call["prompt"]
    inspection_entry = next(
        entry for entry in bundle["trace"] if entry["stage"] == "Evidence inspection"
    )
    assert len(inspection_entry["calls"]) == max_intervals
    assert inspection_entry["repair_count"] == 0
    assert inspection_entry["output"]["evidence"][0]["relevant"] is False
    assert selected_ids[0] in bundle["answer"]["supporting_scenes"]
    assert bundle["answer"]["inference"] == "Isolated inference from test actions."
    assert bundle["status"] == "verified"


def test_per_scene_inspection_repairs_wrong_ids_without_hiding_real_calls():
    selected = [fixture_scenes()[i] for i in [2, 5, 8]]
    values = iter(
        [
            {"scene": 999, "observation": "Wrong-ID fixture.", "relevant": True},
            *[
                {"scene": i, "observation": "Correct-ID fixture.", "relevant": True}
                for i in [2, 5, 8]
            ],
        ]
    )
    pipeline = core.LivePipeline(lambda *args: json.dumps(next(values)), None, "test", "test")
    trace = []
    result = pipeline.inspect_windows(trace, selected, "Isolated question?", ["Test activity"])
    assert [item.scene for item in result.evidence] == [2, 5, 8]
    entry = trace[0]
    assert entry["status"] == "complete"
    assert len(entry["calls"]) == 3 and len(entry["attempts"]) == 4
    assert entry["repair_count"] == 1
    assert [item["scene"] for item in entry["attempts"]] == [2, 2, 5, 8]
    assert [item["model_call"] for item in entry["attempts"]] == [1, 1, 2, 3]
    assert "999" in entry["attempts"][0]["raw_response"]
    assert "validation_error" in entry["attempts"][0]


@pytest.mark.parametrize("failure", ["invalid_response", "runtime_error"])
def test_partial_inspection_failure_keeps_completed_and_failed_call_records(failure):
    selected = [fixture_scenes()[i] for i in [2, 5, 8]]
    calls = []

    def generate(prompt, images, tokens):
        calls.append(list(images))
        if len(calls) == 1:
            return '{"scene":2,"observation":"Completed isolated observation.","relevant":true}'
        if failure == "runtime_error":
            raise RuntimeError("Isolated GPU failure")
        return '{"scene":999,"observation":"Invalid isolated observation.","relevant":true}'

    pipeline = core.LivePipeline(generate, None, "test", "test")
    trace = []
    with pytest.raises((ValueError, RuntimeError)):
        pipeline.inspect_windows(trace, selected, "Isolated question?", ["Test activity"])
    entry = trace[0]
    assert entry["status"] == "failed" and "output" not in entry
    assert len(entry["calls"]) == 2
    assert entry["calls"][0]["status"] == "complete"
    assert entry["calls"][0]["output"]["observation"] == "Completed isolated observation."
    assert entry["calls"][1]["status"] == "failed"
    assert entry["repair_count"] == (1 if failure == "invalid_response" else 0)
    assert all(item["scene"] in [2, 5] for item in entry["attempts"])
    assert all(images != selected[2]["frames"] for images in calls)
    error_key = "validation_error" if failure == "invalid_response" else "error"
    assert error_key in entry["attempts"][-1]


def test_prepare_video_records_source_and_endpoint_aware_cache_profile(tmp_path, monkeypatch):
    video = tmp_path / "source.mp4"
    video.write_bytes(b"Isolated probe fixture; not a playable video")
    monkeypatch.setattr(
        core,
        "checked_command",
        lambda _: json.dumps(
            {
                "streams": [{"codec_type": "video", "width": 640, "height": 480}],
                "format": {"duration": "90"},
            }
        ).encode(),
    )
    calls = install_sampler_fixture(monkeypatch)
    bundle = core.prepare_video(video, tmp_path / "cache")
    assert bundle["video_path"] == str(video.resolve())
    assert Path(bundle["frame_directory"]).is_dir()
    assert len(calls) == len(bundle["scenes"]) == 5
    assert all(call[3] == 4 for call in calls)
    assert all(len(scene["frames"]) == 4 for scene in bundle["scenes"])
    assert bundle["scenes"][0]["frame_times"][0] == 0.25
    assert bundle["scenes"][-1]["frame_times"][-1] == 89.75


@pytest.mark.parametrize(
    "old_version", ["zerogpu-scenes-v1", "zerogpu-scenes-v2-actions-endpoints"]
)
def test_cache_version_invalidates_old_captions_and_new_captioning_uses_four_frames(
    tmp_path, old_version
):
    scenes = fixture_scenes()
    database = str(tmp_path / "index.sqlite")
    old_signature = f"{old_version}:test-model:test-embed"
    core.SceneCache(database, old_signature).save(scenes, np.eye(len(scenes)))
    calls = []

    def generate(prompt, images, tokens):
        calls.append((prompt, list(images)))
        return '{"caption":"New isolated action caption."}'

    pipeline = core.LivePipeline(
        generate, lambda texts: np.eye(len(texts)), "test-model", "test-embed"
    )
    assert pipeline.signature != old_signature
    assert core.CACHE_VERSION in pipeline.signature
    assert core.SceneCache(database, pipeline.signature).load(scenes) is None
    bundle = {"scenes": scenes, "trace": [], "database": database}
    pipeline.index(bundle)
    assert len(calls) == len(scenes)
    for scene, (prompt, images) in zip(scenes, calls, strict=True):
        assert len(images) == 4 and images == scene["frames"]
        assert str(scene["frame_times"]) in prompt
    pipeline.save_index(bundle)
    assert core.SceneCache(database, pipeline.signature).load(scenes) is not None
    assert core.SceneCache(database, old_signature).load(scenes) is None


def test_benchmark_reference_artifacts_are_not_deployed_or_in_runtime_prompts():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "space_library/library.json").read_text())
    allowed_keys = {"title", "file", "dataset", "video_id", "question", "options", "thumbnail"}
    assert all(set(example) <= allowed_keys for example in manifest)
    runtime = "\n".join(path.read_text() for path in (root / "src/videomasys/zerogpu").glob("*.py"))
    for forbidden in [
        "subset_answers.json",
        "video_audit.md",
        "/scratch/DataSets",
        "/scratch/Projects",
    ]:
        assert forbidden not in runtime
    assert all(example["video_id"] not in runtime for example in manifest)
    spec = importlib.util.spec_from_file_location(
        "isolated_deploy_check", root / "scripts/deploy_zerogpu.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    files = module.deployment_files(root / "space_library")
    assert all(not name.startswith(("artifacts/", "tests/")) for name in files)
    assert all(not path.is_relative_to(root / "artifacts/evaluation") for path in files.values())
