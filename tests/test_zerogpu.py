"""Deterministic controller tests; model outputs here are isolated test fixtures."""

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from videomasys.domain.schemas import ToolProgram  # noqa: E402
from videomasys.zerogpu.core import (  # noqa: E402
    Caption,
    LivePipeline,
    SceneCache,
    checked_command,
    execute_program,
    prepare_video,
)


def scenes():
    return [
        {
            "index": i,
            "start": i * 20.0,
            "end": (i + 1) * 20.0,
            "frames": [f"scene{i}.jpg"],
            "caption": f"event {i}",
        }
        for i in range(4)
    ]


def program():
    return ToolProgram.model_validate(
        {
            "rationale": "Find the anchor then its outcome",
            "max_intervals": 2,
            "max_inspected_seconds": 40,
            "locate": [
                {
                    "id": "l1",
                    "query": "receive the gift",
                    "limit": 2,
                    "evidence_need_ids": ["anchor"],
                }
            ],
            "relate": [
                {
                    "id": "r1",
                    "source": "l1",
                    "relation": "after",
                    "limit": 2,
                    "horizon_seconds": 20,
                    "evidence_need_ids": ["anchor"],
                }
            ],
            "inspect": {"sources": ["l1", "r1"], "evidence_need_ids": ["anchor"]},
        }
    )


def test_real_cosine_and_temporal_execution_excludes_redundant_earlier_scene():
    vectors = np.asarray([[0, 1], [1, 0], [0.8, 0.6], [-1, 0]])
    result = execute_program(program(), scenes(), vectors, [[1, 0]])
    assert result["calls"][0]["score_type"] == "semantic_cosine"
    assert result["calls"][0]["hits"] == [
        {"scene": 1, "score": 1.0, "semantic_cosine": 1.0},
        {"scene": 2, "score": 0.8, "semantic_cosine": 0.8},
    ]
    assert [s["index"] for s in result["selected"]] == [1, 2]
    assert result["calls"][-1]["inspected_seconds"] == 40
    tiny = program()
    tiny.max_inspected_seconds = 5
    assert execute_program(tiny, scenes(), vectors, [[1, 0]])["selected"] == []


def test_cache_roundtrip_and_model_change_invalidates(tmp_path):
    cache = SceneCache(str(tmp_path / "index.sqlite"), "model-v1")
    vectors = np.eye(4, dtype=np.float32)
    assert cache.load(scenes()) is None
    cache.save(scenes(), vectors)
    stored_scenes, stored_vectors = cache.load(scenes())
    assert stored_scenes == scenes()
    np.testing.assert_equal(stored_vectors, vectors)
    assert SceneCache(cache.path, "model-v2").load(scenes()) is None
    assert cache.load(scenes()[:2]) is None


def test_invalid_model_response_is_visible_and_never_replaced():
    responses = iter(["bad JSON", '{"caption":"A visible person opens the gift."}'])
    pipeline = LivePipeline(lambda *args: next(responses), None, "test", "test")
    trace = []
    result = pipeline.structured(trace, "Caption", "Describe", Caption)
    assert result.caption == "A visible person opens the gift."
    assert len(trace[0]["attempts"]) == 2
    assert trace[0]["attempts"][0]["raw_response"] == "bad JSON"
    assert "validation_error" in trace[0]["attempts"][0]
    pipeline.generate = lambda *args: "bad JSON"
    with pytest.raises(ValueError, match="two attempts"):
        pipeline.structured(trace, "Planner", "Describe", Caption)
    assert trace[-1]["status"] == "failed"
    assert "output" not in trace[-1]


def test_generation_failure_without_response_is_preserved():
    def fail(*args):
        raise ValueError("Context budget exceeded")

    trace = []
    pipeline = LivePipeline(fail, None, "test", "test")
    with pytest.raises(ValueError, match="two attempts"):
        pipeline.structured(trace, "Caption", "Describe", Caption)
    assert trace[0]["status"] == "failed"
    assert "raw_response" not in trace[0]["attempts"][0]
    assert "Context budget exceeded" in trace[0]["attempts"][1]["prompt"]


def test_invalid_json_repair_starts_fresh_but_full_failed_generation_stays_in_trace():
    repeated_text = "isolated-repetition-marker " * 500
    invalid_json = '{"caption":"' + repeated_text
    responses = iter([invalid_json, '{"caption":"A concise isolated fixture."}'])
    prompts = []

    def generate(prompt, *args):
        prompts.append(prompt)
        return next(responses)

    pipeline = LivePipeline(generate, None, "test", "test")
    trace = []
    pipeline.structured(trace, "Caption", "Describe the isolated test.", Caption)
    assert trace[0]["attempts"][0]["raw_response"] == invalid_json
    assert "validation_error" in trace[0]["attempts"][0]
    assert "The previous output was not valid JSON. Start fresh" in prompts[1]
    assert "Your previous invalid response:" not in prompts[1]
    assert invalid_json not in prompts[1]
    assert "isolated-repetition-marker " * 20 not in prompts[1]
    assert trace[0]["output"] == {"caption": "A concise isolated fixture."}


def test_full_pipeline_trace_with_isolated_model_fixtures(tmp_path):
    plan = {
        "scope": "local_event",
        "objective": "Find what follows receiving a gift",
        "evidence_needs": [
            {"id": "anchor", "description": "Person receiving a gift", "temporal_role": "anchor"}
        ],
        "max_intervals": 2,
        "max_inspected_seconds": 40,
        "stop_condition": "Visible outcome found",
        "reasoning": "Inspect the temporal sequence",
    }
    answers = iter(
        [
            plan,
            program().model_dump(),
            *[{"scene": i, "relevant": True, "observation": "Person opens gift"} for i in [1, 2]],
            "Unwrap it",
            {
                "answer": "To unwrap it",
                "selected_choice": "Unwrap it",
                "supporting_scenes": [1, 2],
                "option_comparisons": [
                    {
                        "choice": "Unwrap it",
                        "status": "best_supported",
                        "reason": "Opening is in the isolated observation.",
                    },
                    {
                        "choice": "Move it",
                        "status": "compatible",
                        "reason": "Movement alone does not explain the fixture outcome.",
                    },
                ],
            },
            {"accepted": False, "issues": ["Test rejection must remain visible"]},
        ]
    )
    vectors = np.asarray([[0, 1], [1, 0], [0.8, 0.6], [-1, 0]])

    def generate(*args):
        value = next(answers)
        return value if isinstance(value, str) else json.dumps(value)

    pipeline = LivePipeline(
        generate,
        lambda texts: np.asarray([[1, 0]]),
        "test-model",
        "test-embedding",
    )
    bundle = {
        "scenes": scenes(),
        "trace": [],
        "duration": 80,
        "database": str(tmp_path / "cache.sqlite"),
    }
    SceneCache(bundle["database"], pipeline.signature).save(scenes(), vectors)
    updates = list(pipeline.qa(bundle, "Why pick up the gift?", ["Unwrap it", "Move it"]))
    assert len(updates) == 7
    assert [t["stage"] for t in bundle["trace"]] == [
        "Planner",
        "Tool agent",
        "Retrieval",
        "Evidence inspection",
        "Choice selection",
        "Answer",
        "Verifier",
    ]
    assert bundle["status"] == "unverified"
    assert bundle["verification"]["accepted"] is False
    inspection = bundle["trace"][3]
    assert len(inspection["calls"]) == len(inspection["attempts"]) == 2
    assert inspection["repair_count"] == 0
    assert [call["output"]["scene"] for call in inspection["calls"]] == [1, 2]
    choice_entry = bundle["trace"][4]
    assert choice_entry["attempts"][0]["raw_response"] == "Unwrap it"
    assert choice_entry["output"] == {"selected_choice": "Unwrap it"}
    answer_entry = bundle["trace"][5]
    raw_answer = json.loads(answer_entry["attempts"][0]["raw_response"])
    assert raw_answer["selected_choice"] == "Unwrap it"
    assert "selected_option" not in raw_answer
    assert answer_entry["controller_transform"] == {
        "operation": "exact_choice_text_to_zero_based_index",
        "selected_choice": "Unwrap it",
        "selected_option": 0,
    }
    assert bundle["answer"]["selected_option"] == 0


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg not installed")
def test_media_bounds_and_content_addressed_frames(tmp_path):
    clip = tmp_path / "test.mp4"
    checked_command(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=blue:s=160x120",
            "-t",
            "1",
            "-c:v",
            "libx264",
            "-threads",
            "1",
            str(clip),
        ]
    )
    bundle = prepare_video(clip, tmp_path / "cache")
    assert len(bundle["scenes"]) == 1
    assert len(bundle["scenes"][0]["frames"]) == 4
    assert all(Path(p).is_file() for p in bundle["scenes"][0]["frames"])
    assert prepare_video(clip, tmp_path / "cache")["key"] == bundle["key"]
    invalid = tmp_path / "unsafe.m3u8"
    invalid.write_text("http://internal-service/")
    with pytest.raises(ValueError, match="Upload an MP4"):
        prepare_video(invalid, tmp_path / "cache")


def test_deploy_allowlist_excludes_credentials_datasets_and_artifacts():
    source = Path(__file__).parents[1] / "scripts/deploy_zerogpu.py"
    spec = importlib.util.spec_from_file_location("deployment", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    files = module.deployment_files()
    assert {
        "README.md",
        "app.py",
        "requirements.txt",
        "packages.txt",
        "pyproject.toml",
    } <= files.keys()
    assert "src/videomasys/zerogpu/core.py" in files
    assert all(
        not name.startswith(("artifacts", ".env", ".venv", "space_library")) for name in files
    )
    assert all(path.is_file() for path in files.values())


def test_gradio_preview_is_explicitly_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("GRADIO_TEMP_DIR", str(tmp_path / "gradio"))
    pytest.importorskip("gradio")
    from videomasys.zerogpu.ui import build_demo

    demo = build_demo(cache_root=tmp_path / "cache", library_root=tmp_path / "library")
    config = demo.get_config_file()
    buttons = [c for c in config["components"] if c["type"] == "button"]
    assert buttons[0]["props"]["interactive"] is False
    assert any(
        "live analysis is disabled" in str(c["props"].get("value", ""))
        for c in config["components"]
    )
    html_components = [c for c in config["components"] if c["type"] == "html"]
    assert html_components
    assert all(c["props"]["apply_default_css"] is False for c in html_components)
    components = {
        c["props"].get("elem_id"): c for c in config["components"] if c["props"].get("elem_id")
    }
    for elem_id in ("results", "evidence-section", "run-details"):
        assert components[elem_id]["props"]["visible"] is False
    assert components["choices-panel"]["props"]["open"] is False
    assert not components["answer"]["props"].get("value")
    demo.close()


def test_library_deployment_is_explicit_and_bounded(tmp_path):
    source = Path(__file__).parents[1] / "scripts/deploy_zerogpu.py"
    spec = importlib.util.spec_from_file_location("deployment", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / "example.mp4").write_bytes(b"isolated asset fixture")
    entry = {"title": "Example", "file": "example.mp4", "question": "What happened?"}
    manifest = tmp_path / "library.json"
    manifest.write_text(json.dumps([entry]))
    files = module.deployment_files(tmp_path)
    assert files["space_library/example.mp4"] == tmp_path / "example.mp4"
    assert "space_library/library.json" in files
    assert "space_library/example.mp4" not in module.deployment_files()
    manifest.write_text(json.dumps([{**entry, "ground_truth": "Do not ship this"}]))
    with pytest.raises(ValueError, match="reference answers"):
        module.deployment_files(tmp_path)
    manifest.write_text(json.dumps([{**entry, "file": "../outside.mp4"}]))
    with pytest.raises(ValueError, match="inside"):
        module.deployment_files(tmp_path)


def test_library_selection_prefills_without_inference_and_clears_old_results(tmp_path, monkeypatch):
    monkeypatch.setenv("GRADIO_TEMP_DIR", str(tmp_path / "gradio"))
    pytest.importorskip("gradio")
    from videomasys.zerogpu.ui import IDLE_STATUS, build_demo

    (tmp_path / "example.mp4").write_bytes(b"isolated asset fixture")
    (tmp_path / "library.json").write_text(
        json.dumps(
            [
                {
                    "title": "Example",
                    "file": "example.mp4",
                    "question": "What happened?",
                    "options": ["First", "Second"],
                }
            ]
        )
    )
    demo = build_demo(cache_root=tmp_path / "cache", library_root=tmp_path)
    config = demo.get_config_file()
    radios = [c for c in config["components"] if c["type"] == "radio"]
    assert len(radios) == 1
    assert radios[0]["props"].get("value") is None
    choose = next(f.fn for f in demo.fns.values() if f.fn and f.fn.__name__ == "choose")
    dependency = next(d for d in config["dependencies"] if d["api_name"] == "choose")
    components = {
        c["props"].get("elem_id"): c["id"]
        for c in config["components"]
        if c["props"].get("elem_id")
    }
    result = choose(0)
    assert len(result) == len(dependency["outputs"])
    assert result[:3] == (str(tmp_path / "example.mp4"), "What happened?", "First\nSecond")
    by_id = dict(zip(dependency["outputs"], result, strict=True))
    assert by_id[components["status"]] == IDLE_STATUS
    assert by_id[components["answer"]] == ""
    assert by_id[components["raw-trace"]] == []
    assert by_id[components["evidence-gallery"]] == []
    for elem_id in ("results", "evidence-section", "run-details"):
        assert by_id[components[elem_id]]["visible"] is False
    assert by_id[components["choices-panel"]]["open"] is True
    cleared = choose(None)
    assert len(cleared) == len(dependency["outputs"])
    assert cleared[:3] == (None, "", "")
    assert cleared[3:-1] == result[3:-1]
    assert cleared[-1]["open"] is False
    demo.close()


@pytest.mark.parametrize("with_library", [False, True])
def test_run_click_feedback_and_error_reset(tmp_path, monkeypatch, with_library):
    from types import SimpleNamespace

    monkeypatch.setenv("GRADIO_TEMP_DIR", str(tmp_path / "gradio"))
    pytest.importorskip("gradio")
    from videomasys.zerogpu.ui import build_demo, status_panel

    if with_library:
        (tmp_path / "example.mp4").write_bytes(b"isolated asset fixture")
        (tmp_path / "library.json").write_text(
            json.dumps([{"title": "Example", "file": "example.mp4"}])
        )
    demo = build_demo(
        SimpleNamespace(model_id="isolated-ui-test", signature="test"),
        cache_root=tmp_path / "cache",
        library_root=tmp_path if with_library else tmp_path / "no-library",
    )
    config = demo.get_config_file()
    components = {
        c["props"].get("elem_id"): c["id"]
        for c in config["components"]
        if c["props"].get("elem_id")
    }
    begin = next(d for d in config["dependencies"] if d.get("js"))
    assert begin["backend_fn"] is False
    assert begin["queue"] is False
    updates = json.loads(begin["js"].removeprefix("() => "))
    assert len(updates) == len(begin["outputs"])
    assert updates[0]["interactive"] is False
    assert updates[0]["value"] == "Running analysis…"
    assert "Request sent" in updates[1]
    by_id = dict(zip(begin["outputs"], updates, strict=True))
    for elem_id in ("results", "run-details", "evidence-section"):
        assert by_id[components[elem_id]]["visible"] is False
    controls = [c["id"] for c in config["components"] if c["type"] in {"video", "textbox", "radio"}]
    assert len(controls) == (4 if with_library else 3)
    assert all(by_id[component]["interactive"] is False for component in controls)
    run = next(f.fn for f in demo.fns.values() if f.fn and f.fn.__name__ == "run")
    analysis = next(d for d in config["dependencies"] if d["api_name"] == "analyze")
    with pytest.warns(UserWarning, match="Upload a video"):
        failed = list(run(None, "Question?", "", None))[-1]
    assert len(failed) == len(analysis["outputs"])
    assert 'data-state="error"' in failed[0]
    assert "Upload a video" in failed[0]
    failed_by_id = dict(zip(analysis["outputs"], failed, strict=True))
    assert failed_by_id[components["answer"]] == ""
    assert failed_by_id[components["evidence-gallery"]] == []
    for elem_id in ("results", "evidence-section"):
        assert failed_by_id[components[elem_id]]["visible"] is False
    # Diagnostics remain available even when no answer or evidence exists.
    assert failed_by_id[components["run-details"]]["visible"] is True
    assert failed_by_id[components["raw-trace"]][-1]["status"] == "failed"
    finish = next(f.fn for f in demo.fns.values() if f.fn and f.fn.__name__ == "finish")
    finish_dependency = next(d for d in config["dependencies"] if d["api_name"] == "finish")
    restored = finish(failed[0])
    assert len(restored) == len(finish_dependency["outputs"])
    assert restored[0]["interactive"] is True
    assert restored[0]["value"] == "Run analysis"
    assert restored[1] == failed[0]
    assert all(update["interactive"] is True for update in restored[2:])
    assert 'data-state="error"' in finish(status_panel("Waiting"))[1]
    assert "&lt;script&gt;" in status_panel("<script>", "error")
    demo.close()


def test_value_changes_clear_results_without_writing_back_to_inputs(tmp_path, monkeypatch):
    monkeypatch.setenv("GRADIO_TEMP_DIR", str(tmp_path / "gradio"))
    pytest.importorskip("gradio")
    from videomasys.zerogpu.ui import IDLE_STATUS, build_demo

    demo = build_demo(cache_root=tmp_path / "cache", library_root=tmp_path / "no-library")
    config = demo.get_config_file()
    components = {c["id"]: c for c in config["components"]}
    analysis = next(d for d in config["dependencies"] if d["api_name"] == "analyze")
    resets = [
        dependency
        for dependency in config["dependencies"]
        if any(event == "change" for _, event in dependency["targets"])
    ]
    expected = {
        (component["id"], event)
        for component in config["components"]
        for event in (["change"] if component["type"] in {"textbox", "video"} else [])
    }
    actual = {tuple(target) for dependency in resets for target in dependency["targets"]}
    assert actual == expected
    editable_inputs = {
        component["id"]
        for component in config["components"]
        if component["type"] in {"video", "textbox"}
    }
    for dependency in resets:
        assert dependency["queue"] is False
        assert dependency["outputs"] == analysis["outputs"]
        # Resetting results must never write input values, so .change cannot loop.
        assert not editable_inputs.intersection(dependency["outputs"])
        result = demo.fns[dependency["id"]].fn()
        assert len(result) == len(dependency["outputs"])
        by_id = dict(zip(dependency["outputs"], result, strict=True))
        assert result[0] == IDLE_STATUS
        for component_id, value in by_id.items():
            component = components[component_id]
            if component["props"].get("elem_id") in {
                "results",
                "evidence-section",
                "run-details",
            }:
                assert value["visible"] is False
            if component["type"] in {"json", "gallery"}:
                assert value == []
            if component["type"] == "state":
                assert value == {"trace": []}
    demo.close()


@pytest.mark.parametrize("cache_hit", [False, True])
def test_live_ui_output_arity_for_every_stage_with_isolated_callbacks(
    tmp_path, monkeypatch, cache_hit
):
    from types import SimpleNamespace

    monkeypatch.setenv("GRADIO_TEMP_DIR", str(tmp_path / "gradio"))
    pytest.importorskip("gradio")
    from videomasys.zerogpu import ui

    fixture_scenes = [scenes()[0]]
    monkeypatch.setattr(
        ui,
        "prepare_video",
        lambda path, root: {
            "key": "isolated-fixture",
            "duration": 20,
            "database": str(tmp_path / "cache.sqlite"),
            "scenes": fixture_scenes,
            "trace": [],
            "cached": False,
        },
    )
    monkeypatch.setattr(
        ui,
        "SceneCache",
        lambda *args: SimpleNamespace(
            load=lambda *args: (fixture_scenes, np.eye(1)) if cache_hit else None
        ),
    )
    indexed = []

    def index(bundle):
        indexed.append(bundle["key"])
        bundle["trace"].append(
            {"stage": "Caption scene 0", "status": "complete", "output": {"caption": "Blue"}}
        )
        return bundle

    def qa(bundle, question, options):
        assert bundle["question"] == question == "What happened?"
        assert bundle["options"] == options == []
        for stage in (
            "Planner",
            "Tool agent",
            "Retrieval",
            "Evidence inspection",
            "Answer",
            "Verifier",
        ):
            output = {}
            if stage == "Retrieval":
                bundle["selected"] = fixture_scenes
                output = {"selected": fixture_scenes, "calls": []}
            elif stage == "Answer":
                bundle["answer"] = {
                    "answer": "Isolated UI fixture, not an inference result",
                    "supporting_scenes": [0],
                }
                output = bundle["answer"]
            elif stage == "Verifier":
                bundle["verification"] = {"accepted": False, "issues": ["Fixture rejection"]}
                bundle["status"] = "unverified"
                output = bundle["verification"]
            bundle["trace"].append({"stage": stage, "status": "complete", "output": output})
            yield bundle

    demo = ui.build_demo(
        SimpleNamespace(model_id="isolated-ui-test", signature="test"),
        index,
        qa,
        cache_root=tmp_path / "cache",
        library_root=tmp_path / "no-library",
    )
    config = demo.get_config_file()
    analysis = next(d for d in config["dependencies"] if d["api_name"] == "analyze")
    updates = list(demo.fns[analysis["id"]].fn("fixture.mp4", "What happened?", "", None))
    assert len(updates) == (9 if cache_hit else 10)
    assert all(len(update) == len(analysis["outputs"]) for update in updates)
    assert indexed == ([] if cache_hit else ["isolated-fixture"])
    assert 'data-state="complete"' in updates[-1][0]
    assert "Needs review" in updates[-1][0]
    assert "Fixture rejection" in updates[-1][1]
    assert "Isolated UI fixture" in updates[-1][1]
    demo.close()
