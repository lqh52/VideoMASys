"""Offline semantic-editor fixtures exercise real retrieval and evidence validation."""

import json
from copy import deepcopy

import numpy as np
import pytest
from PIL import Image

from videomasys.domain.visual_evidence import ObservationRecord
from videomasys.zerogpu import editing_agent as agent
from videomasys.zerogpu import structured_pipeline as inspection
from videomasys.zerogpu.core import LivePipeline, SceneCache
from videomasys.zerogpu.openai import strict_json_schema


def scene(tmp_path, index=1, *, prefix="original"):
    paths = []
    for offset in range(6):
        path = tmp_path / f"{prefix}_{index}_{offset}.jpg"
        color = (index * 40, offset * 30, 20 if prefix == "original" else 180)
        Image.new("RGB", (20, 20), color).save(path)
        paths.append(str(path))
    return {"index": index, "start": index * 10.0, "end": (index + 1) * 10.0,
            "frames": paths, "frame_times": [index * 10.0 + 0.5 + i * 1.8 for i in range(6)],
            "caption": "A person opens a box" if index else "Unrelated empty background"}


def plan():
    return {"scope": "local_event", "objective": "Find the requested box interaction",
            "evidence_needs": [{"id": "box_use", "description": "How is the box opened?",
                                "temporal_role": "anchor", "depends_on": []}],
            "max_intervals": 3, "max_inspected_seconds": 60.0,
            "option_strategy": None, "stop_condition": "A matching action is visible",
            "reasoning": "Inspect the requested interaction"}


def program():
    return {"version": "3.0", "compiler": "model", "rationale": "Find box opening",
            "max_intervals": 3, "max_inspected_seconds": 60.0,
            "steps": [{"op": "locate", "id": "box", "source": "video",
                       "query": "person opens box", "limit": 1, "evidence_need_ids": ["box_use"]}],
            "inspect": {"sources": ["box"], "evidence_need_ids": ["box_use"]}}


def cut(frames):
    return {"status": "supported", "reason": "Requested interaction is visible", "clips": [{
        "start_frame_id": frames[1].id, "end_frame_id": frames[3].id,
        "label": "Box opening", "mute": False, "frame_ids": [frames[2].id],
        "observation_ids": [], "need_ids": ["box_use"], "support": "The hand opens the box.",
    }]}


def fixture(tmp_path, *, proposal=None, refine=False):
    scenes = [scene(tmp_path, index) for index in range(2)]
    key = "a" * 64
    frames = inspection.frame_registry(scenes[1], key)
    proposals = iter(proposal if proposal is not None else [cut(frames)])
    requests = []

    def generate(prompt, images, tokens, schema):
        requests.append({"prompt": prompt, "images": list(images), "schema": schema})
        properties = schema["properties"]
        if "objective" in properties:
            output = plan()
        elif "steps" in properties:
            output = program()
        elif "facts" in properties:
            actual = json.loads(prompt.split("Frames in image order: ", 1)[1].split("\n", 1)[0])
            output = {"facts": [{"kind": "action", "subject": "hand", "predicate": "opens",
                        "object": "box", "frame_ids": [actual[2]["id"]],
                        "need_ids": ["box_use"], "uncertainty": ""}],
                      "unresolved": [], "refine_frame_id": (
                          actual[2]["id"] if refine and "final close look" not in prompt else None
                      )}
        else:
            assert "clips" in properties  # No QA selector, answer or verifier is called.
            output = next(proposals)
        return json.dumps(output)

    def embed(texts):
        return np.asarray([[1.0, 0.0] for _ in texts])

    pipeline = LivePipeline(lambda *_args: pytest.fail("Use native schemas"), embed,
                            "gpt-5.4-fixture", "fixture-embed", prompt_profile="structured",
                            generate_structured=generate)
    bundle = {"key": key, "duration": 20.0, "scenes": scenes, "trace": [],
              "database": str(tmp_path / "index.sqlite")}
    SceneCache(bundle["database"], pipeline.signature).save(scenes, np.array([[0., 1.], [1., 0.]]))
    return pipeline, bundle, requests, frames


def test_proposes_real_retrieved_cuts_without_qa_answer_stages(tmp_path):
    pipeline, bundle, requests, frames = fixture(tmp_path)
    updates = list(agent.propose_edit(pipeline, bundle, "Keep the box opening", 5))
    assert len(updates) == 5 and all(item is bundle for item in updates)
    assert [step["stage"] for step in bundle["trace"]] == [
        "Edit planner", "Tool agent", "Retrieval", "Evidence inspection", "Edit proposal",
    ]
    retrieval = bundle["trace"][2]["output"]
    assert [item["index"] for item in retrieval["selected"]] == [1]
    assert retrieval["calls"][0]["source_scene_ids"] == [0, 1]
    assert retrieval["calls"][0]["hits"][0]["semantic_cosine"] == 1.0
    assert len(requests) == 4
    assert bundle["edit_document"] == {
        "version": "1.0", "source_id": bundle["key"], "title": "", "clips": [
            {"id": "clip_1", "start": frames[1].seek_seconds, "end": frames[3].seek_seconds,
             "label": "Box opening", "mute": False},
        ],
    }
    assert bundle["edit_proposal"]["status"] == "complete"
    assert bundle["edit_caveats"] == [agent.BOUNDARY_NOTE]
    assert bundle["trace"][-1]["resolved_edit_document"] == bundle["edit_document"]
    assert "answer" not in bundle and "verification" not in bundle
    assert requests[-1]["images"] == bundle["selected"][0]["frames"]
    wire = strict_json_schema(requests[-1]["schema"])
    assert wire["$defs"]["_ClipProposal"]["properties"]["start_frame_id"]["enum"] == [
        frame.id for frame in frames
    ]


def test_no_matching_visual_moment_produces_explicit_no_edit_not_default_clip(tmp_path):
    pipeline, bundle, requests, _ = fixture(tmp_path, proposal=[{
        "status": "unresolved", "clips": [], "reason": "Requested action is not visible.",
    }])
    bundle["edit_document"] = {"stale": "must not survive failure"}
    with pytest.raises(ValueError, match="No supported edit"):
        list(agent.propose_edit(pipeline, bundle, "Keep the unseen action", 5))
    assert "edit_document" not in bundle
    assert bundle["edit_proposal"]["status"] == "failed" and len(requests) == 4
    assert bundle["trace"][-1]["output"]["status"] == "unresolved"


def test_unsupported_frame_reference_gets_one_native_repair_not_fallback(tmp_path):
    frames = inspection.frame_registry(scene(tmp_path), "a" * 64)
    bad = cut(frames)
    bad["clips"][0]["start_frame_id"] = "fabricated_frame"
    pipeline, bundle, requests, _ = fixture(tmp_path, proposal=[bad, bad])
    with pytest.raises(ValueError, match="Edit proposal failed validation after two attempts"):
        list(agent.propose_edit(pipeline, bundle, "Keep box opening", 5))
    assert len(requests) == 5 and "edit_document" not in bundle
    assert len(bundle["trace"][-1]["attempts"]) == 2
    assert "actual active frames" in requests[-1]["prompt"]


@pytest.mark.parametrize("fault", ["overlap", "cross_scene", "unknown_need", "stale_observation",
                                  "evidence_outside", "evidence_at_end", "over_maximum", "empty"])
def test_cut_validator_rejects_invalid_evidence_and_bounds(tmp_path, fault):
    frames = inspection.frame_registry(scene(tmp_path), "a" * 64)
    other = inspection.frame_registry(scene(tmp_path, 0), "a" * 64)
    value, maximum = cut(frames), 5.0
    clip = value["clips"][0]
    if fault == "overlap":
        value["clips"].append(deepcopy(clip))
    elif fault == "cross_scene":
        clip["start_frame_id"] = other[0].id
    elif fault == "unknown_need":
        clip["need_ids"] = ["invented"]
    elif fault == "stale_observation":
        clip["observation_ids"] = ["obs_s1_superseded"]
    elif fault == "evidence_outside":
        clip["frame_ids"] = [frames[0].id]
    elif fault == "evidence_at_end":
        clip["frame_ids"] = [frames[3].id]
    elif fault == "over_maximum":
        maximum = 1.0
    else:
        value["clips"] = []
    with pytest.raises(ValueError):
        agent._validate_cuts(agent._CutProposal.model_validate(value), frames + other, [],
                             ["box_use"], maximum)


def test_broad_observation_repair_can_keep_cut_and_use_interior_frames(tmp_path):
    frames = inspection.frame_registry(scene(tmp_path), "a" * 64)
    observation = ObservationRecord(
        id="obs_broad", scene=1, kind="action", subject="hand", predicate="opens", object="box",
        frame_ids=[frames[2].id, frames[3].id], need_ids=["box_use"], uncertainty="",
    )
    value = cut(frames)
    value["clips"][0]["observation_ids"] = [observation.id]
    with pytest.raises(ValueError) as error:
        agent._validate_cuts(agent._CutProposal.model_validate(value), frames, [observation],
                             ["box_use"], 5)
    message = str(error.value)
    assert "Clip 1 [12.300,15.900)s" in message
    assert f"obs_broad:{frames[3].id}@15.900s" in message
    assert "Total 3.600s / 5s maximum" in message
    assert "Omit broad optional observations" in message
    value["clips"][0]["observation_ids"] = []
    validated = agent._validate_cuts(agent._CutProposal.model_validate(value), frames,
                                     [observation], ["box_use"], 5)
    assert validated.clips[0].end_frame_id == frames[3].id


def test_repair_reports_reference_and_duration_violations_together(tmp_path):
    frames = inspection.frame_registry(scene(tmp_path), "a" * 64)
    value = cut(frames)
    value["clips"][0]["frame_ids"] = [frames[3].id]
    with pytest.raises(ValueError) as error:
        agent._validate_cuts(agent._CutProposal.model_validate(value), frames, [], ["box_use"], 1)
    assert f"frame_ids:{frames[3].id}" in str(error.value)
    assert "Duration exceeds maximum by 2.600s" in str(error.value)


def test_refined_active_frames_are_used_for_final_boundary_proposal(tmp_path, monkeypatch):
    refined_scene = scene(tmp_path, prefix="refined")
    refined_scene["detail_span"] = [12.0, 16.0]
    refined_frames = inspection.frame_registry(refined_scene, "a" * 64)
    pipeline, bundle, requests, original = fixture(tmp_path, proposal=[cut(refined_frames)],
                                                   refine=True)
    bundle.update(video_path=str(tmp_path / "source.mp4"), frame_directory=str(tmp_path))
    monkeypatch.setattr(agent, "sample_scene_frames", lambda _video, _directory, value, count:
                        deepcopy(value))
    refined_calls = []

    def refine(*_args, **kwargs):
        refined_calls.append(kwargs)
        return deepcopy(refined_scene)

    monkeypatch.setattr(inspection, "refine_scene_frames", refine)
    list(agent.propose_edit(pipeline, bundle, "Keep box opening", 5))
    assert refined_calls == [{"count": 6, "retain_context": True}]
    assert requests[-1]["images"] == refined_scene["frames"]
    assert original[2].id not in requests[-1]["prompt"]
    assert refined_frames[2].id in requests[-1]["prompt"]
    refinement = bundle["trace"][3]["refinement"]
    assert refinement["same_frames_for_inspection_edit_proposal"]
    assert "same_frames_for_inspection_answer_review" not in refinement


@pytest.mark.parametrize("maximum", [0, 61, float("nan"), True])
def test_invalid_limit_fails_before_model_calls(tmp_path, maximum):
    pipeline, bundle, requests, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match="maximum"):
        list(agent.propose_edit(pipeline, bundle, "Keep box opening", maximum))
    assert requests == []
