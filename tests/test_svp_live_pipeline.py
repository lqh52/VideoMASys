"""Offline SVP integration contracts; synthetic responses are not accuracy evidence."""

import copy
import json

import numpy as np
import pytest

from videomasys.zerogpu.core import LivePipeline, SceneCache
from videomasys.zerogpu.presentation import render_flow


def plan():
    return {
        "scope": "local_event", "objective": "Identify the visible interaction",
        "evidence_needs": [{"id": "interaction", "description": "Observed object contact",
                            "temporal_role": "context", "depends_on": []}],
        "max_intervals": 1, "max_inspected_seconds": 10,
        "stop_condition": "The visible contact is identified",
        "reasoning": "Follow the subject to its interaction",
    }


def program(seconds=10):
    return {
        "version": "3.0", "compiler": "model", "rationale": "Narrow the subject results",
        "max_intervals": 1, "max_inspected_seconds": seconds,
        "steps": [
            {"op": "locate", "id": "subject", "source": "video", "query": "subject",
             "limit": 2, "evidence_need_ids": []},
            {"op": "locate", "id": "contact", "source": "subject", "query": "contact",
             "limit": 1, "evidence_need_ids": ["interaction"]},
        ],
        "inspect": {"sources": ["contact"], "evidence_need_ids": ["interaction"]},
    }


def fixture_pipeline(tmp_path, outputs):
    pending = iter(outputs)
    requests = []

    def generate(prompt, images, tokens):
        requests.append({"prompt": prompt, "images": list(images)})
        return json.dumps(next(pending))

    pipeline = LivePipeline(
        generate, lambda texts: np.asarray([[1., 0.] for _ in texts]),
        "fixture", "fixture", prompt_profile="svp",
    )
    scenes = [{"index": 7, "start": 0., "end": 10., "caption": "Subject contact",
               "frames": ["synthetic-frame.jpg"], "frame_times": [5.]}]
    bundle = {"scenes": scenes, "database": str(tmp_path / "index.sqlite"), "trace": []}
    SceneCache(bundle["database"], pipeline.signature).save(scenes, np.asarray([[1., 0.]]))
    return pipeline, bundle, requests


def answer_outputs():
    return [
        {"scene": 7, "observation": "At 5s, the subject touches an object.",
         "relevant": True, "uncertainty": "", "refine_at": None},
        {"answer": "The subject touches the object.", "supporting_scenes": [7],
         "limitations": [], "inference": "Visible contact."},
        {"accepted": True, "needs_revision": False, "issues": []},
    ]


@pytest.mark.parametrize("repair", [False, True])
def test_real_scoped_executor_reaches_answer_and_preserves_feedback(tmp_path, repair):
    programs = [program(5), program()] if repair else [program()]
    pipeline, bundle, requests = fixture_pipeline(tmp_path, [plan(), *programs, *answer_outputs()])
    list(pipeline.qa(bundle, "What is the subject doing?", []))
    assert bundle["answer"]["supporting_scenes"] == [7]
    assert [scene["index"] for scene in bundle["selected"]] == [7]
    stages = bundle["trace"]
    compiler = [entry for entry in stages if entry["stage"].startswith("Tool agent")]
    assert len(compiler) == 1 + int(repair)
    assert compiler[-1]["output"]["steps"][1]["source"] == "subject"
    assert compiler[-1]["output"]["compiler"] == ("model_repaired" if repair else "model")
    inspection = next(entry for entry in stages if entry["stage"] == "Evidence inspection")
    assert inspection["calls"][0]["evidence_sought"] == ["Observed object contact"]
    assert len(requests) == 5 + int(repair)
    if repair:
        assert "EXECUTION FEEDBACK" in compiler[-1]["attempts"][0]["prompt"]
        assert stages[2]["stage"] == "Retrieval" and stages[2]["status"] == "failed"
        assert not stages[2]["output"]["selected"]


def test_execution_retry_stops_without_answer_or_fallback(tmp_path):
    pipeline, bundle, requests = fixture_pipeline(tmp_path, [plan(), program(5), program(5)])
    with pytest.raises(ValueError, match="one execution-feedback repair"):
        list(pipeline.qa(bundle, "What is the subject doing?", []))
    assert "answer" not in bundle
    assert len(requests) == 3
    assert all(not request["images"] for request in requests)
    assert bundle["trace"][-1]["status"] == "failed"


def test_svp_is_opt_in_and_has_isolated_cache():
    clean = LivePipeline(None, None, "fixture", "fixture")
    candidate = LivePipeline(None, None, "fixture", "fixture", prompt_profile="svp")
    assert clean.prompt_profile == "clean"
    assert clean.signature != candidate.signature


def _temporal_retrieval_fixture(tmp_path, steps, vectors, query_vectors, *, intervals=3):
    from videomasys.zerogpu.svp_retrieval import SVPPlan

    payload = {
        "version": "3.0", "compiler": "model", "rationale": "Find temporal evidence",
        "max_intervals": intervals, "max_inspected_seconds": float(intervals * 20),
        "steps": steps,
        "inspect": {"sources": [step["id"] for step in steps],
                    "evidence_need_ids": ["anchor_need", "context_need"]},
    }
    pipeline, bundle, requests = fixture_pipeline(tmp_path, [payload, payload])
    pipeline.embed = lambda texts: np.asarray([query_vectors[text] for text in texts])
    bundle["scenes"] = [
        {"index": index, "start": float(index * 20), "end": float((index + 1) * 20),
         "caption": "Synthetic indexed interaction", "frames": [], "frame_times": []}
        for index in range(len(vectors))
    ]
    SceneCache(bundle["database"], pipeline.signature).save(bundle["scenes"], np.asarray(vectors))
    value = plan()
    value.update(max_intervals=intervals, max_inspected_seconds=float(intervals * 20))
    value["evidence_needs"] = [
        {"id": "anchor_need", "description": "Identify the anchor interaction",
         "temporal_role": "anchor", "depends_on": []},
        {"id": "context_need", "description": "Inspect the related context",
         "temporal_role": "context", "depends_on": ["anchor_need"]},
    ]
    return pipeline, bundle, requests, SVPPlan.model_validate(value)


def test_redundant_broad_source_does_not_trigger_execution_repair(tmp_path):
    steps = [
        {"op": "locate", "id": "anchor", "source": "video", "query": "anchor query",
         "limit": 1, "evidence_need_ids": ["anchor_need"]},
        {"op": "locate", "id": "broad", "source": "video", "query": "target query",
         "limit": 1, "evidence_need_ids": ["context_need"]},
        {"op": "relate", "id": "after", "source": "anchor", "relation": "after",
         "horizon_seconds": 45.0, "limit": 3, "evidence_need_ids": ["context_need"]},
    ]
    pipeline, bundle, requests, value = _temporal_retrieval_fixture(
        tmp_path, steps, [[1., 0.], [.8, .6], [.6, .8], [.2, .98], [0., 1.]],
        {"anchor query": [1., 0.], "target query": [0., 1.]},
    )
    list(pipeline._retrieve_svp(bundle, "Question: What is the related context?", value))
    retrieval = bundle["trace"][-1]
    assert retrieval["stage"] == "Retrieval" and retrieval["status"] == "complete"
    assert [scene["index"] for scene in bundle["selected"]] == [0, 2, 3]
    assert retrieval["output"]["uncovered_sources"] == ["broad"]
    assert not retrieval["output"]["uncovered_evidence_needs"]
    assert not retrieval["output"]["uncovered_temporal_outputs"]
    assert len(requests) == 1


def test_uncovered_temporal_branch_still_triggers_bounded_repair(tmp_path):
    steps = [
        {"op": "locate", "id": "anchor", "source": "video", "query": "anchor query",
         "limit": 1, "evidence_need_ids": ["anchor_need"]},
        *[{"op": "relate", "id": direction, "source": "anchor", "relation": direction,
           "horizon_seconds": 1.0, "limit": 1, "evidence_need_ids": ["context_need"]}
          for direction in ("before", "after")],
    ]
    pipeline, bundle, requests, value = _temporal_retrieval_fixture(
        tmp_path, steps, [[0., 1.], [1., 0.], [0., 1.]], {"anchor query": [1., 0.]},
        intervals=2,
    )
    with pytest.raises(ValueError, match="one execution-feedback repair"):
        list(pipeline._retrieve_svp(bundle, "Question: What is the related context?", value))
    retrieval = bundle["trace"][-1]
    assert not retrieval["output"]["uncovered_evidence_needs"]
    assert retrieval["output"]["uncovered_temporal_outputs"] == ["after"]
    assert retrieval["status"] == "failed"
    assert len(requests) == 2


def test_v3_presentation_keeps_execution_order_without_changing_input():
    output = program()
    output["steps"].insert(1, {
        "op": "relate", "id": "after_subject", "source": "subject", "relation": "after",
        "horizon_seconds": 20, "limit": 2, "evidence_need_ids": [],
    })
    output["steps"][2]["source"] = "after_subject"
    original = copy.deepcopy(output)
    rendered = render_flow({"trace": [{"stage": "Tool agent", "output": output}]})
    assert rendered.index("subject = locate") < rendered.index("after_subject = relate")
    assert rendered.index("after_subject = relate") < rendered.index("contact = locate")
    assert "op=" not in rendered
    assert output == original


def test_inspection_targets_use_only_provenance_need_ids():
    from videomasys.zerogpu.svp_retrieval import SVPPlan

    targets = LivePipeline._inspection_targets(SVPPlan.model_validate(plan()), {
        "selection_reasons": [{"scene": 7, "reasons": [
            {"evidence_need_ids": ["interaction", "invented"], "caption": "Untrusted claim"},
            {"evidence_need_ids": ["interaction"]},
        ]}],
    })
    assert targets == {7: ["Observed object contact"]}


def test_inspection_targets_can_follow_untagged_terminal_lineage():
    from videomasys.zerogpu.svp_retrieval import SVPPlan

    targets = LivePipeline._inspection_targets(SVPPlan.model_validate(plan()), {
        "selection_reasons": [{"scene": 7, "reasons": [{
            "evidence_need_ids": [], "lineage_evidence_need_ids": ["interaction"],
        }]}],
    })
    assert targets == {7: ["Observed object contact"]}


def test_overview_presentation_names_the_real_score_not_cosine():
    rendered = render_flow({"trace": [{"stage": "Retrieval", "output": {
        "calls": [{"call": {"op": "overview", "source": "video", "id": "context"},
                   "score_type": "representative_diversity", "hits": [{
                       "scene": 7, "representative_score": 0.6,
                       "selection_method": "semantic_medoid",
                   }]}],
    }}]})
    assert "Representative score: 0.6" in rendered
    assert "Method: semantic medoid" in rendered
    assert "Cosine similarity: 0.6" not in rendered
