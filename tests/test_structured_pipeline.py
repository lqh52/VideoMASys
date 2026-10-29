"""Offline contract fixtures, not model accuracy measurements."""

import json
from copy import deepcopy

import numpy as np
import pytest
from PIL import Image
from pydantic import create_model

from videomasys.domain.visual_evidence import validate_answer
from videomasys.zerogpu import structured_pipeline as runtime
from videomasys.zerogpu import structured_prompts, svp_prompts
from videomasys.zerogpu.core import LivePipeline, SceneCache
from videomasys.zerogpu.presentation import render_answer, render_evidence, render_flow
from videomasys.zerogpu.svp_retrieval import SVPPlan


def make_scene(tmp_path, *, prefix="initial"):
    paths = []
    for index in range(6):
        path = tmp_path / f"{prefix}_{index}.jpg"
        Image.new("RGB", (20, 20), (index * 25, 50, 20)).save(path)
        paths.append(str(path))
    return {"index": 0, "start": 0.0, "end": 10.0,
            "frames": paths, "frame_times": [float(i + 1) for i in range(6)],
            "caption": "Fallible synthetic cache hint"}


def plan():
    return SVPPlan.model_validate({
        "scope": "local_event", "objective": "Identify visible contact",
        "evidence_needs": [{"id": "contact", "description": "What does the hand contact?",
                            "temporal_role": "context", "depends_on": []}],
        "max_intervals": 1, "max_inspected_seconds": 10.0,
        "stop_condition": "Contact is visible or uncertain", "reasoning": "Inspect contact",
    })


def observation(frames, *, refine=False):
    return {
        "facts": [{"kind": "state", "subject": "hand", "predicate": "touches",
                   "object": "lever", "frame_ids": [frames[0].id], "need_ids": ["contact"],
                   "uncertainty": ""}],
        "unresolved": [], "refine_frame_id": frames[2].id if refine else None,
    }


def answer(frames, *, options=(), phase="initial", inferred=False, limitation=False):
    claim = {
        "id": "contact_claim", "kind": "limitation" if limitation else "observed",
        "statement": (
            "The contact cannot be established." if limitation else "The hand touches the lever."
        ),
        "frame_ids": [frames[0].id], "observation_ids": [] if limitation else [f"obs_s0_{phase}_0"],
        "depends_on": [], "need_ids": ["contact"], "rationale": "",
    }
    claims = [claim]
    if inferred:
        claims.append({**claim, "id": "purpose", "kind": "inferred",
                       "statement": "The lever may control the visible indicator.",
                       "observation_ids": [], "depends_on": ["contact_claim"],
                       "rationale": "The contact and visible response support this possibility."})
    return {
        "claims": claims, "answer_claim_ids": [claims[-1]["id"]],
        "selected_choice": options[0] if options and not limitation else None,
        "choice_change_reason": (
            "The cited contact shows touching rather than pulling." if options else ""
        ),
        "option_comparisons": [
            {"choice": option,
             "status": "supported" if i == 0 and not limitation else "unsupported",
             "claim_ids": [claims[-1]["id"]] if i == 0 and not limitation else []}
            for i, option in enumerate(options)
        ],
    }


def review(draft, *, reject=False):
    targets = [claim["id"] for claim in draft["claims"]]
    return {"claim_checks": [{"target_id": target,
                        "status": "revise" if reject and index == 0 else "pass",
                        "reason": "Fixture visual audit"} for index, target in enumerate(targets)],
            "option_assessments": [
                {"choice": option["choice"], "status": option["status"],
                 "reason": "Fixture option support assessment"}
                for option in draft["option_comparisons"]
            ],
            "answer_judgment": {"status": "pass", "selected_choice": draft["selected_choice"],
                                "reason": "Fixture best-answer judgment"}}


def pipeline(outputs, requests, *, candidate="Pull the lever"):
    pending = iter(outputs)

    def generate(prompt, images, tokens, schema):
        requests.append({"prompt": prompt, "images": list(images), "schema": schema})
        if set(schema.get("properties", {})) == {"selected_choice"}:
            return json.dumps({"selected_choice": candidate})
        return json.dumps(next(pending))

    return LivePipeline(lambda *_args: candidate, lambda texts: np.array([[1., 0.] for _ in texts]),
                        "fixture", "fixture", prompt_profile="structured",
                        generate_structured=generate)


def bundle(scene):
    return {"key": "fixture_video", "selected": [scene], "scenes": [deepcopy(scene)],
            "trace": [], "options": [], "question": "What does the hand contact?"}


RETRIEVAL = {"selection_reasons": [{"scene": 0, "reasons": [{"source": "contact_search",
               "evidence_need_ids": ["contact"]}]}]}


def test_structured_repair_preserves_complete_long_json_without_answer_specific_callback():
    envelope = create_model("LongProgramFixture", steps=(list[str], ...), inspect=(str, ...))
    previous = {"steps": ["fixture-step " * 300], "inspect": "repair"}
    corrected = {**previous, "inspect": "final_sources"}
    requests = []
    app = pipeline([previous, corrected], requests)

    def validate(value):
        if value.inspect != "final_sources":
            raise ValueError("Check the final inspect sources")

    trace = []
    app.structured(trace, "Tool agent", "Compile this fixture", envelope, validate=validate)
    compact = json.dumps(previous, separators=(",", ":"))
    assert len(compact) > 3000
    assert compact in requests[1]["prompt"]
    assert json.loads(trace[0]["attempts"][0]["raw_response"]) == previous
    assert trace[0]["output"] == corrected


def test_whole_video_conceptual_dependency_is_repaired_before_compilation(tmp_path):
    invalid = plan().model_dump()
    invalid["scope"] = "whole_video"
    invalid["evidence_needs"].append({
        "id": "activity", "description": "What activity uses the contacted objects?",
        "temporal_role": "context", "depends_on": ["contact"],
    })
    corrected = deepcopy(invalid)
    corrected["evidence_needs"][1]["depends_on"] = []
    requests = []
    app = pipeline([invalid, corrected], requests)
    state = bundle(make_scene(tmp_path))
    execution = app.qa(state, "What is the overall activity?", [])
    assert next(execution) is state  # Pause after the planner, before tool compilation.
    execution.close()
    assert len(requests) == 2
    assert "conceptual connection is not a temporal dependency" in requests[1]["prompt"]
    assert state["trace"][0]["output"] == corrected


@pytest.mark.parametrize("correct_scope", [True, False])
def test_whole_video_repair_can_preserve_real_event_dependency_by_changing_scope(
    tmp_path, correct_scope,
):
    invalid = plan().model_dump()
    invalid["scope"] = "whole_video"
    invalid["evidence_needs"][0]["temporal_role"] = "anchor"
    invalid["evidence_needs"].append({
        "id": "following_action", "description": "What happens just after the lever contact?",
        "temporal_role": "after", "depends_on": ["contact"],
    })
    corrected = deepcopy(invalid)
    if correct_scope:
        corrected["scope"] = "local_event"
    requests = []
    app = pipeline([invalid, corrected], requests)
    state = bundle(make_scene(tmp_path))
    execution = app.qa(state, "How does the lever contribute to the overall process?", [])
    try:
        if correct_scope:
            assert next(execution) is state
            assert state["trace"][0]["output"] == corrected
            assert corrected["evidence_needs"] == invalid["evidence_needs"]
        else:
            with pytest.raises(ValueError, match="Planner failed validation after two attempts"):
                next(execution)
            assert state["trace"][0]["status"] == "failed"
            assert all("Whole-video needs must be independent" in attempt["validation_error"]
                       for attempt in state["trace"][0]["attempts"])
    finally:
        execution.close()
    assert len(requests) == 2
    feedback = requests[1]["prompt"]
    assert "use scope='local_event' and preserve" in feedback
    assert "Do not erase a genuine before/after dependency" in feedback
    assert json.loads(state["trace"][0]["attempts"][0]["raw_response"]) == invalid


@pytest.mark.parametrize("options", [[], ["Touch the lever", "Pull the lever"]])
def test_actual_typed_stages_preserve_records_and_derive_rendered_output(tmp_path, options):
    scene = make_scene(tmp_path)
    frames = runtime.frame_registry(scene, "fixture_video")
    draft = answer(frames, options=options)
    requests = []
    app = pipeline([observation(frames), draft, review(draft)], requests)
    state = bundle(scene)
    state["options"] = options
    list(app._answer_retrieved(state, state["question"], options, plan(), RETRIEVAL))
    assert state["answer"]["answer"] == draft["claims"][0]["statement"]
    assert state["answer"]["supporting_scenes"] == [0]
    assert state["status"] == "verified"
    assert state["evidence_graph"]["schema_version"] == "visual-evidence-v6"
    assert state["final_answer_complete"] and state["review_complete"]
    assert state["evidence_graph"]["observations"][0]["id"] == "obs_s0_initial_0"
    assert all(request["images"] == scene["frames"] for request in requests)
    assert "contact" in requests[0]["prompt"]
    assert "Claims" not in state["answer"]["answer"]
    if options:
        choice = next(stage for stage in state["trace"] if stage["stage"] == "Choice selection")
        assert choice["output"]["selected_choice"] == options[1]
        assert choice["candidate_is_provisional"] is True
        assert state["answer"]["selected_choice"] == options[0]
        assert state["answer"]["selected_option"] == 0
    assert "hand touches lever" in render_evidence(state)
    assert "contact_claim" in render_flow(state)
    assert "Claim and option checks" in render_flow(state)


def test_pass_reviews_of_unsupported_alternatives_preserve_supported_selected_answer(tmp_path):
    scene = make_scene(tmp_path)
    frames = runtime.frame_registry(scene, "fixture_video")
    options = ["Touch the lever", "Pull the lever", "Wave at somebody"]
    draft = answer(frames, options=options)
    verdict = review(draft)
    requests = []
    app = pipeline([observation(frames), draft, verdict], requests)
    state = bundle(scene)
    state["options"] = options
    list(app._answer_retrieved(state, state["question"], options, plan(), RETRIEVAL))
    assert [item["status"] for item in state["answer"]["option_comparisons"]] == [
        "supported", "unsupported", "unsupported",
    ]
    assert state["answer"]["selected_choice"] == options[0]
    assert state["verification"]["direct_assessments"] == verdict
    assert all(check["status"] == "pass" for check in state["verification"]["checks"])
    assert state["verification"]["issues"] == []
    assert state["verification"]["accepted"] is True
    assert state["status"] == "verified"
    assert len(requests) == 4  # Inspection, choice, answer, review; no answer correction.
    assert len([stage for stage in state["trace"] if stage["stage"] == "Answer"]) == 1


def test_reference_and_option_normalization_is_traced_without_replacing_raw_output(tmp_path):
    scene = make_scene(tmp_path)
    frames = runtime.frame_registry(scene, "fixture_video")
    options = ["Touch", "Pull"]
    raw = answer(frames, options=options)
    raw["claims"][0]["frame_ids"] *= 2
    raw["option_comparisons"].reverse()
    app = pipeline([observation(frames), raw, review(raw)], [], candidate="Touch")
    state = bundle(scene)
    list(app._answer_retrieved(state, state["question"], options, plan(), RETRIEVAL))
    stage = next(item for item in state["trace"] if item["stage"] == "Answer")
    assert len(stage["attempts"]) == 1
    assert json.loads(stage["attempts"][0]["raw_response"]) == raw
    assert stage["structured_output"]["claims"][0]["frame_ids"] == [frames[0].id]
    assert [item["choice"] for item in stage["structured_output"]["option_comparisons"]] == options
    assert stage["attempts"][0]["canonicalization"]["changed_fields"] == [
        "claims", "option_comparisons",
    ]
    assert state["answer"]["selected_choice"] == "Touch"
    assert state["final_answer_complete"] and state["status"] == "verified"


def test_refinement_answer_and_review_share_exact_frame_files_and_supersede_old_refs(
    tmp_path, monkeypatch,
):
    scene = make_scene(tmp_path)
    initial = runtime.frame_registry(scene, "fixture_video")
    refined_scene = make_scene(tmp_path, prefix="refined")
    refined_scene["frame_times"] = [1.0, 2.5, 3.5, 4.5, 5.5, 6.0]
    refined_scene["detail_span"] = [1.0, 5.0]
    refined = runtime.frame_registry(refined_scene, "fixture_video")
    draft = answer(refined, phase="refined")
    requests = []
    app = pipeline([observation(initial, refine=True), observation(refined), draft, review(draft)],
                   requests)
    state = bundle(scene)
    state.update(video_path="fixture.mp4", frame_directory=str(tmp_path))
    monkeypatch.setattr(runtime, "sample_scene_frames", lambda *_a, **_k: deepcopy(scene))

    def refine(*_args, **kwargs):
        assert kwargs == {"count": 6, "retain_context": True}
        return deepcopy(refined_scene)

    monkeypatch.setattr(runtime, "refine_scene_frames", refine)
    list(app._answer_retrieved(state, state["question"], [], plan(), RETRIEVAL))
    assert requests[0]["images"] == scene["frames"]
    assert all(request["images"] == refined_scene["frames"] for request in requests[1:])
    assert state["evidence_graph"]["observations"][0]["id"] == "obs_s0_refined_0"
    inspection = state["trace"][0]
    assert inspection["refinement"]["superseded_observation_ids"] == ["obs_s0_initial_0"]
    assert inspection["refinement"]["sample_count"] == 6
    assert initial[0].id == refined[0].id  # Retained context remains valid.
    assert initial[2].id not in {frame.id for frame in refined}
    stale = deepcopy(draft)
    stale["claims"][0].update(frame_ids=[initial[2].id], observation_ids=[])
    with pytest.raises(ValueError, match="unknown IDs"):
        validate_answer(stale, refined, state["evidence_graph"]["observations"], ["contact"], [])


def test_incomplete_review_is_repaired_not_silently_accepted(tmp_path):
    scene = make_scene(tmp_path)
    frames = runtime.frame_registry(scene, "fixture_video")
    draft = answer(frames, options=["Touch", "Pull"])
    invalid = review(draft)
    invalid["claim_checks"].pop()
    app = pipeline([observation(frames), draft, invalid, review(draft)], [], candidate="Pull")
    state = bundle(scene)
    list(app._answer_retrieved(state, state["question"], ["Touch", "Pull"], plan(), RETRIEVAL))
    assert len(state["trace"][-1]["attempts"]) == 2
    assert "validation_error" in state["trace"][-1]["attempts"][0]
    assert state["verification"]["accepted"] is True


def test_observation_and_dependency_refs_resolve_once_for_answer_and_review(tmp_path):
    scene = make_scene(tmp_path)
    frames = runtime.frame_registry(scene, "fixture_video")
    draft = answer(frames, inferred=True)
    for claim in draft["claims"]:
        claim["frame_ids"] = []
    requests = []
    app = pipeline([observation(frames), draft, review(draft)], requests)
    state = bundle(scene)
    list(app._answer_retrieved(state, state["question"], [], plan(), RETRIEVAL))
    assert state["answer"]["supporting_scenes"] == [0]
    assert state["answer"]["claims"][1]["frame_ids"] == []  # No rewriting model output.
    resolved = state["evidence_graph"]["claim_provenance"]
    assert resolved["purpose"]["frame_ids"] == [frames[0].id]
    assert resolved["purpose"]["observation_ids"] == ["obs_s0_initial_0"]
    assert '"resolved_provenance"' in requests[-1]["prompt"]
    assert state["status"] == "verified"


def test_repeated_image_bytes_do_not_consume_multiple_inspection_slots(tmp_path):
    scene = make_scene(tmp_path)
    scene["frames"][1] = scene["frames"][0]
    frames = runtime.frame_registry(scene, "fixture_video")
    assert len(frames) == 5
    assert frames[0].seek_seconds == 1.0
    assert len({frame.sha256 for frame in frames}) == 5
    scene["frame_times"][:2] = [2.0, 1.0]
    with pytest.raises(ValueError, match="chronological"):
        runtime.frame_registry(scene, "fixture_video")


def test_reviewer_revise_gets_only_one_answer_correction_then_remains_unverified(tmp_path):
    scene = make_scene(tmp_path)
    frames = runtime.frame_registry(scene, "fixture_video")
    draft = answer(frames)
    requests = []
    app = pipeline([observation(frames), draft, review(draft, reject=True),
                    draft, review(draft, reject=True)], requests)
    state = bundle(scene)
    list(app._answer_retrieved(state, state["question"], [], plan(), RETRIEVAL))
    assert len(requests) == 5
    assert state["verification"]["accepted"] is True
    assert state["verification"]["explanation_accepted"] is False
    assert state["verification"]["revision_scope"] == "explanation"
    assert state["verification"]["needs_revision"] is False
    assert state["status"] == "unverified"
    answers = [stage for stage in state["trace"] if stage["stage"] == "Answer"]
    reviews = [stage for stage in state["trace"] if stage["stage"] == "Verifier"]
    assert [stage["revision"] for stage in answers] == [0, 1]
    assert [stage["revision"] for stage in reviews] == [0, 1]
    assert all(stage["structured_output"]["claim_checks"][0]["status"] == "revise"
               for stage in reviews)
    assert all(len(stage["attempts"]) == 1 for stage in answers + reviews)
    assert "same evidence" in requests[-2]["prompt"].lower()
    assert "final review" in requests[-1]["prompt"].lower()
    assert all(request["images"] == scene["frames"] for request in requests)


def test_honest_limitation_is_not_displayed_as_verified_answer_or_duplicated(tmp_path):
    scene = make_scene(tmp_path)
    frames = runtime.frame_registry(scene, "fixture_video")
    draft = answer(frames, limitation=True)
    app = pipeline([observation(frames), draft, review(draft)], [])
    state = bundle(scene)
    list(app._answer_retrieved(state, state["question"], [], plan(), RETRIEVAL))
    assert state["verification"]["accepted"] is True
    assert state["status"] == "unverified"
    assert state["resolution"] == "insufficient_evidence"
    html = render_answer(state)
    assert "Insufficient evidence" in html and "Model-checked" not in html
    assert html.count(draft["claims"][0]["statement"]) == 1


def test_full_qa_uses_hybrid_program_with_typed_inspection(tmp_path):
    scene = make_scene(tmp_path)
    frames = runtime.frame_registry(scene, "fixture_video")
    draft = answer(frames)
    program = {
        "version": "3.0", "compiler": "model", "rationale": "Search stated contact",
        "max_intervals": 1, "max_inspected_seconds": 10.0,
        "steps": [{"op": "locate", "id": "contact_search", "source": "video",
                   "query": "hand contact", "limit": 1, "evidence_need_ids": ["contact"]}],
        "inspect": {"sources": ["contact_search"], "evidence_need_ids": ["contact"]},
    }
    app = pipeline([plan().model_dump(), program, observation(frames), draft, review(draft)], [])
    state = bundle(scene)
    state["database"] = str(tmp_path / "index.sqlite3")
    state["answer"] = {"answer": "stale previous output"}
    state.update(retained_reviewed_answer=True, explanation_repair_failed=True,
                 reviewed_pair={"answer_trace_index": -1, "review_trace_index": -1})
    SceneCache(state["database"], app.signature).save(state["scenes"], np.asarray([[1., 0.]]))
    list(app.qa(state, state["question"], []))
    assert [stage["stage"] for stage in state["trace"]] == [
        "Planner", "Tool agent", "Retrieval", "Evidence inspection", "Answer", "Verifier",
    ]
    assert state["status"] == "verified"
    assert state["answer"]["answer"] != "stale previous output"
    assert not state["retained_reviewed_answer"] and not state["explanation_repair_failed"]
    assert state["reviewed_pair"]["answer_trace_index"] >= 0


def test_profile_reuses_hybrid_demonstrations_and_keeps_clean_default():
    clean = LivePipeline(None, None, "fixture", "fixture")
    hybrid = LivePipeline(None, None, "fixture", "fixture", prompt_profile="svp")
    structured = LivePipeline(None, None, "fixture", "fixture", prompt_profile="structured")
    assert clean.prompt_profile == "clean"
    assert len({clean.signature, hybrid.signature, structured.signature}) == 3
    assert structured_prompts.planner_prompt is not svp_prompts.planner_prompt
    assert structured_prompts.tool_prompt is not svp_prompts.tool_prompt
    assert structured_prompts.PROMPT_VERSION == "structured-evidence-v7"
    context = "Question: What happens after contact?\nCached scenes: fixture hints."
    for role, args in (("planner_prompt", (context,)),
                       ("tool_prompt", (context, plan().model_dump()))):
        current, current_ids = getattr(structured_prompts, role)(*args)
        original, original_ids = getattr(svp_prompts, role)(*args)
        assert current_ids == original_ids
        assert current.split("BEGIN_FICTIONAL_EXAMPLES", 1)[1] == original.split(
            "BEGIN_FICTIONAL_EXAMPLES", 1
        )[1]


def test_frame_ids_bind_video_scene_seek_and_bytes_and_reject_bad_registry(tmp_path):
    scene = make_scene(tmp_path)
    frames = runtime.frame_registry(scene, "video_a")
    assert runtime.frame_registry(scene, "video_a") == frames
    integer_times = {**scene, "frame_times": [int(t) for t in scene["frame_times"]]}
    assert runtime.frame_registry(integer_times, "video_a") == frames
    assert runtime.frame_registry(scene, "video_b")[0].id != frames[0].id
    changed = deepcopy(scene)
    changed["frame_times"][0] = 0.5
    assert runtime.frame_registry(changed, "video_a")[0].id != frames[0].id
    changed["frame_times"][0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        runtime.frame_registry(changed, "video_a")
    changed["frame_times"] = []
    with pytest.raises(ValueError, match="aligned"):
        runtime.frame_registry(changed, "video_a")


def test_unused_context_stays_visible_to_review_without_becoming_answer_support(tmp_path):
    scene = make_scene(tmp_path)
    frames = runtime.frame_registry(scene, "fixture_video")
    draft = answer(frames)
    context = {**deepcopy(draft["claims"][0]), "id": "extra_context",
               "statement": "The indicator remains off."}
    draft["claims"].append(context)
    requests = []
    app = pipeline([observation(frames), draft, review(draft)], requests)
    state = bundle(scene)
    list(app._answer_retrieved(state, state["question"], [], plan(), RETRIEVAL))
    assert state["answer"]["context_claim_ids"] == ["extra_context"]
    assert state["answer"]["answer_lineage_ids"] == ["contact_claim"]
    assert state["answer"]["claims"] == draft["claims"]
    assert "The indicator remains off." in requests[-1]["prompt"]
    assert "The indicator remains off." not in state["answer"]["answer"]
    assert state["final_answer_complete"]


@pytest.mark.parametrize("failed_stage", ["revision", "review"])
def test_optional_failure_preserves_valid_draft_but_cannot_pass_release_gate(
    tmp_path, failed_stage,
):
    scene = make_scene(tmp_path)
    frames = runtime.frame_registry(scene, "fixture_video")
    draft = answer(frames)
    if failed_stage == "revision":
        invalid = deepcopy(draft)
        invalid["claims"][0]["frame_ids"] = ["unknown_frame"]
        verdict = review(draft, reject=True)
        verdict["answer_judgment"].update(
            status="revise", reason="The main conclusion lacks evidence.",
        )
        outputs = [observation(frames), draft, verdict, invalid, invalid]
    else:
        invalid = {**review(draft), "claim_checks": []}
        outputs = [observation(frames), draft, invalid, invalid]
    state = bundle(scene)
    list(pipeline(outputs, [])._answer_retrieved(
        state, state["question"], [], plan(), RETRIEVAL,
    ))
    assert state["answer"]["answer"] == draft["claims"][0]["statement"]
    assert state["status"] == "unverified"
    assert state["retained_previous_draft"]
    assert not state["final_answer_complete"] and not state["review_complete"]
    assert not state["verification"]["accepted"]
    assert state["trace"][-1]["status"] == "failed"
    assert len(state["trace"][-1]["attempts"]) == 2
    if failed_stage == "revision":
        assert state["verification"]["checks"][0]["status"] == "revise"
        assert "repair_diagnostics" in state["trace"][-1]["attempts"][0]


def test_request_schemas_bind_current_frames_needs_options_and_review_targets(tmp_path):
    from videomasys.domain.visual_evidence import StructuredAnswer
    from videomasys.zerogpu.openai import strict_json_schema
    from videomasys.zerogpu.structured_schema import bounded_schema

    scene = make_scene(tmp_path)
    frames = runtime.frame_registry(scene, "fixture_video")
    options = ["Touch", "Pull"]
    draft = answer(frames, options=options)
    requests = []
    state = bundle(scene)
    list(pipeline([observation(frames), draft, review(draft)], requests, candidate="Touch")
         ._answer_retrieved(state, state["question"], options, plan(), RETRIEVAL))
    assert len(requests) == 4
    for request in requests:
        strict_json_schema(request["schema"])
    fact_fields = requests[0]["schema"]["$defs"]["ObservedFact"]["properties"]
    assert fact_fields["frame_ids"]["items"]["enum"] == [frame.id for frame in frames]
    assert fact_fields["need_ids"]["items"]["enum"] == ["contact"]
    choice = requests[1]["schema"]["properties"]["selected_choice"]
    assert choice["anyOf"][0]["enum"] == options
    claim_fields = requests[2]["schema"]["$defs"]["ObservedClaim"]["properties"]
    assert claim_fields["observation_ids"]["items"]["enum"] == ["obs_s0_initial_0"]
    target = requests[3]["schema"]["$defs"]["ReviewCheck"]["properties"]["target_id"]
    assert target["enum"] == ["contact_claim"]
    open_schema = bounded_schema(StructuredAnswer, observations=[], options=[])
    strict_json_schema(open_schema)
    assert open_schema["properties"]["selected_choice"] == {"type": "null"}
    assert open_schema["properties"]["option_comparisons"]["maxItems"] == 0
    assert "enum" not in StructuredAnswer.model_json_schema()["$defs"]["ObservedClaim"][
        "properties"]["frame_ids"]["items"]


def test_context_and_unchosen_option_notes_do_not_revise_the_answer(tmp_path):
    scene = make_scene(tmp_path)
    frames = runtime.frame_registry(scene, "fixture_video")
    options = ["Touch", "Pull"]
    draft = answer(frames, options=options)
    draft["claims"].append({**deepcopy(draft["claims"][0]), "id": "context",
                            "statement": "An unrelated indicator is green."})
    verdict = review(draft)
    verdict["claim_checks"][1].update(status="revise", reason="Indicator color is unclear.")
    verdict["option_assessments"][1].update(
        status="supported", reason="Some contact could support it.",
    )
    requests = []
    state = bundle(scene)
    list(pipeline([observation(frames), draft, verdict], requests, candidate="Touch")
         ._answer_retrieved(state, state["question"], options, plan(), RETRIEVAL))
    assert len(requests) == 4
    assert state["answer"]["selected_choice"] == "Touch"
    assert state["verification"]["issues"] == []
    assert len(state["verification"]["advisories"]) == 2
    assert state["verification"]["revision_scope"] == "none"
    assert not state["verification"]["needs_revision"]
    assert state["status"] == "verified" and state["final_answer_complete"]


def test_explanation_repair_locks_choice_in_schema_and_runtime(tmp_path):
    scene = make_scene(tmp_path)
    frames = runtime.frame_registry(scene, "fixture_video")
    options = ["Touch", "Pull"]
    draft = answer(frames, options=options)
    changed = answer(frames, options=list(reversed(options)))
    fixed = deepcopy(draft)
    fixed["choice_change_reason"] = ""
    requests = []
    state = bundle(scene)
    list(pipeline([observation(frames), draft, review(draft, reject=True),
                   changed, fixed, review(fixed)], requests, candidate="Touch")
         ._answer_retrieved(state, state["question"], options, plan(), RETRIEVAL))
    stages = [step for step in state["trace"] if step["stage"] == "Answer"]
    assert stages[1]["revision_scope"] == "explanation"
    assert "Explanation-only repair" in stages[1]["attempts"][0]["validation_error"]
    assert "Explanation-only repair" in stages[1]["attempts"][1]["prompt"]
    assert stages[1]["attempts"][0]["response_schema"]["properties"]["selected_choice"] == {
        "type": "string", "enum": ["Touch"],
    }
    assert json.loads(stages[1]["attempts"][0]["raw_response"]) == changed
    assert state["answer"]["selected_choice"] == "Touch"
    assert state["status"] == "verified" and state["final_answer_complete"]


@pytest.mark.parametrize("failure", ["answer", "review"])
def test_failed_optional_explanation_edit_restores_exact_reviewed_pair(tmp_path, failure):
    scene = make_scene(tmp_path)
    frames = runtime.frame_registry(scene, "fixture_video")
    options = ["Touch", "Pull"]
    draft = answer(frames, options=options)
    first_review = review(draft, reject=True)
    invalid = deepcopy(draft) if failure == "answer" else review(draft)
    if failure == "answer":
        invalid["claims"][0]["frame_ids"] = ["unknown"]
        outputs = [observation(frames), draft, first_review, invalid, invalid]
    else:
        invalid["claim_checks"] = []
        edited = deepcopy(draft)
        edited["claims"][0]["statement"] = "A changed but unreviewed explanation."
        outputs = [observation(frames), draft, first_review, edited, invalid, invalid]
    state = bundle(scene)
    list(pipeline(outputs, [], candidate="Touch")._answer_retrieved(
        state, state["question"], options, plan(), RETRIEVAL,
    ))
    assert state["answer"]["claims"] == draft["claims"]
    assert state["answer"]["selected_choice"] == "Touch"
    assert state["verification"]["direct_assessments"] == first_review
    assert state["verification"]["accepted"] and not state["verification"]["explanation_accepted"]
    assert state["retained_reviewed_answer"] and state["explanation_repair_failed"]
    assert not state["retained_previous_draft"]
    assert state["final_answer_complete"] and state["review_complete"]
    assert state["status"] == "unverified"
    assert state["trace"][-1]["status"] == "failed"
    assert state["evidence_graph"]["reasoning"]["claims"] == draft["claims"]
    pair = state["reviewed_pair"]
    assert state["trace"][pair["answer_trace_index"]]["output"] == state["answer"]
    assert state["trace"][pair["review_trace_index"]]["structured_output"] == first_review
    assert pair["review_trace_index"] < len(state["trace"]) - 1


def test_real_answer_issue_can_change_choice_with_explicit_reason(tmp_path):
    scene = make_scene(tmp_path)
    frames = runtime.frame_registry(scene, "fixture_video")
    options = ["Touch", "Pull"]
    draft = answer(frames, options=options)
    verdict = review(draft)
    verdict["answer_judgment"].update(status="revise", selected_choice="Pull",
                                      reason="The evidence shows pulling, not just contact.")
    verdict["option_assessments"][1]["status"] = "supported"
    corrected = answer(frames, options=list(reversed(options)))
    corrected["claims"][0]["statement"] = "The hand pulls the lever."
    corrected["choice_change_reason"] = "The motion in contact_claim supports pulling."
    state = bundle(scene)
    list(pipeline([observation(frames), draft, verdict, corrected, review(corrected)], [],
                  candidate="Touch")._answer_retrieved(
        state, state["question"], options, plan(), RETRIEVAL,
    ))
    stage = [step for step in state["trace"] if step["stage"] == "Answer"][1]
    assert stage["revision_scope"] == "answer"
    assert stage["decision_transition"]["previous_choice"] == "Touch"
    assert stage["decision_transition"]["selected_choice"] == "Pull"
    assert stage["decision_transition"]["changed"]
    assert stage["decision_transition"]["reason"] == corrected["choice_change_reason"]
    assert stage["decision_transition"]["previous_conclusion"] == draft["claims"][0]["statement"]
    assert state["answer"]["decision"]["conclusion"] == "The hand pulls the lever."
    assert state["answer"]["decision"]["selected_choice"] == "Pull"
    assert state["status"] == "verified"


def test_missing_choice_change_reason_is_repaired_without_inventing_a_decision(tmp_path):
    scene = make_scene(tmp_path)
    frames = runtime.frame_registry(scene, "fixture_video")
    options = ["Touch", "Pull"]
    corrected = answer(frames, options=options)
    invalid = {**deepcopy(corrected), "choice_change_reason": ""}
    state = bundle(scene)
    app = pipeline([observation(frames), invalid, corrected, review(corrected)], [],
                   candidate="Pull")
    list(app._answer_retrieved(state, state["question"], options, plan(), RETRIEVAL))
    stage = next(step for step in state["trace"] if step["stage"] == "Answer")
    assert "choice_change_reason" in stage["attempts"][1]["prompt"]
    diagnostic = stage["attempts"][0]["repair_diagnostics"]
    assert "Changing selected_choice" in diagnostic["validation_error"]
    assert json.loads(stage["attempts"][0]["raw_response"]) == invalid
    assert stage["decision_transition"]["changed"]
    assert state["answer"]["selected_choice"] == "Touch"


def test_exploratory_plan_need_stays_in_trace_without_forcing_answer_claims(tmp_path):
    scene = make_scene(tmp_path)
    frames = runtime.frame_registry(scene, "fixture_video")
    expanded = plan().model_dump()
    expanded["evidence_needs"].append({
        "id": "background", "description": "What is behind the hand?",
        "temporal_role": "context", "depends_on": [],
    })
    expanded = SVPPlan.model_validate(expanded)
    inspection = observation(frames)
    inspection["unresolved"] = [{"need_id": "background", "reason": "Not clearly visible."}]
    retrieval = deepcopy(RETRIEVAL)
    retrieval["selection_reasons"][0]["reasons"][0]["evidence_need_ids"].append("background")
    draft = answer(frames)
    state = bundle(scene)
    list(pipeline([inspection, draft, review(draft)], [])._answer_retrieved(
        state, state["question"], [], expanded, retrieval,
    ))
    assert state["answer"]["plan_coverage"]["unaddressed_need_ids"] == ["background"]
    assert state["answer"]["claims"] == draft["claims"]
    assert state["evidence_graph"]["unresolved"][0]["need_id"] == "background"
    assert state["status"] == "verified"
