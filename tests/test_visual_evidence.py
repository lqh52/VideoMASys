"""Offline structural tests; synthetic assertions are not model accuracy evidence."""

from copy import deepcopy

import pytest

from videomasys.domain.visual_evidence import (
    Claim,
    DirectStructuredReview,
    FrameRef,
    LimitationClaim,
    ObservationRecord,
    ObservedClaim,
    StructuredAnswer,
    StructuredReview,
    WindowObservation,
    answer_diagnostics,
    answer_need_coverage,
    claim_partition,
    claim_provenance,
    validate_answer,
    validate_choice_change,
    validate_direct_review,
    validate_review,
    validate_window,
)


def frames():
    return [FrameRef(
        id=f"f{index}", scene=1 if index < 3 else 2, seek_seconds=float(index),
        path=f"/fixture/f{index}.jpg", sha256=str(index) * 64,
    ) for index in range(1, 4)]


def fact(**changes):
    return {
        "kind": "action", "subject": "Worker", "predicate": "holds", "object": "item",
        "frame_ids": ["f1"], "need_ids": ["contact"], "uncertainty": "", **changes,
    }


def observation(**changes):
    return ObservationRecord(**fact(), id="obs_contact", scene=1, **changes)


def window(**changes):
    return {
        "facts": [fact()], "unresolved": [], "refine_frame_id": None, **changes,
    }


def claim(**changes):
    return {
        "id": "contact_claim", "kind": "observed", "statement": "The worker holds an item.",
        "frame_ids": ["f1"], "observation_ids": ["obs_contact"], "depends_on": [],
        "need_ids": ["contact"], "rationale": "", **changes,
    }


def answer(**changes):
    return {
        "claims": [
            claim(),
            claim(id="goal_claim", kind="inferred", statement="The worker may move the item.",
                  frame_ids=["f2"], observation_ids=[], depends_on=["contact_claim"],
                  need_ids=["goal"], rationale="Holding the item supports a tentative move."),
        ],
        "answer_claim_ids": ["goal_claim"], "selected_choice": "Move the item",
        "choice_change_reason": "",
        "option_comparisons": [
            {"choice": "Move the item", "status": "supported", "claim_ids": ["goal_claim"]},
            {"choice": "Signal somebody", "status": "unsupported", "claim_ids": []},
        ],
        **changes,
    }


def checked_answer(value):
    return validate_answer(value, frames(), [observation()], ["contact", "goal"],
                           ["Move the item", "Signal somebody"])


def test_window_binds_facts_to_requested_frames_and_needs_without_mutating_input():
    data = window()
    previous = deepcopy(data)
    value = validate_window(data, frames()[:2], ["contact"])
    assert isinstance(value, WindowObservation)
    assert value.facts[0].frame_ids == ["f1"]
    assert data == previous


def test_all_unresolved_window_is_valid_and_partial_facts_may_leave_need_unresolved():
    value = validate_window(window(facts=[], unresolved=[
        {"need_id": "contact", "reason": "Contact is not resolved by these samples."}
    ]), frames()[:2], ["contact"])
    assert not value.facts
    assert validate_window(window(unresolved=[
        {"need_id": "contact", "reason": "The exact contact point remains unclear."}
    ]), frames()[:2], ["contact"]).unresolved


@pytest.mark.parametrize("change,match", [
    ({"facts": []}, "Every requested need"),
    ({"facts": [fact(frame_ids=["f3"])]}, "unknown IDs"),
    ({"facts": [fact(frame_ids=["f1", "f1"])]}, "unique"),
    ({"facts": [fact(need_ids=["other"])]}, "unknown IDs"),
    ({"facts": [fact(need_ids=["contact", "contact"])]}, "unique"),
    ({"refine_frame_id": "not_shown"}, "unknown IDs"),
    ({"unresolved": [{"need_id": "other", "reason": "Unknown need."}]}, "unknown IDs"),
    ({"unresolved": [{"need_id": "contact", "reason": "Unclear."}] * 2}, "unique"),
])
def test_window_rejects_missing_unknown_or_duplicate_references(change, match):
    with pytest.raises(ValueError, match=match):
        validate_window(window(**change), frames()[:2], ["contact"])


@pytest.mark.parametrize("ids", [["f1"], ["f2", "f1"], ["f1", "f1"]])
def test_change_needs_distinct_chronological_frames(ids):
    with pytest.raises(ValueError):
        validate_window(window(facts=[fact(kind="change", frame_ids=ids)]),
                        frames()[:2], ["contact"])


def test_change_uses_actual_seek_times_not_frame_id_order():
    registry = frames()[:2]
    registry[1].seek_seconds = registry[0].seek_seconds
    with pytest.raises(ValueError, match="distinct chronological"):
        validate_window(window(facts=[fact(kind="change", frame_ids=["f1", "f2"])]),
                        registry, ["contact"])
    registry[1].seek_seconds = 2.0
    assert validate_window(window(facts=[fact(kind="change", frame_ids=["f1", "f2"])]),
                           registry, ["contact"])


def test_different_seek_times_with_identical_pixels_cannot_support_a_change():
    registry = frames()[:2]
    registry[1].sha256 = registry[0].sha256
    with pytest.raises(ValueError, match="distinct frame image hashes"):
        validate_window(window(facts=[fact(kind="change", frame_ids=["f1", "f2"])]),
                        registry, ["contact"])
    # Repeated pixels can still ground a state or action interpretation.
    assert validate_window(window(facts=[fact(frame_ids=["f1", "f2"])]),
                           registry, ["contact"])


def test_refinement_requests_reference_actual_frames_and_cannot_recur():
    value = window(refine_frame_id="f2")
    assert validate_window(value, frames()[:2], ["contact"]).refine_frame_id == "f2"
    with pytest.raises(ValueError, match="refinement|Refinement"):
        validate_window(value, frames()[:2], ["contact"], allow_refine=False)


@pytest.mark.parametrize("registry", [[], frames(), [frames()[0], frames()[0]]])
def test_window_rejects_empty_mixed_scene_or_duplicate_frame_registry(registry):
    with pytest.raises(ValueError):
        validate_window(window(), registry, ["contact"])


def test_valid_purpose_inference_keeps_dependency_provenance_and_original_input():
    data = answer()
    previous = deepcopy(data)
    value = checked_answer(data)
    provenance = claim_provenance(value, [observation()])
    assert value.selected_choice == "Move the item"
    assert provenance["goal_claim"] == {
        "frame_ids": ["f2", "f1"], "observation_ids": ["obs_contact"],
    }
    assert data == previous


def test_answer_reference_sets_preserve_provenance_claims_and_raw_response():
    data = answer(answer_claim_ids=["goal_claim", "goal_claim"])
    data["claims"][0].update(observation_ids=["obs_contact", "obs_contact"],
                             need_ids=["contact", "contact"])
    data["claims"][1].update(frame_ids=["f2", "f2"],
                             depends_on=["contact_claim", "contact_claim"])
    data["option_comparisons"][0]["claim_ids"] = ["goal_claim", "goal_claim"]
    original = deepcopy(data)
    value = checked_answer(data)
    expected = checked_answer(answer())
    assert value == expected
    assert claim_provenance(value, [observation()]) == claim_provenance(expected, [observation()])
    assert answer_diagnostics(data, frames(), [observation()], ["contact", "goal"],
                              ["Move the item", "Signal somebody"])["issues"] == []
    assert data == original


@pytest.mark.parametrize("mutation", ["oversized", "unknown", "self_dependency"])
def test_reference_normalization_cannot_bypass_bounds_or_graph_validation(mutation):
    data = answer()
    if mutation == "oversized":
        data["claims"][0]["frame_ids"] = ["f1"] * 7
    elif mutation == "unknown":
        data["claims"][0]["frame_ids"] = ["unknown", "unknown"]
    else:
        data["claims"][1]["depends_on"] = ["goal_claim", "goal_claim"]
    with pytest.raises(ValueError):
        checked_answer(data)


@pytest.mark.parametrize("direct_frames", [[], ["f1"]])
def test_observation_links_supply_pixels_without_redundant_direct_frame_copies(direct_frames):
    data = answer()
    data["claims"][0]["frame_ids"] = direct_frames
    data["claims"][1]["frame_ids"] = []
    record = ObservationRecord(**{**fact(frame_ids=["f1", "f2"]),
                                  "id": "obs_contact", "scene": 1})
    original = deepcopy(data)
    value = validate_answer(data, frames(), [record], ["contact", "goal"],
                            ["Move the item", "Signal somebody"])
    provenance = claim_provenance(value, [record])
    expected = {"frame_ids": ["f1", "f2"], "observation_ids": ["obs_contact"]}
    assert provenance["contact_claim"] == provenance["goal_claim"] == expected
    assert value.claims[0].frame_ids == direct_frames
    assert value.claims[1].frame_ids == value.claims[1].observation_ids == []
    assert data == original


@pytest.mark.parametrize("kind", ["observed", "inferred", "limitation"])
def test_no_claim_kind_may_have_empty_resolved_pixel_provenance(kind):
    data = answer(claims=[claim(
        id="goal_claim", kind=kind, frame_ids=[], observation_ids=[],
        need_ids=["contact", "goal"], rationale="Visible context supports this." if
        kind == "inferred" else "",
    )])
    with pytest.raises(ValueError, match="nonempty resolved provenance"):
        checked_answer(data)


def test_observation_only_limitations_are_displayed_roots_and_cover_planned_needs():
    data = answer(answer_claim_ids=["contact_claim"])
    data["claims"][1].update(kind="limitation", statement="The goal remains unresolved.",
                             frame_ids=[], observation_ids=["obs_contact"],
                             depends_on=[], rationale="")
    data["option_comparisons"][0]["claim_ids"] = ["contact_claim"]
    value = checked_answer(data)
    assert value.answer_claim_ids == ["contact_claim"]
    assert claim_provenance(value, [observation()])["goal_claim"]["frame_ids"] == ["f1"]
    targets = [item.id for item in value.claims] + ["option_0", "option_1"]
    checks = [{"target_id": name, "status": "pass", "reason": "Checked."}
              for name in targets]
    assert validate_review({"checks": checks}, targets)
    with pytest.raises(ValueError, match="every expected target"):
        validate_review({"checks": [item for item in checks if
                                    item["target_id"] != "goal_claim"]}, targets)


def test_provenance_deduplicates_direct_observation_and_dependency_references():
    data = answer()
    data["claims"][1].update(frame_ids=["f1", "f2"], observation_ids=["obs_contact"])
    value = checked_answer(data)
    original = value.model_dump()
    assert claim_provenance(value, [observation()])["goal_claim"] == {
        "frame_ids": ["f1", "f2"], "observation_ids": ["obs_contact"],
    }
    assert value.model_dump() == original


def test_observation_only_claim_rejects_inactive_observation_and_stale_pixels():
    data = answer()
    data["claims"][0]["frame_ids"] = []
    data["claims"][0]["observation_ids"] = ["inactive_observation"]
    with pytest.raises(ValueError, match="unknown IDs"):
        checked_answer(data)
    data["claims"][0]["observation_ids"] = ["obs_contact"]
    with pytest.raises(ValueError, match="unknown IDs"):
        validate_answer(data, frames()[1:], [observation()], ["contact", "goal"],
                        ["Move the item", "Signal somebody"])


def test_direct_pixel_inference_can_correct_or_omit_an_observer_without_an_extra_call():
    data = answer(claims=[claim(
        id="goal_claim", kind="inferred", frame_ids=["f2"], observation_ids=[],
        need_ids=["contact", "goal"],
        rationale="The visible motion supports the interpretation.",
    )])
    assert checked_answer(data).claims[0].observation_ids == []


@pytest.mark.parametrize("change,match", [
    ({"frame_ids": ["stale_refinement_frame"]}, "unknown IDs"),
    ({"observation_ids": ["inactive_observation"]}, "unknown IDs"),
    ({"need_ids": ["other_need"]}, "unknown IDs"),
    ({"id": "option_0"}, "reserved"),
    ({"id": "obs_claim"}, "reserved"),
])
def test_claims_reject_unbound_or_ambiguous_references(change, match):
    data = answer()
    data["claims"][0].update(change)
    with pytest.raises(ValueError, match=match):
        checked_answer(data)


def test_observation_pixels_must_match_its_scene_and_active_final_registry():
    bad = observation()
    bad.scene = 2
    with pytest.raises(ValueError, match="scene must match"):
        validate_answer(answer(), frames(), [bad], ["contact", "goal"],
                        ["Move the item", "Signal somebody"])
    with pytest.raises(ValueError, match="unknown IDs"):
        validate_answer(answer(), frames()[1:], [observation()], ["contact", "goal"],
                        ["Move the item", "Signal somebody"])


def test_inference_needs_a_nonempty_reasoning_bridge():
    with pytest.raises(ValueError, match="requires a rationale"):
        Claim.model_validate(claim(kind="inferred", rationale="  "))


@pytest.mark.parametrize("parent", ["goal_claim", "missing"])
def test_claim_graph_rejects_forward_self_or_unknown_dependencies(parent):
    data = answer()
    data["claims"][0].update(
        kind="inferred", rationale="A proposed inference.", depends_on=[parent]
    )
    with pytest.raises(ValueError, match="earlier claims only"):
        checked_answer(data)


def test_claim_graph_preserves_context_but_rejects_duplicate_claim_ids():
    data = answer()
    data["claims"].append(claim(id="orphan"))
    original = deepcopy(data)
    value = checked_answer(data)
    assert claim_partition(value) == {
        "answer_lineage_ids": ["contact_claim", "goal_claim"],
        "option_lineage_ids": ["contact_claim", "goal_claim"],
        "context_claim_ids": ["orphan"], "limitation_claim_ids": [],
    }
    assert value.model_dump() == data == original
    data["claims"][-1]["id"] = "contact_claim"
    with pytest.raises(ValueError, match="unique"):
        checked_answer(data)


def test_multiple_supported_choices_keep_selection_separate():
    data = answer()
    data["option_comparisons"][1].update(status="supported", claim_ids=["contact_claim"])
    assert checked_answer(data).selected_choice == "Move the item"
    data["option_comparisons"][0]["status"] = "unsupported"
    with pytest.raises(ValueError, match="selected option must be supported"):
        checked_answer(data)


def test_supported_options_need_positive_claims_and_unsupported_options_may_have_no_refs():
    data = answer()
    assert checked_answer(data).option_comparisons[1].claim_ids == []
    data["option_comparisons"][1]["status"] = "supported"
    with pytest.raises(ValueError, match="requires observed or inferred claims"):
        checked_answer(data)


def test_limitations_cannot_be_used_as_positive_option_support():
    data = answer(claims=[claim(
        id="goal_claim", kind="limitation", statement="The motion is ambiguous.",
        observation_ids=[], need_ids=["contact", "goal"],
    )])
    with pytest.raises(ValueError, match="not limitations"):
        checked_answer(data)


@pytest.mark.parametrize("support", [
    "limitation_only", "direct_frame", "direct_observation", "positive_ancestor",
])
def test_positive_support_cannot_be_inherited_only_through_limitations(support):
    data = answer(claims=[
        claim(id="gap", kind="limitation", statement="The purpose is unclear."),
        claim(),  # Unrelated positive evidence cannot rescue the disconnected inference.
        claim(id="bridge", kind="inferred", frame_ids=["f2"] if support == "direct_frame" else [],
              observation_ids=["obs_contact"] if support == "direct_observation" else [],
              depends_on=["gap", "contact_claim"] if support == "positive_ancestor" else ["gap"],
              rationale="The cited evidence supports this interpretation."),
        claim(id="goal_claim", kind="inferred", frame_ids=[], observation_ids=[],
              depends_on=["bridge"], need_ids=["goal"],
              rationale="The earlier interpretation supports the purpose."),
    ])
    original = deepcopy(data)
    if support == "limitation_only":
        with pytest.raises(ValueError, match="positive evidence path"):
            checked_answer(data)
        result = answer_diagnostics(data, frames(), [observation()], ["contact", "goal"],
                                    ["Move the item", "Signal somebody"])
        assert "invalid_option_support" in {item["code"] for item in result["issues"]}
    else:
        assert checked_answer(data).selected_choice == "Move the item"
    assert data == original


def test_unresolved_multiple_choice_is_explicit_and_does_not_force_a_selection():
    data = answer(claims=[claim(
        id="goal_claim", kind="limitation", statement="The purpose remains unresolved.",
        frame_ids=["f1", "f1"], observation_ids=[], need_ids=["contact", "goal"],
    )], selected_choice=None)
    data["option_comparisons"][0].update(status="unsupported", claim_ids=[])
    value = checked_answer(data)
    assert value.selected_choice is None
    assert value.claims[0].frame_ids == ["f1"]
    assert data["claims"][0]["frame_ids"] == ["f1", "f1"]
    data["claims"][0]["kind"] = "observed"
    with pytest.raises(ValueError, match="explicit limitation"):
        checked_answer(data)


def test_ambiguous_multiple_choice_can_have_support_without_forcing_selection():
    data = answer(selected_choice=None)
    data["claims"].append(claim(
        id="ambiguity", kind="limitation", statement="Both purposes remain plausible.",
        need_ids=["contact", "goal"],
    ))
    data["answer_claim_ids"].append("ambiguity")
    data["option_comparisons"][1].update(status="supported", claim_ids=["contact_claim"])
    assert checked_answer(data).selected_choice is None


@pytest.mark.parametrize("change", [
    {"selected_choice": "Unknown option"}, {"selected_choice": None},
    {"option_comparisons": []},
])
def test_selected_option_and_exact_option_coverage_are_validated(change):
    with pytest.raises(ValueError):
        checked_answer(answer(**change))


def test_selected_option_cannot_be_justified_by_claims_outside_the_answer():
    data = answer(answer_claim_ids=["contact_claim"])
    data["claims"][0]["need_ids"] = ["contact", "goal"]
    with pytest.raises(ValueError, match="answer's claim lineage"):
        checked_answer(data)


def test_unaddressed_planner_needs_are_advisory_not_an_answer_obligation():
    data = answer(answer_claim_ids=["contact_claim"])
    data["option_comparisons"][0]["claim_ids"] = ["contact_claim"]
    data["option_comparisons"][1]["claim_ids"] = ["goal_claim"]
    original = deepcopy(data)
    value = checked_answer(data)
    coverage = answer_need_coverage(value, ["contact", "goal"])
    assert coverage == {"covered_need_ids": ["contact"], "unaddressed_need_ids": ["goal"]}
    result = answer_diagnostics(data, frames(), [observation()], ["contact", "goal"],
                                ["Move the item", "Signal somebody"])
    assert result["issues"] == []
    assert result["unaddressed_need_ids"] == result["advisories"][0]["ids"] == ["goal"]
    assert data == original
    with pytest.raises(ValueError, match="unknown IDs"):
        answer_need_coverage(value, ["contact"])
    data["claims"][1].update(kind="limitation", statement="The goal remains unclear.",
                             depends_on=[], rationale="")
    # A displayed limitation addresses a need without claiming to satisfy it.
    value = checked_answer(data)
    assert answer_need_coverage(value, ["goal", "contact"]) == {
        "covered_need_ids": ["goal", "contact"], "unaddressed_need_ids": [],
    }


def test_option_order_is_stable_for_reviewer_target_ids():
    data = answer()
    data["option_comparisons"].reverse()
    original = deepcopy(data)
    value = checked_answer(data)
    assert value == checked_answer(answer())
    assert data == original
    data["option_comparisons"][1] = deepcopy(data["option_comparisons"][0])
    with pytest.raises(ValueError, match="unique"):
        checked_answer(data)


def test_open_answer_is_choice_free_and_does_not_need_a_limitation():
    data = answer(selected_choice=None, option_comparisons=[])
    assert validate_answer(data, frames(), [observation()], ["contact", "goal"], [])
    data["selected_choice"] = "Move the item"
    with pytest.raises(ValueError, match="selected_choice"):
        validate_answer(data, frames(), [observation()], ["contact", "goal"], [])


@pytest.mark.parametrize("targets", [
    ["goal_claim"], ["goal_claim", "goal_claim"],
    ["goal_claim", "unknown"], ["goal_claim", "option_0", "extra"],
])
def test_review_requires_exact_nonduplicated_claim_and_option_coverage(targets):
    value = {"checks": [{"target_id": name, "status": "pass", "reason": "Checked pixels."}
                         for name in targets]}
    with pytest.raises(ValueError):
        validate_review(value, ["goal_claim", "option_0"])


def test_review_passes_or_requests_revision_without_model_authored_aggregate_acceptance():
    value = {"checks": [
        {"target_id": "goal_claim", "status": "revise",
         "reason": "Qualify this claim: the samples do not resolve the interaction."},
        {"target_id": "option_0", "status": "pass",
         "reason": "The unsupported assessment correctly preserves missing evidence."},
    ]}
    targets = ["goal_claim", "option_0"]
    assert [check.status for check in validate_review(value, targets).checks] == ["revise", "pass"]
    with pytest.raises(ValueError, match="Extra inputs"):
        validate_review({**value, "accepted": True}, targets)
    value["checks"][0]["status"] = "unsupported"
    with pytest.raises(ValueError, match="pass.*revise"):
        validate_review(value, targets)


def test_structured_models_and_mutated_records_are_revalidated():
    parsed = StructuredAnswer.model_validate(answer())
    parsed.claims[0].frame_ids.append("not_shown")
    with pytest.raises(ValueError, match="unknown IDs"):
        checked_answer(parsed)
    registry = frames()[:2]
    registry[0].seek_seconds = float("nan")
    with pytest.raises(ValueError):
        validate_window(window(), registry, ["contact"])


@pytest.mark.parametrize("schema", [
    WindowObservation, StructuredAnswer, StructuredReview, DirectStructuredReview,
])
def test_model_response_schemas_normalize_including_nested_records(schema):
    from videomasys.zerogpu.openai import strict_json_schema

    source = schema.model_json_schema()
    original = deepcopy(source)
    normalized = strict_json_schema(source)
    assert normalized["additionalProperties"] is False
    assert set(normalized["required"]) == set(normalized["properties"])
    assert source == original


def test_answer_wire_schema_encodes_kind_specific_constraints_without_discriminator():
    from videomasys.zerogpu.openai import strict_json_schema

    schema = strict_json_schema(StructuredAnswer.model_json_schema())
    items = schema["properties"]["claims"]["items"]
    assert set(items) == {"anyOf"}
    assert {branch["$ref"] for branch in items["anyOf"]} == {
        "#/$defs/ObservedClaim", "#/$defs/InferredClaim", "#/$defs/LimitationClaim",
    }
    for kind in ("ObservedClaim", "LimitationClaim"):
        fields = schema["$defs"][kind]["properties"]
        assert fields["depends_on"]["maxItems"] == 0
        assert fields["rationale"]["maxLength"] == 0
        assert "minItems" not in fields["frame_ids"]
    inferred = schema["$defs"]["InferredClaim"]["properties"]
    assert inferred["depends_on"]["maxItems"] == 3
    assert inferred["rationale"]["minLength"] == 1
    assert inferred["rationale"]["maxLength"] == 220


@pytest.mark.parametrize("schema,kind", [(ObservedClaim, "observed"),
                                         (LimitationClaim, "limitation")])
@pytest.mark.parametrize("change", [{"depends_on": ["earlier"]},
                                    {"rationale": "An independent explanation."}])
def test_direct_claim_variants_reject_dependencies_and_redundant_rationale(schema, kind, change):
    with pytest.raises(ValueError):
        schema.model_validate(claim(kind=kind, **change))


def direct_review(draft):
    return {
        "claim_checks": [
            {"target_id": item.id, "status": "pass", "reason": "Checked supplied evidence."}
            for item in draft.claims
        ],
        "option_assessments": [
            {"choice": item.choice, "status": item.status,
             "reason": "Support assessed from the supplied evidence."}
            for item in draft.option_comparisons
        ],
        "answer_judgment": {"status": "pass", "selected_choice": draft.selected_choice,
                            "reason": "This selection answers the question."},
    }


def test_context_and_limitations_remain_available_without_becoming_proof():
    data = answer()
    data["claims"].extend([
        claim(id="context", statement="A second worker watches."),
        claim(id="ambiguity", kind="limitation", statement="Exact contact is unclear."),
    ])
    draft = checked_answer(data)
    partition = claim_partition(draft)
    assert partition["context_claim_ids"] == ["context", "ambiguity"]
    assert partition["limitation_claim_ids"] == ["ambiguity"]
    assert partition["answer_lineage_ids"] == ["contact_claim", "goal_claim"]
    review = direct_review(draft)
    validate_direct_review(review, draft, [item.choice for item in draft.option_comparisons])
    review["claim_checks"] = review["claim_checks"][:2]
    with pytest.raises(ValueError, match="including context"):
        validate_direct_review(review, draft, [item.choice for item in draft.option_comparisons])
    data["answer_claim_ids"] = ["contact_claim"]
    data["option_comparisons"][0]["claim_ids"] = ["contact_claim"]
    value = checked_answer(data)
    assert answer_need_coverage(value, ["contact", "goal"])["unaddressed_need_ids"] == ["goal"]


def test_direct_review_reports_option_support_without_inverting_unsupported():
    draft = checked_answer(answer())
    review = direct_review(draft)
    review["option_assessments"][1]["reason"] = "No evidence of signalling is established."
    original = deepcopy(review)
    value = validate_direct_review(review, draft, ["Move the item", "Signal somebody"])
    assert value.option_assessments[1].status == "unsupported"
    assert value.answer_judgment.status == "pass"
    assert review == original
    # Disagreement is an actual model judgment, not a schema failure or automatic change.
    review["option_assessments"][0]["status"] = "unsupported"
    review["answer_judgment"].update(status="revise", selected_choice=None)
    value = validate_direct_review(review, draft, ["Move the item", "Signal somebody"])
    assert value.option_assessments[0].status == "unsupported"
    assert draft.selected_choice == "Move the item"


@pytest.mark.parametrize("mutation", ["missing_claim", "duplicate_claim", "missing_option"])
def test_direct_review_requires_complete_unique_claim_and_ordered_option_judgments(mutation):
    draft = checked_answer(answer())
    review = direct_review(draft)
    if mutation == "missing_claim":
        review["claim_checks"].pop()
    elif mutation == "duplicate_claim":
        review["claim_checks"][1] = deepcopy(review["claim_checks"][0])
    else:
        review["option_assessments"].pop()
    with pytest.raises(ValueError):
        validate_direct_review(review, draft, ["Move the item", "Signal somebody"])


def test_direct_review_canonicalizes_option_order_without_reassigning_judgments():
    draft = checked_answer(answer())
    review = direct_review(draft)
    review["option_assessments"].reverse()
    original = deepcopy(review)
    value = validate_direct_review(review, draft, ["Move the item", "Signal somebody"])
    assert value.model_dump() == direct_review(draft)
    assert review == original
    review["option_assessments"][1] = deepcopy(review["option_assessments"][0])
    with pytest.raises(ValueError, match="unique"):
        validate_direct_review(review, draft, ["Move the item", "Signal somebody"])


def test_direct_review_alternate_selection_requires_revision_and_known_option():
    draft = checked_answer(answer())
    review = direct_review(draft)
    review["answer_judgment"]["selected_choice"] = "Signal somebody"
    review["option_assessments"][1]["status"] = "supported"
    with pytest.raises(ValueError, match="passing answer judgment"):
        validate_direct_review(review, draft, ["Move the item", "Signal somebody"])
    review["answer_judgment"]["status"] = "revise"
    assert validate_direct_review(review, draft, ["Move the item", "Signal somebody"])
    review["answer_judgment"]["selected_choice"] = "Invented choice"
    with pytest.raises(ValueError, match="supplied option or null"):
        validate_direct_review(review, draft, ["Move the item", "Signal somebody"])
    open_draft = validate_answer(answer(selected_choice=None, option_comparisons=[]),
                                 frames(), [observation()], ["contact", "goal"], [])
    assert validate_direct_review(direct_review(open_draft), open_draft, [])


@pytest.mark.parametrize("judgment_status", ["pass", "revise"])
def test_direct_review_selected_choice_requires_its_own_supported_assessment(judgment_status):
    draft = checked_answer(answer())
    review = direct_review(draft)
    review["answer_judgment"]["status"] = judgment_status
    review["option_assessments"][0]["status"] = "unsupported"
    with pytest.raises(ValueError, match="'Move the item' must be marked supported"):
        validate_direct_review(review, draft, ["Move the item", "Signal somebody"])
    assert draft.selected_choice == "Move the item"


def test_answer_diagnostics_reports_simultaneous_ids_coverage_and_selection_conflicts():
    data = answer()
    data["claims"][0].update(frame_ids=["stale"], observation_ids=["inactive"])
    data["claims"][1]["depends_on"] = ["missing_parent"]
    data["option_comparisons"][0]["status"] = "unsupported"
    original = deepcopy(data)
    result = answer_diagnostics(data, frames(), [observation()], ["contact", "goal"],
                                ["Move the item", "Signal somebody"])
    invalid = {item["path"]: item["ids"] for item in result["issues"]
               if item["code"] == "invalid_reference"}
    assert invalid == {
        "claims[0].frame_ids": ["stale"], "claims[0].observation_ids": ["inactive"],
        "claims[1].depends_on": ["missing_parent"],
    }
    assert result["unaddressed_need_ids"] == ["contact"]
    assert result["advisories"][0]["ids"] == ["contact"]
    assert "missing_need_coverage" not in {item["code"] for item in result["issues"]}
    assert result["context_claim_ids"] == ["contact_claim"]
    assert result["available_observation_ids_by_need"] == {"contact": ["obs_contact"], "goal": []}
    assert "selected_status_conflict" in {item["code"] for item in result["issues"]}
    assert data == original
    with pytest.raises(ValueError):
        checked_answer(data)


def test_answer_diagnostics_does_not_invent_errors_or_loop_on_cyclic_raw_dependencies():
    result = answer_diagnostics(answer(), frames(), [observation()], ["contact", "goal"],
                                ["Move the item", "Signal somebody"])
    assert result["issues"] == result["unaddressed_need_ids"] == result["context_claim_ids"] == []
    assert result["advisories"] == []
    data = answer()
    data["claims"][0].update(kind="inferred", rationale="An unsupported reverse link.",
                             depends_on=["goal_claim"])
    result = answer_diagnostics(data, frames(), [observation()], ["contact", "goal"],
                                ["Move the item", "Signal somebody"])
    assert any(item["path"] == "claims[0].depends_on" for item in result["issues"])


def test_answer_diagnostics_explains_abstention_and_positive_support_without_repairing():
    data = answer(selected_choice=None)
    data["claims"][0]["kind"] = "limitation"
    data["option_comparisons"][1].update(status="supported", claim_ids=["contact_claim"])
    original = deepcopy(data)
    result = answer_diagnostics(data, frames(), [observation()], ["contact", "goal"],
                                ["Move the item", "Signal somebody"])
    codes = {item["code"] for item in result["issues"]}
    assert {"missing_abstention_limitation", "invalid_option_support"} <= codes
    assert data == original
    with pytest.raises(ValueError):
        checked_answer(data)


@pytest.mark.parametrize("kind,code", [
    ("inferred", "selected_lineage_conflict"),
    ("limitation", "missing_abstention_limitation"),
])
def test_root_omission_diagnostics_identify_existing_conclusion_without_promoting_it(kind, code):
    data = answer(
        claims=[claim(id=f"c{i}", need_ids=["contact", "goal"]) for i in range(1, 4)] + [
            claim(id="c4", kind=kind, need_ids=["contact", "goal"],
                  depends_on=["c1", "c2", "c3"] if kind == "inferred" else [],
                  rationale="The premises support this purpose." if kind == "inferred" else ""),
        ],
        answer_claim_ids=["c1", "c2", "c3"],
        selected_choice="Move the item" if kind == "inferred" else None,
    )
    data["option_comparisons"][0].update(
        status="supported" if kind == "inferred" else "unsupported", claim_ids=["c4"],
    )
    original = deepcopy(data)
    result = answer_diagnostics(data, frames(), [observation()], ["contact", "goal"],
                                ["Move the item", "Signal somebody"])
    issue = next(item for item in result["issues"] if item["code"] == code)
    assert issue["ids"] == ["c4"]
    assert "answer_claim_ids" in issue["message"] and "premise-only roots" in issue["message"]
    assert data == original
    with pytest.raises(ValueError):
        checked_answer(data)
    # Only an explicit model/user root change fixes this graph; no new edges are needed.
    corrected = deepcopy(data)
    corrected["answer_claim_ids"] = ["c4"]
    assert checked_answer(corrected).claims == StructuredAnswer.model_validate(data).claims


@pytest.mark.parametrize("previous,selected", [
    ("Signal somebody", "Move the item"), ("Move the item", None), (None, "Move the item"),
])
def test_choice_changes_need_an_explicit_reason_without_rewriting_selection(previous, selected):
    data = answer(selected_choice=selected, choice_change_reason="  ")
    original = deepcopy(data)
    with pytest.raises(ValueError, match="choice_change_reason"):
        validate_choice_change(data, previous)
    assert data == original
    data["choice_change_reason"] = "The sampled contact supports a different interpretation."
    value = validate_choice_change(data, previous)
    assert value.selected_choice == selected
    assert value.model_dump() == data
    assert validate_choice_change(answer(), "Move the item").choice_change_reason == ""
