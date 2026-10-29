"""The global compiler contract is a hard gate, not a prompt-only preference."""

import pytest

from videomasys.agents.tool_sequence import InvalidToolProgram, ToolProgramValidator
from videomasys.domain.schemas import LocateCall, RelateCall, ToolProgram
from videomasys.zerogpu.core import QAPlan


def whole_plan():
    return QAPlan(
        scope="whole_video",
        objective="Determine the overall task across the video.",
        evidence_needs=[
            {
                "id": need,
                "description": f"Representative actions during the {need} period.",
                "temporal_role": "context",
            }
            for need in ("early", "middle", "late")
        ],
        max_intervals=3,
        max_inspected_seconds=60,
        stop_condition="Compare independently sampled periods.",
        reasoning="The whole activity requires coverage across time.",
    )


def whole_program():
    return ToolProgram(
        rationale="Inspect independent representatives across the timeline.",
        max_intervals=3,
        max_inspected_seconds=60,
        locate=[
            {
                "id": f"l{index}",
                "query": f"Representative visible actions during {need}",
                "limit": 4,
                "evidence_need_ids": [need],
                "temporal_position": position,
            }
            for index, (need, position) in enumerate(
                zip(("early", "middle", "late"), ("beginning", "middle", "end"), strict=True),
                start=1,
            )
        ],
        inspect={"sources": ["l1", "l2", "l3"], "evidence_need_ids": ["early", "middle", "late"]},
    )


def test_independent_whole_video_program_passes_without_mutation():
    program = whole_program()
    before = program.model_dump()
    assert ToolProgramValidator().validate(program, whole_plan()) is program
    assert program.model_dump() == before


def test_supplied_two_relation_global_shape_is_rejected_with_actionable_repair():
    program = whole_program()
    program.relate = [
        RelateCall(
            id=f"r{index}",
            source=f"l{index}",
            relation="after",
            horizon_seconds=60,
            limit=4,
            evidence_need_ids=[need],
        )
        for index, need in enumerate(("middle", "late"), start=1)
    ]
    program.inspect.sources += ["r1", "r2"]
    with pytest.raises(InvalidToolProgram, match="whole_video forbids relate") as failure:
        ToolProgramValidator().validate(program, whole_plan())
    assert "set relate=[]" in str(failure.value)
    assert "inspect.sources must contain exactly l1,l2,l3" in str(failure.value)


@pytest.mark.parametrize("position", ["any", "middle", "end"])
def test_first_need_cannot_be_relabelled_as_another_temporal_probe(position):
    program = whole_program()
    program.locate[0].temporal_position = position
    with pytest.raises(
        InvalidToolProgram, match="l1 must carry only.*temporal_position='beginning'"
    ):
        ToolProgramValidator().validate(program, whole_plan())


def test_one_probe_cannot_claim_multiple_periods_even_when_all_labels_are_present():
    program = whole_program()
    program.locate[0].evidence_need_ids = ["early", "middle", "late"]
    with pytest.raises(InvalidToolProgram, match="l1 must carry only"):
        ToolProgramValidator().validate(program, whole_plan())


def test_plan_order_determines_need_to_period_mapping():
    program = whole_program()
    program.locate[0].evidence_need_ids = ["late"]
    program.locate[2].evidence_need_ids = ["early"]
    with pytest.raises(InvalidToolProgram, match="preserve plan order"):
        ToolProgramValidator().validate(program, whole_plan())


@pytest.mark.parametrize("count", [2, 4])
def test_exactly_three_direct_locates_are_required(count):
    program = whole_program()
    if count == 2:
        program.locate.pop()
        program.inspect.sources.pop()
    else:
        program.locate.append(
            LocateCall(id="l4", query="Extra action probe", evidence_need_ids=["late"])
        )
        program.inspect.sources.append("l4")
    with pytest.raises(InvalidToolProgram, match="exactly three direct locates"):
        ToolProgramValidator().validate(program, whole_plan())


@pytest.mark.parametrize("field", ["sources", "evidence_need_ids"])
def test_inspect_requires_exact_global_outputs_and_needs(field):
    program = whole_program()
    getattr(program.inspect, field).pop()
    with pytest.raises(InvalidToolProgram, match=f"whole_video inspect.{field}"):
        ToolProgramValidator().validate(program, whole_plan())


def test_inspect_listing_order_does_not_change_independent_probe_contract():
    program = whole_program()
    program.inspect.sources.reverse()
    program.inspect.evidence_need_ids.reverse()
    assert ToolProgramValidator().validate(program, whole_plan()) is program


def test_local_event_dependency_still_requires_and_accepts_temporal_relation():
    plan = whole_plan().model_copy(update={"scope": "local_event"})
    plan.evidence_needs = plan.evidence_needs[:2]
    plan.evidence_needs[1].depends_on = ["early"]
    plan.evidence_needs[1].temporal_role = "outcome"
    program = whole_program()
    program.locate = program.locate[:2]
    program.inspect.sources = ["l1", "l2"]
    program.inspect.evidence_need_ids = ["early", "middle"]
    with pytest.raises(InvalidToolProgram, match="requires an inspected relate"):
        ToolProgramValidator().validate(program, plan)
    program.relate = [
        RelateCall(id="r1", source="l1", relation="after", evidence_need_ids=["middle"])
    ]
    program.inspect.sources.append("r1")
    assert ToolProgramValidator().validate(program, plan) is program
