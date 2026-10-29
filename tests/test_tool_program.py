import pytest
from pydantic import ValidationError

from videomasys.agents.tool_sequence import (
    InvalidToolProgram,
    ToolProgramValidator,
    ToolSequenceAgent,
)
from videomasys.domain.schemas import ReasoningPlan, ToolProgram


def reasoning_plan() -> ReasoningPlan:
    return ReasoningPlan(
        objective="Determine why the person moved the object.",
        evidence_needs=[
            {
                "id": "pickup",
                "description": "The person picks up the object.",
                "temporal_role": "anchor",
            },
            {
                "id": "outcome",
                "description": "What the person does with it next.",
                "temporal_role": "outcome",
                "depends_on": ["pickup"],
            },
        ],
        temporal_constraints=["The outcome must occur after the pickup."],
        max_intervals=4,
        max_inspected_seconds=80,
        stop_condition="The anchor and its outcome distinguish the choices.",
        reasoning="A purpose is supported by what happens after the anchor action.",
    )


def valid_program() -> ToolProgram:
    return ToolProgram(
        rationale="Locate the anchor, relate forward, then inspect both outputs.",
        max_intervals=4,
        max_inspected_seconds=80,
        locate=[
            {
                "id": "l1",
                "query": "person picks up object",
                "limit": 6,
                "evidence_need_ids": ["pickup"],
            },
            {
                "id": "l2",
                "query": "person uses object afterward",
                "limit": 6,
                "evidence_need_ids": ["outcome"],
            },
        ],
        relate=[
            {
                "id": "r1",
                "source": "l1",
                "relation": "after",
                "horizon_seconds": 60,
                "limit": 4,
                "evidence_need_ids": ["outcome"],
            }
        ],
        inspect={
            "sources": ["l1", "l2", "r1"],
            "evidence_need_ids": ["pickup", "outcome"],
        },
    )


def test_reasoning_plan_requires_prior_dependencies() -> None:
    with pytest.raises(ValidationError):
        ReasoningPlan(
            objective="Resolve an ordered event.",
            evidence_needs=[
                {
                    "id": "outcome",
                    "description": "Observe the outcome.",
                    "temporal_role": "outcome",
                    "depends_on": ["missing_anchor"],
                }
            ],
            stop_condition="The outcome is observed.",
            reasoning="Compare ordered events.",
        )


def test_validator_accepts_bounded_locate_relate_program() -> None:
    program = valid_program()
    assert ToolProgramValidator().validate(program, reasoning_plan()) is program


def test_validator_rejects_unknown_relation_source() -> None:
    program = valid_program()
    program.relate[0].source = "r2"
    with pytest.raises(InvalidToolProgram, match="unknown or later source"):
        ToolProgramValidator().validate(program, reasoning_plan())


def test_validator_rejects_incomplete_inspection_coverage() -> None:
    program = valid_program()
    program.inspect.evidence_need_ids = ["pickup"]
    with pytest.raises(InvalidToolProgram, match="cover every"):
        ToolProgramValidator().validate(program, reasoning_plan())


def test_validator_rejects_plan_budget_violation() -> None:
    program = valid_program()
    program.max_inspected_seconds = 100
    with pytest.raises(InvalidToolProgram, match="video-seconds budget"):
        ToolProgramValidator().validate(program, reasoning_plan())


def test_controller_adds_direct_locate_for_uncovered_need() -> None:
    program = valid_program()
    program.locate = program.locate[:1]
    program.inspect.sources = ["l1", "r1"]

    normalized = ToolSequenceAgent._ensure_recall_coverage(program, reasoning_plan())

    assert normalized.locate[-1].evidence_need_ids == ["outcome"]
    assert normalized.locate[-1].id in normalized.inspect.sources
    ToolProgramValidator().validate(normalized, reasoning_plan())
