import asyncio

import pytest

from videomasys.agents.tool_sequence import (
    TOOL_DATAFLOW_RULES,
    InvalidToolProgram,
    ToolProgramValidator,
    ToolSequenceAgent,
)
from videomasys.domain.schemas import (
    EvidenceNeed,
    LocateCall,
    ReasoningPlan,
    RelateCall,
    ToolProgram,
)


@pytest.fixture
def plan():
    return ReasoningPlan(
        objective="Explain an action from its observed context and outcome.",
        evidence_needs=[
            EvidenceNeed(id="anchor", description="An initial action", temporal_role="anchor"),
            EvidenceNeed(
                id="outcome",
                description="The subsequent outcome",
                temporal_role="outcome",
                depends_on=["anchor"],
            ),
        ],
        max_intervals=3,
        max_inspected_seconds=60,
        stop_condition="The evidence distinguishes the candidate explanations.",
        reasoning="An outcome can support an inference about the initial action.",
    )


@pytest.fixture
def program(plan):
    return ToolSequenceAgent._fallback(plan)


def test_rejects_user_reported_unused_temporal_output(plan, program):
    program.inspect.sources = ["l1", "l2"]
    with pytest.raises(InvalidToolProgram, match="unused retrieval outputs: r1"):
        ToolProgramValidator().validate(program, plan)
    # Reject/repair, rather than quietly adding a source the model never consumed.
    assert program.inspect.sources == ["l1", "l2"]


def test_one_repair_error_explains_all_missing_required_relation_connections(plan):
    plan.evidence_needs.append(
        EvidenceNeed(
            id="later",
            description="A later state change",
            temporal_role="after",
            depends_on=["outcome"],
        )
    )
    program = ToolSequenceAgent._fallback(plan)
    program.inspect.sources = ["l1", "l2", "l3"]
    with pytest.raises(InvalidToolProgram) as failure:
        ToolProgramValidator().validate(program, plan)
    message = str(failure.value)
    assert "unused retrieval outputs: r1, r2" in message
    assert "dependency anchor -> outcome: add r1 to inspect.sources" in message
    assert "dependency outcome -> later: add r2 to inspect.sources" in message
    assert "deleting these calls breaks the dependency" in message
    assert "or remove" not in message
    assert len(message) < 1000  # The live controller's repair-error budget.
    assert program.inspect.sources == ["l1", "l2", "l3"]


def test_repair_error_aggregates_missing_source_coverage_with_missing_dependency(plan, program):
    program.inspect.sources = ["l2"]
    with pytest.raises(InvalidToolProgram) as failure:
        ToolProgramValidator().validate(program, plan)
    message = str(failure.value)
    assert "unused retrieval outputs" in message
    assert "do not supply evidence for: anchor" in message
    assert "dependency anchor -> outcome: add r1 to inspect.sources" in message


def test_rejects_unused_direct_probe(plan, program):
    program.inspect.sources = ["l1", "r1"]
    with pytest.raises(InvalidToolProgram, match="unused retrieval outputs: l2"):
        ToolProgramValidator().validate(program, plan)


def test_declared_inspect_labels_do_not_replace_actual_source_coverage(plan, program):
    # l1 is used as r1's source, but no inspected output supplies anchor evidence.
    program.inspect.sources = ["l2", "r1"]
    with pytest.raises(InvalidToolProgram, match="do not supply evidence for: anchor"):
        ToolProgramValidator().validate(program, plan)


def test_dependency_cannot_be_satisfied_with_independent_locates_only(plan, program):
    program.relate = []
    program.inspect.sources = ["l1", "l2"]
    with pytest.raises(InvalidToolProgram, match="dependency anchor -> outcome"):
        ToolProgramValidator().validate(program, plan)


@pytest.mark.parametrize("mutation", ["wrong_source", "wrong_target", "wrong_direction"])
def test_temporal_call_must_implement_the_declared_edge(plan, program, mutation):
    if mutation == "wrong_source":
        program.relate[0].source = "l2"
    elif mutation == "wrong_target":
        program.relate[0].evidence_need_ids = ["anchor"]
    else:
        program.relate[0].relation = "before"
    with pytest.raises(InvalidToolProgram, match="dependency anchor -> outcome"):
        ToolProgramValidator().validate(program, plan)


def test_before_need_requires_before_traversal(plan):
    plan.evidence_needs[1].temporal_role = "before"
    program = ToolSequenceAgent._fallback(plan)
    assert program.relate[0].relation == "before"
    ToolProgramValidator().validate(program, plan)


def test_independent_context_does_not_require_a_temporal_relation(plan):
    plan.evidence_needs[1].temporal_role = "context"
    plan.evidence_needs[1].depends_on = []
    program = ToolSequenceAgent._fallback(plan)
    assert not program.relate
    ToolProgramValidator().validate(program, plan)


def test_dependency_chain_requires_intermediate_temporal_candidates(plan):
    plan.evidence_needs.append(
        EvidenceNeed(
            id="later",
            description="A later state change",
            temporal_role="after",
            depends_on=["outcome"],
        )
    )
    program = ToolSequenceAgent._fallback(plan)
    assert program.relate[1].source == "r1"
    ToolProgramValidator().validate(program, plan)

    # r1 is no longer dead code, but its temporal outcome candidates are not inspected.
    program.inspect.sources.remove("r1")
    with pytest.raises(InvalidToolProgram, match="dependency anchor -> outcome"):
        ToolProgramValidator().validate(program, plan)


def test_retrieval_anchor_may_be_transitively_used_if_other_source_supplies_evidence(plan):
    plan.evidence_needs[1].depends_on = []
    program = ToolSequenceAgent._fallback(plan)
    program.locate.append(
        LocateCall(id="l3", query="Alternative anchor description", evidence_need_ids=["anchor"])
    )
    program.relate = [
        RelateCall(id="r1", source="l1", relation="around", evidence_need_ids=["outcome"])
    ]
    program.inspect.sources = ["l2", "l3", "r1"]
    ToolProgramValidator().validate(program, plan)


def test_each_of_multiple_dependencies_must_be_covered(plan):
    plan.evidence_needs.insert(
        1,
        EvidenceNeed(id="context", description="A second initial action", temporal_role="context"),
    )
    plan.evidence_needs[-1].depends_on.append("context")
    program = ToolSequenceAgent._fallback(plan)
    assert len(program.relate) == 2
    ToolProgramValidator().validate(program, plan)
    program.relate = program.relate[:1]
    program.inspect.sources.remove("r2")
    with pytest.raises(InvalidToolProgram, match="dependency context -> outcome"):
        ToolProgramValidator().validate(program, plan)


def test_fallback_with_small_inspection_budget_still_has_valid_retrieval_limits(plan):
    plan.max_intervals = 1
    plan.max_inspected_seconds = 5
    program = ToolSequenceAgent._fallback(plan)
    assert program.relate[0].limit == 2
    assert program.relate[0].horizon_seconds == 20
    ToolProgramValidator().validate(program, plan)


def test_compiler_requests_a_model_repair_for_dead_code(plan, program):
    class Gateway:
        def __init__(self):
            self.prompts = []

        async def complete_structured(self, **kwargs):
            self.prompts.append(kwargs["prompt"])
            result = program.model_copy(deep=True)
            if len(self.prompts) == 1:
                result.inspect.sources.remove("r1")
            return result

    gateway = Gateway()
    compiled = asyncio.run(
        ToolSequenceAgent(gateway).compile(
            question="Why did the person act?", options=None, plan=plan
        )
    )
    assert compiled.compiler == "model_repaired"
    assert "r1" in compiled.inspect.sources
    assert len(gateway.prompts) == 2
    assert "unused retrieval outputs: r1" in gateway.prompts[1]
    assert all(TOOL_DATAFLOW_RULES.strip() in prompt for prompt in gateway.prompts)


def test_duplicate_evidence_labels_are_rejected(plan, program):
    program.relate[0].evidence_need_ids = ["outcome", "outcome"]
    with pytest.raises(InvalidToolProgram, match="evidence need ids must be unique"):
        ToolProgramValidator().validate(program, plan)


def test_reported_attire_only_program_names_environment_need_and_complete_repair():
    # Reproduce the held-out compiler's structural error, without answer labels
    # or any automatic controller probe insertion masking the invalid program.
    plan = ReasoningPlan(
        objective="Explain the man's attire and its contextual relevance.",
        evidence_needs=[
            EvidenceNeed(
                id="attire_observation",
                description="Describe the man's clothing and any visible body parts exposed.",
                temporal_role="context",
            ),
            EvidenceNeed(
                id="environment_context",
                description="Note surrounding elements indicating setting or social norms.",
                temporal_role="context",
            ),
        ],
        max_intervals=3,
        max_inspected_seconds=60,
        stop_condition="The attire and surrounding context are inspected.",
        reasoning="Distinguish observed context from inferred causes.",
    )
    program = ToolProgram(
        rationale="Inspect the anchor and subsequent interaction.",
        max_intervals=3,
        max_inspected_seconds=60,
        locate=[
            LocateCall(
                id="l1",
                query="Man holds a glass of amber liquid",
                evidence_need_ids=["attire_observation"],
            ),
            LocateCall(
                id="l2",
                query="Man looks down at the glass and closes his eyes",
                evidence_need_ids=["attire_observation"],
            ),
        ],
        inspect={
            "sources": ["l1", "l2"],
            "evidence_need_ids": ["attire_observation"],
        },
    )
    before = program.model_dump()
    with pytest.raises(InvalidToolProgram) as failure:
        ToolProgramValidator().validate(program, plan)
    message = str(failure.value)
    assert "missing direct locate IDs: environment_context" in message
    assert "inspect.evidence_need_ids: environment_context" in message
    assert "inspect.sources do not supply evidence for: environment_context" in message
    assert plan.evidence_needs[1].description in message[:1000]
    assert (
        "direct locate queries" in message and "include the outputs in inspect.sources" in message
    )
    assert "Do not only relabel an unrelated query" in message
    assert program.model_dump() == before

    # A label-only repair must still fail actual retrieval coverage.
    repaired = program.model_copy(deep=True)
    repaired.inspect.evidence_need_ids.append("environment_context")
    with pytest.raises(InvalidToolProgram, match="missing direct locate IDs: environment_context"):
        ToolProgramValidator().validate(repaired, plan)

    repaired.locate[1].query = plan.evidence_needs[1].description
    repaired.locate[1].evidence_need_ids = ["environment_context"]
    assert ToolProgramValidator().validate(repaired, plan) is repaired
    assert program.model_dump() == before


def test_direct_locate_deficit_is_distinct_from_valid_relation_coverage(plan, program):
    program.locate.pop()
    program.inspect.sources = ["l1", "r1"]
    before = program.model_dump()
    with pytest.raises(InvalidToolProgram) as failure:
        ToolProgramValidator().validate(program, plan)
    message = str(failure.value)
    assert "missing direct locate IDs: outcome" in message
    assert plan.evidence_needs[1].description in message
    assert "inspect.sources do not supply" not in message
    assert "missing inspect.evidence_need_ids" not in message
    assert "dependency anchor -> outcome" not in message
    assert program.model_dump() == before


def test_one_error_aggregates_locate_labels_sources_unused_outputs_and_dependencies(plan):
    plan.evidence_needs.append(
        EvidenceNeed(
            id="later",
            description="A later state change",
            temporal_role="after",
            depends_on=["outcome"],
        )
    )
    program = ToolSequenceAgent._fallback(plan)
    program.locate.pop()
    program.inspect.sources = ["l1"]
    program.inspect.evidence_need_ids = ["anchor"]
    before = program.model_dump()
    with pytest.raises(InvalidToolProgram) as failure:
        ToolProgramValidator().validate(program, plan)
    message = str(failure.value)
    for detail in (
        "missing direct locate IDs: later",
        "inspect.evidence_need_ids: later, outcome",
        "unused retrieval outputs: l2, r1, r2",
        "inspect.sources do not supply evidence for: later, outcome",
        "dependency anchor -> outcome: add r1 to inspect.sources",
        "dependency outcome -> later: add r2 to inspect.sources",
    ):
        assert detail in message[:1000]
    assert "later: 'A later state change'" in message
    assert program.model_dump() == before


def test_inspect_label_and_actual_source_deficits_name_different_needs(plan, program):
    program.inspect.sources = ["l2", "r1"]
    program.inspect.evidence_need_ids = ["anchor"]
    with pytest.raises(InvalidToolProgram) as failure:
        ToolProgramValidator().validate(program, plan)
    message = str(failure.value)
    assert "inspect.evidence_need_ids: outcome" in message
    assert "inspect.sources do not supply evidence for: anchor" in message
    assert "missing direct locate IDs" not in message
    assert "unused retrieval outputs: l1" not in message


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("relation_source", "unknown or later source"),
        ("inspect_source", "inspect references an unknown retrieval output"),
        ("locate_need", "l1 references an unknown evidence need"),
        ("relate_need", "r1 references an unknown evidence need"),
        ("inspect_need", "inspect references an unknown evidence need"),
        ("inspect_duplicate", "inspect sources must be unique"),
        ("need_duplicate", "evidence need ids must be unique"),
        ("locate_sequence", "locate ids must be sequential"),
        ("relate_sequence", "relate ids must be sequential"),
    ],
)
def test_invalid_structure_fails_before_aggregating_coverage(plan, program, mutation, expected):
    # Every variant also lacks a direct outcome locate. Structural faults must
    # take priority, so unsafe graph traversal never hides behind that omission.
    program.locate[1].evidence_need_ids = ["anchor"]
    if mutation == "relation_source":
        program.relate[0].source = "r2"
    elif mutation == "inspect_source":
        program.inspect.sources.append("l4")
    elif mutation == "locate_need":
        program.locate[0].evidence_need_ids = ["unknown"]
    elif mutation == "relate_need":
        program.relate[0].evidence_need_ids = ["unknown"]
    elif mutation == "inspect_need":
        program.inspect.evidence_need_ids = ["unknown"]
    elif mutation == "inspect_duplicate":
        program.inspect.sources.append("l1")
    elif mutation == "need_duplicate":
        program.inspect.evidence_need_ids = ["anchor", "anchor"]
    elif mutation == "locate_sequence":
        program.locate[1].id = "l3"
    else:
        program.relate[0].id = "r2"
    before = program.model_dump()
    with pytest.raises(InvalidToolProgram, match=expected) as failure:
        ToolProgramValidator().validate(program, plan)
    assert "missing direct locate" not in str(failure.value)
    assert program.model_dump() == before


def test_diagnostic_aggregation_preserves_exact_candidate_budget(plan, program):
    for index in (3, 4):
        program.locate.append(
            LocateCall(id=f"l{index}", query="Additional context", evidence_need_ids=["anchor"])
        )
        program.inspect.sources.append(f"l{index}")
    for call in program.locate:
        call.limit = 6
    assert ToolProgramValidator().validate(program, plan) is program
    program.locate[-1].limit = 7
    before = program.model_dump()
    with pytest.raises(InvalidToolProgram, match="locate candidate budget exceeds 24"):
        ToolProgramValidator().validate(program, plan)
    assert program.model_dump() == before
