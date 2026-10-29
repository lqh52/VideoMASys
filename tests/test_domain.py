from uuid import uuid4

import pytest
from pydantic import ValidationError

from videomasys.domain.schemas import (
    Citation,
    MultipleChoiceDecision,
    QAAnswer,
    QAPlan,
    Verification,
)


def test_plan_normalizes_and_deduplicates_queries() -> None:
    plan = QAPlan(
        search_queries=["  person   enters ", "person enters"],
        reasoning="Find the entry event.",
    )
    assert plan.search_queries == ["person enters"]


def test_plan_rejects_unbounded_interval_budget() -> None:
    with pytest.raises(ValidationError):
        QAPlan(search_queries=["anything"], max_intervals=100, reasoning="Too broad")


def test_answer_contract_accepts_typed_citations() -> None:
    node_id = uuid4()
    answer = QAAnswer(
        answer="A person enters.",
        confidence=0.8,
        citations=[Citation(node_id=node_id, start_seconds=1.0, end_seconds=2.0)],
    )
    assert answer.citations[0].node_id == node_id


def test_verifier_rejection_requires_an_actionable_reason() -> None:
    with pytest.raises(ValidationError):
        Verification(accepted=False, issues=[], confidence=0.7)


def test_verifier_correction_is_atomic() -> None:
    with pytest.raises(ValidationError):
        Verification(
            accepted=False,
            issues=["Wrong option."],
            corrected_selected_option=2,
            confidence=0.7,
        )


def test_choice_judge_cannot_select_a_contradicted_option() -> None:
    with pytest.raises(ValidationError):
        MultipleChoiceDecision(
            assessments=[
                {"option_index": 0, "verdict": "contradicted", "reason": "Not seen."},
                {"option_index": 1, "verdict": "supported", "reason": "Observed."},
            ],
            selected_option=0,
            reason="Invalid selection.",
        )
