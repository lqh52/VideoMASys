from uuid import uuid4

from videomasys.agents.qa import valid_citation_ids
from videomasys.domain.schemas import Citation, QAAnswer


def test_citation_allowlist() -> None:
    allowed = uuid4()
    rejected = uuid4()
    answer = QAAnswer(
        answer="Observation",
        confidence=0.7,
        citations=[Citation(node_id=allowed, start_seconds=0, end_seconds=1)],
    )
    assert valid_citation_ids(answer, {allowed})
    answer.citations.append(Citation(node_id=rejected, start_seconds=1, end_seconds=2))
    assert not valid_citation_ids(answer, {allowed})

