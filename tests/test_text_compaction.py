from videomasys.tasks.qa.pipeline import (
    canonical_multiple_choice_answer,
    compact_observation,
)


def test_compact_observation_removes_duplicate_sentences() -> None:
    value = "The person washes a fork. The person washes a fork. Then they rinse it."

    assert compact_observation(value) == (
        "The person washes a fork. Then they rinse it."
    )


def test_compact_observation_respects_character_budget() -> None:
    compacted = compact_observation("word " * 100, max_characters=40)

    assert len(compacted) <= 40
    assert compacted.endswith("…")


def test_multiple_choice_answer_uses_zero_based_index() -> None:
    assert canonical_multiple_choice_answer(1, ["first", "second"]) == "[1] second"
