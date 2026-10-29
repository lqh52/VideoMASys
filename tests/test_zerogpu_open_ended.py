"""Real controller with isolated model fixtures: open answers never need choices."""

import json

import numpy as np
import pytest

from videomasys.zerogpu.core import LivePipeline, SceneCache


def run_open(tmp_path, answers, verdict=None, after_review=()):
    replies = iter(
        [
            {
                "scope": "local_event",
                "objective": "Observe the event",
                "evidence_needs": [
                    {
                        "id": "context",
                        "description": "The visible action and setting",
                        "temporal_role": "context",
                        "depends_on": [],
                    }
                ],
                "max_intervals": 1,
                "max_inspected_seconds": 20,
                "stop_condition": "Enough visible context",
                "reasoning": "Observe before inferring",
            },
            {
                "rationale": "Inspect the context",
                "max_intervals": 1,
                "max_inspected_seconds": 20,
                "locate": [
                    {
                        "id": "l1",
                        "query": "Visible action and setting",
                        "limit": 2,
                        "evidence_need_ids": ["context"],
                    }
                ],
                "relate": [],
                "inspect": {"sources": ["l1"], "evidence_need_ids": ["context"]},
            },
            {"scene": 0, "observation": "A person handles an object.", "relevant": True},
            *answers,
            verdict or {"accepted": True, "issues": []},
            *after_review,
        ]
    )
    calls = []

    def generate(prompt, images, tokens):
        calls.append(prompt)
        return json.dumps(next(replies))

    pipeline = LivePipeline(generate, lambda _: np.asarray([[1.0, 0.0]]), "fixture", "fixture")
    scene = {
        "index": 0,
        "start": 0,
        "end": 20,
        "frames": ["fixture.jpg"],
        "caption": "A person handles an object.",
    }
    bundle = {"scenes": [scene], "database": str(tmp_path / "index.sqlite"), "trace": []}
    SceneCache(bundle["database"], pipeline.signature).save([scene], np.asarray([[1.0, 0.0]]))
    return pipeline, bundle, calls


def draft(text="The person is handling the object."):
    return {"answer": text, "supporting_scenes": [0], "inference": "", "limitations": []}


def test_open_answer_and_verifier_have_dedicated_choice_free_schema_and_task(tmp_path):
    pipeline, bundle, calls = run_open(tmp_path, [draft()])
    list(pipeline.qa(bundle, "What is the person doing?", []))
    assert len(calls) == 5
    assert all("Choice selection" != s["stage"] for s in bundle["trace"])
    answer_call = next(s for s in bundle["trace"] if s["stage"] == "Answer")
    answer_prompt = answer_call["attempts"][0]["prompt"]
    schema = json.loads(answer_prompt.split("The required JSON schema is:\n", 1)[1])
    assert set(schema["properties"]) == {
        "answer",
        "supporting_scenes",
        "limitations",
        "inference",
    }
    assert "First compare EVERY supplied choice" not in answer_prompt
    assert "visual choice reader selected" not in answer_prompt
    assert "Response mode: open-ended" in answer_prompt
    assert answer_call["qa_mode"] == "open_ended"
    assert "controller_transform" not in answer_call
    verifier = next(s for s in bundle["trace"] if s["stage"] == "Verifier")
    review_prompt = verifier["attempts"][0]["prompt"]
    assert "The draft selects EXACTLY" not in review_prompt
    review_draft = json.loads(review_prompt.split("\nDraft: ", 1)[1].split("\n", 1)[0])
    assert "selected_choice" not in review_draft
    assert "option_comparisons" not in review_draft
    assert bundle["answer"]["selected_option"] is None
    assert bundle["answer"]["selected_choice"] is None
    assert bundle["answer"]["option_comparisons"] == []


@pytest.mark.parametrize("placeholder", ["None", " null ", "N/A.", "   "])
def test_placeholder_is_repaired_to_real_open_answer_with_raw_failure_retained(
    tmp_path, placeholder
):
    pipeline, bundle, _ = run_open(tmp_path, [draft(placeholder), draft()])
    list(pipeline.qa(bundle, "What is the person doing?", []))
    entry = next(s for s in bundle["trace"] if s["stage"] == "Answer")
    assert len(entry["attempts"]) == 2
    first, second = entry["attempts"]
    assert json.loads(first["raw_response"])["answer"] == placeholder
    assert "validation_error" in first
    assert "substantive" in second["prompt"]
    assert bundle["answer"]["answer"] == draft()["answer"]


def test_repeated_placeholder_is_not_a_completed_answer(tmp_path):
    pipeline, bundle, _ = run_open(tmp_path, [draft("None"), draft("None")])
    with pytest.raises(ValueError, match="Answer failed validation after two attempts"):
        list(pipeline.qa(bundle, "What is the person doing?", []))
    assert "answer" not in bundle
    assert bundle["trace"][-1]["status"] == "failed"
    assert len(bundle["trace"][-1]["attempts"]) == 2


def test_specific_evidence_limitation_is_valid_not_a_placeholder(tmp_path):
    value = draft("The person handles the object, but these frames do not establish why.")
    value["limitations"] = ["The motive is not visible in the sampled interaction."]
    pipeline, bundle, _ = run_open(tmp_path, [value])
    list(pipeline.qa(bundle, "Why does the person handle the object?", []))
    assert bundle["status"] == "verified"
    assert bundle["answer"]["limitations"] == value["limitations"]


@pytest.mark.parametrize("final_accepts", [True, False])
def test_review_can_request_one_correction_without_forcing_acceptance(tmp_path, final_accepts):
    original = draft("The person has completed the task.")
    corrected = draft("The person handles the object; completion is not visible.")
    corrected["limitations"] = ["The ending is not shown."]
    first_review = {
        "accepted": False,
        "issues": ["Completion was not observed."],
        "needs_revision": True,
    }
    final_review = {
        "accepted": final_accepts,
        "issues": [] if final_accepts else ["The purpose remains unclear."],
        "needs_revision": not final_accepts,
    }
    pipeline, bundle, calls = run_open(
        tmp_path,
        [original],
        first_review,
        [corrected, final_review],
    )
    list(pipeline.qa(bundle, "What is happening?", []))
    assert len(calls) == 7  # plan, tools, inspection, answer/review, ONE correction/review
    answers = [s for s in bundle["trace"] if s["stage"] == "Answer"]
    reviews = [s for s in bundle["trace"] if s["stage"] == "Verifier"]
    assert len(answers) == len(reviews) == 2
    assert answers[0]["output"]["answer"] == original["answer"]
    assert answers[1]["output"]["answer"] == corrected["answer"]
    assert answers[1]["revision"] == reviews[1]["revision"] == 1
    assert reviews[1]["max_answer_revisions"] == 1
    assert "Completion was not observed." in answers[1]["attempts"][0]["prompt"]
    assert all(s["attempts"][0]["images"] == ["fixture.jpg"] for s in answers + reviews)
    assert bundle["answer"]["answer"] == corrected["answer"]
    assert bundle["verification"]["accepted"] is final_accepts
    assert bundle["status"] == ("verified" if final_accepts else "unverified")


def test_rejection_without_revision_request_does_not_start_an_acceptance_loop(tmp_path):
    review = {"accepted": False, "issues": ["The needed event was not sampled."]}
    pipeline, bundle, calls = run_open(tmp_path, [draft()], review)
    list(pipeline.qa(bundle, "What is happening?", []))
    assert len(calls) == 5
    assert bundle["status"] == "unverified"
