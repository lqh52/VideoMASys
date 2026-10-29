"""Hold-out tooling tests: isolated deterministic fixtures, never actual inference."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from videomasys.zerogpu.core import LivePipeline, SceneCache

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_holdout_qa.py"
spec = importlib.util.spec_from_file_location("holdout_evaluation", SCRIPT)
holdout = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(SCRIPT.parent))
try:
    spec.loader.exec_module(holdout)
finally:
    sys.path.remove(str(SCRIPT.parent))


def test_manifest_keeps_original_question_and_scorer_only_references(tmp_path):
    (tmp_path / "fixture.mp4").write_bytes(b"fixture; never decoded")
    manifest = {
        "cases": [
            {
                "id": "open",
                "video": "fixture.mp4",
                "question": " What happens? ",
                "options": [],
                "reference": {"answer": "SCORER_ONLY_SENTINEL"},
            },
            {
                "id": "mc",
                "video": "fixture.mp4",
                "question": " What happens? ",
                "options": ["One", "Two"],
                "reference": {"correct_option": 1},
            },
        ]
    }
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(manifest))
    original, cases = holdout.load_cases(path)
    assert original == manifest
    assert cases[0]["question"] == " What happens? "
    assert cases[0]["options"] == []
    assert cases[0]["reference"]["answer"] == "SCORER_ONLY_SENTINEL"
    assert cases[0]["video_sha256"] == cases[1]["video_sha256"]
    manifest["cases"][1]["reference"]["correct_option"] = True
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="correct_option"):
        holdout.load_cases(path)


@pytest.mark.parametrize("fail_answer", [False, True])
def test_real_open_pipeline_skips_choice_selection_and_keeps_failed_output(tmp_path, fail_answer):
    scenes = [
        {
            "index": 0,
            "start": 0,
            "end": 20,
            "frames": ["fixture.jpg"],
            "caption": "An isolated fixture event.",
        }
    ]
    plan = {
        "scope": "local_event",
        "objective": "Observe the event",
        "evidence_needs": [
            {"id": "anchor", "description": "Visible event", "temporal_role": "anchor"}
        ],
        "max_intervals": 1,
        "max_inspected_seconds": 20,
        "stop_condition": "Visible observation available",
        "reasoning": "Inspect the event",
    }
    program = {
        "rationale": "Locate the event",
        "max_intervals": 1,
        "max_inspected_seconds": 20,
        "locate": [
            {"id": "l1", "query": "Visible event", "limit": 2, "evidence_need_ids": ["anchor"]}
        ],
        "relate": [],
        "inspect": {"sources": ["l1"], "evidence_need_ids": ["anchor"]},
    }
    responses = [
        plan,
        program,
        {"scene": 0, "observation": "A visible fixture event.", "relevant": True},
    ]
    if fail_answer:
        responses += ['{"answer":"TRUNCATED_OPEN_DRAFT', '{"answer":"SECOND_OPEN_DRAFT']
    else:
        responses += [
            {
                "answer": "The person performs the observed event.",
                "selected_choice": None,
                "option_comparisons": [],
                "supporting_scenes": [0],
            },
            {"accepted": True, "issues": []},
        ]
    iterator, prompts = iter(responses), []

    def generate(prompt, *_args):
        prompts.append(prompt)
        assert "SCORER_ONLY_SENTINEL" not in prompt
        value = next(iterator)
        return value if isinstance(value, str) else json.dumps(value)

    pipeline = LivePipeline(generate, lambda texts: np.asarray([[1.0, 0.0]]), "fixture", "fixture")

    def forbidden_choice_selection(*_args):
        pytest.fail("An open-ended question must not invoke Choice selection.")

    pipeline.choose_option = forbidden_choice_selection
    database = str(tmp_path / "index.sqlite")
    SceneCache(database, pipeline.signature).save(scenes, np.asarray([[1.0, 0.0]]))
    core = SimpleNamespace(
        SceneCache=SceneCache,
        prepare_video=lambda *_args: {
            "trace": [],
            "scenes": scenes,
            "duration": 20,
            "database": database,
            "cached": False,
        },
    )
    case = {
        "id": "open-fixture",
        "number": 1,
        "video": "fixture.mp4",
        "question": "What happens?",
        "options": [],
        "reference": {"answer": "SCORER_ONLY_SENTINEL"},
    }
    record = holdout.run_local_example(core, pipeline, case, tmp_path, tmp_path / "cache")
    result = holdout.holdout_result(record)
    assert prompts
    assert result["choice_selection_calls"] == 0
    assert result["open_ended_skipped_choice_selection"] is True
    assert result["exact_correct"] is None
    assert result["manual_review"] == "required"
    assert result["completed"] is not fail_answer
    saved = json.loads((tmp_path / "01-open-fixture.json").read_text())
    if fail_answer:
        assert len(result["raw_answer_attempts"]) == 2
        assert result["raw_answer_attempts"][0]["raw_response"].endswith("TRUNCATED_OPEN_DRAFT")
        assert "SECOND_OPEN_DRAFT" in json.dumps(saved)
    else:
        assert result["answer"]["answer"] == "The person performs the observed event."
        assert saved["bundle"]["answer"]["selected_option"] is None
    summary = holdout.summarize_records([record])
    assert summary["multiple_choice"]["accuracy_over_mc_attempted"] is None
    assert summary["multiple_choice"]["attempted"] == 0
    assert summary["open_ended"]["manual_review_required"] == 1


def test_mixed_summary_does_not_score_open_ended_as_wrong():
    base = {"bundle": {"trace": []}, "wall_seconds": 1, "error": None}
    opened = {
        **base,
        "example": {"id": "open", "options": []},
        "summary": {"completed": True, "correct": None, "answer": {"answer": "Open"}},
    }
    mc = {
        **base,
        "example": {"id": "mc", "options": ["One", "Two"]},
        "summary": {"completed": True, "correct": True, "selected_option": 1},
    }
    summary = holdout.summarize_records([opened, mc])
    assert summary["multiple_choice"]["accuracy_over_mc_attempted"] == 1.0
    assert summary["multiple_choice"]["attempted"] == 1
    assert summary["open_ended"]["attempted"] == 1
