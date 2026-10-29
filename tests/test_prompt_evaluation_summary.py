"""Offline artifact-accounting checks; these fixtures establish no model accuracy."""

import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "prompt_evaluation_summary",
    Path(__file__).resolve().parents[1] / "scripts/summarize_prompt_evaluation.py",
)
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


def make_run(tmp_path, report, trace):
    (tmp_path / "report.json").write_text(json.dumps(report))
    (tmp_path / "raw-trace.json").write_text(json.dumps(trace))
    return tmp_path


def usage_stage(model="gpt-5.4-mini-2026-03-17", attempts=None):
    return {
        "stage": "Hosted inference usage",
        "calls": [
            {
                "requested_model": model,
                "attempts": attempts
                if attempts is not None
                else [
                    {
                        "request_attempted": True,
                        "reported_model": model,
                        "usage": {
                            "prompt_tokens": 1000,
                            "completion_tokens": 100,
                            "prompt_tokens_details": {"cached_tokens": 400},
                            "completion_tokens_details": {"reasoning_tokens": 30},
                        },
                    }
                ],
            }
        ],
    }


@pytest.mark.parametrize(
    "model,cost",
    [
        ("gpt-5.4-mini-2026-03-17", 0.000930),
        ("gpt-4.1-mini-2025-04-14", 0.000440),
    ],
)
def test_cost_discounts_cached_input_and_does_not_double_bill_reasoning(model, cost):
    result = summary.usage_summary([usage_stage(model)])
    assert result["estimated_cost_usd"] == pytest.approx(cost)
    assert result["reasoning_tokens"] == 30
    assert result["output_tokens"] == 100
    assert result["api_calls"] == result["request_attempts"] == 1


def test_failed_attempt_usage_is_retained_and_unknown_usage_is_not_free():
    attempts = usage_stage()["calls"][0]["attempts"]
    attempts[0]["http_status"] = 500
    attempts.append({"request_attempted": True, "http_status": 503})
    attempts.append({"request_attempted": False})
    result = summary.usage_summary([usage_stage(attempts=attempts)])
    assert result["api_calls"] == 1
    assert result["request_attempts"] == 2
    assert result["transport_retries"] == 1
    assert result["input_tokens"] == 1000
    assert result["attempts_without_complete_usage"] == 1
    assert result["estimated_cost_usd"] is None
    assert result["cost_of_known_reported_usage_usd"] == pytest.approx(0.000930)


def test_unknown_model_does_not_inherit_mini_rates():
    result = summary.usage_summary([usage_stage("gpt-5.4")])
    assert result["attempts_without_known_rates"] == 1
    assert result["estimated_cost_usd"] is None


def test_retains_revisions_failures_and_independent_grading(tmp_path):
    directory = make_run(
        tmp_path,
        {
            "result": "failed",
            "error": "UI check failed",
            "qa_completed": True,
            "question": "What happens?",
            "options": [],
            "status": "0.0s\nAnalysis complete\nModel check passed · 35.7s total",
        },
        [
            {
                "stage": "Evidence inspection",
                "repair_count": 0,
                "attempts": [{"prompt": "a"}, {"prompt": "bb"}, {"prompt": "ccc"}],
            },
            {
                "stage": "Answer",
                "attempts": [{"prompt": "bad"}, {"prompt": "fixed"}],
                "output": {"answer": "Original unsupported draft"},
            },
            {"stage": "Verifier", "output": {"accepted": False}},
            {"stage": "Answer", "revision": 1, "output": {"answer": "Qualified draft"}},
            {"stage": "Verifier", "revision": 1, "output": {"accepted": True}},
            usage_stage(),
        ],
    )
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    result = summary.summarize_run(directory)
    assert result["report_result"] == "failed"
    assert result["report_error"] == "UI check failed"
    assert result["total_elapsed_seconds"] == 35.7
    assert result["schema_repairs"] == 1  # Three independent inspections are not repairs.
    assert result["stages"][0]["total_prompt_characters"] == 6
    assert result["final_answer"] == {"answer": "Qualified draft"}
    assert len(result["answers"]) == len(result["reviews"]) == 2
    assert result["answer_correct"] is result["frame_grounding"] is None
    assert before == {p.name: p.read_bytes() for p in directory.iterdir()}


def test_cli_keeps_missing_artifacts_next_to_valid_runs(tmp_path, capsys):
    directory = make_run(tmp_path, {"result": "failed"}, [])
    summary.main([str(directory), str(directory / "missing")])
    result = json.loads(capsys.readouterr().out)
    assert len(result["runs"]) == 2
    assert result["runs"][0]["report_result"] == "failed"
    assert len(result["runs"][1]["artifact_errors"]) == 2
    assert result["runs"][1]["usage"]["estimated_cost_usd"] is None
