"""Summarize saved hosted smoke artifacts offline; never call models or change artifacts.

Usage: python scripts/summarize_prompt_evaluation.py RUN_DIRECTORY [RUN_DIRECTORY ...]
JSON is printed to stdout. Prices are fixed evaluation assumptions, not fetched rates
or an invoice. A passed verifier is not an independently graded correct answer.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# Standard USD per million input, cached input, and output tokens for these minis.
MODEL_RATES = {
    "gpt-4.1-mini": (0.40, 0.10, 1.60),
    "gpt-5.4-mini": (0.75, 0.075, 4.50),
}


def read_artifact(path: Path, expected_type: type, errors: list[str]):
    try:
        value = json.loads(path.read_text())
        if not isinstance(value, expected_type):
            raise ValueError(f"expected {expected_type.__name__}")
        return value
    except (OSError, ValueError) as exc:
        errors.append(f"{path.name}: {exc}")
        return expected_type()


def records(value):
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def token_count(value):
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def usage_summary(trace):
    calls = [
        call
        for stage in trace
        if stage.get("stage") == "Hosted inference usage"
        for call in records(stage.get("calls"))
    ]
    totals = dict(input_tokens=0, cached_input_tokens=0, output_tokens=0, reasoning_tokens=0)
    attempts = missing_usage = unknown_rate = 0
    observed_cost = 0.0
    models = set()
    for call in calls:
        for attempt in records(call.get("attempts")):
            if attempt.get("request_attempted") is False:
                continue
            attempts += 1
            model = attempt.get("reported_model") or call.get("requested_model")
            if isinstance(model, str):
                models.add(model)
            usage = attempt.get("usage")
            if not isinstance(usage, dict):
                missing_usage += 1
                continue
            incoming = token_count(usage.get("prompt_tokens"))
            outgoing = token_count(usage.get("completion_tokens"))
            prompt_details = usage.get("prompt_tokens_details") or {}
            completion_details = usage.get("completion_tokens_details") or {}
            if not isinstance(prompt_details, dict) or not isinstance(completion_details, dict):
                missing_usage += 1
                continue
            cached = token_count(prompt_details.get("cached_tokens", 0))
            reasoning = token_count(completion_details.get("reasoning_tokens", 0))
            if None in (incoming, outgoing, cached, reasoning) or cached > incoming:
                missing_usage += 1
                continue
            totals["input_tokens"] += incoming
            totals["cached_input_tokens"] += cached
            totals["output_tokens"] += outgoing
            totals["reasoning_tokens"] += reasoning
            rates = next(
                (
                    rates
                    for family, rates in MODEL_RATES.items()
                    if model == family
                    or (isinstance(model, str) and model.startswith(family + "-"))
                ),
                None,
            )
            if rates is None:
                unknown_rate += 1
            else:
                # Reasoning tokens are already included in completion_tokens.
                observed_cost += (
                    (incoming - cached) * rates[0] + cached * rates[1] + outgoing * rates[2]
                ) / 1_000_000
    complete = attempts > 0 and missing_usage == 0 and unknown_rate == 0
    return {
        "models": sorted(models),
        "api_calls": len(calls),
        "request_attempts": attempts,
        "transport_retries": sum(
            max(
                0,
                len(
                    [
                        a
                        for a in records(call.get("attempts"))
                        if a.get("request_attempted") is not False
                    ]
                )
                - 1,
            )
            for call in calls
        ),
        **totals,
        "attempts_without_complete_usage": missing_usage,
        "attempts_without_known_rates": unknown_rate,
        "estimated_cost_usd": round(observed_cost, 9) if complete else None,
        "cost_of_known_reported_usage_usd": round(observed_cost, 9),
        "cost_estimate_complete": complete,
        "allowance": [
            stage.get("allowance")
            for stage in trace
            if stage.get("stage") == "Hosted inference usage"
        ],
    }


def summarize_run(directory: Path):
    directory = directory.resolve()
    errors = []
    report = read_artifact(directory / "report.json", dict, errors)
    raw_trace = read_artifact(directory / "raw-trace.json", list, errors)
    trace = records(raw_trace)
    if len(trace) != len(raw_trace):
        errors.append("raw-trace.json: non-object stage records were omitted")
    stages = []
    retrievals = []
    for position, entry in enumerate(trace):
        if entry.get("stage") == "Hosted inference usage":
            continue
        attempts = records(entry.get("attempts"))
        lengths = [len(a["prompt"]) for a in attempts if isinstance(a.get("prompt"), str)]
        # Inspection flattens several independent scene calls into attempts.
        repairs = entry.get("repair_count")
        if repairs is None:
            repairs = (
                sum(max(0, len(records(c.get("attempts"))) - 1) for c in records(entry["calls"]))
                if "calls" in entry
                else max(0, len(attempts) - 1)
            )
        stages.append(
            {
                "trace_index": position,
                "stage": entry.get("stage"),
                "status": entry.get("status"),
                "model": entry.get("model"),
                "prompt_profile": entry.get("prompt_profile"),
                "prompt_version": entry.get("prompt_version"),
                "revision": entry.get("revision", 0),
                "prompt_characters": lengths,
                "total_prompt_characters": sum(lengths),
                "structured_attempts": len(attempts),
                "schema_repairs": repairs,
                "latency_ms": entry.get("latency_ms"),
            }
        )
        if entry.get("stage") == "Retrieval":
            output = entry.get("output") or {}
            retrievals.append(
                [
                    {key: scene.get(key) for key in ("index", "start", "end", "frame_times")}
                    for scene in records(output.get("selected"))
                ]
            )
    answers = [e for e in trace if e.get("stage") == "Answer"]
    reviews = [e for e in trace if e.get("stage") == "Verifier"]
    elapsed = re.search(r"([\d.]+)s\s+total\b", str(report.get("status", "")))
    return {
        "directory": str(directory),
        "artifact_errors": errors,
        "report_result": report.get("result"),
        "report_error": report.get("error"),
        "qa_completed": report.get("qa_completed"),
        "qa_state": report.get("qa_state"),
        "trace_captured": report.get("trace_captured"),
        "ui_checks_passed": report.get("ui_checks_passed"),
        "question": report.get("question"),
        "options": report.get("options"),
        "status": report.get("status"),
        "total_elapsed_seconds": float(elapsed.group(1)) if elapsed else None,
        "cache_hits": [e.get("cache_hit") for e in trace if e.get("stage") == "Video cache"],
        "stages": stages,
        "total_prompt_characters": sum(s["total_prompt_characters"] for s in stages),
        "schema_repairs": sum(s["schema_repairs"] for s in stages),
        "retrievals": retrievals,
        "refinements": [e.get("refinement") for e in trace if e.get("refinement")],
        "answers": [e.get("output") for e in answers],
        "reviews": [e.get("output") for e in reviews],
        "final_answer": answers[-1].get("output") if answers else None,
        "final_review": reviews[-1].get("output") if reviews else None,
        "displayed_answer": report.get("answer"),
        "answer_correct": None,
        "frame_grounding": None,
        "usage": usage_summary(trace),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            {
                "note": "Verifier acceptance is not correctness or frame grounding.",
                "cost_note": "Fixed evaluation rates; reported usage estimate, not an invoice.",
                "rates_usd_per_million": {
                    model: dict(zip(("input", "cached_input", "output"), rates, strict=True))
                    for model, rates in MODEL_RATES.items()
                },
                "runs": [summarize_run(directory) for directory in args.directories],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
