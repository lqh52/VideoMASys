"""Evaluate new MC/open-ended cases without changing the production QA pipeline.

Manifest shape: {"cases": [{"id": "case-open", "video": "/path/video.mp4",
"question": "...", "options": [], "reference": {"answer": "for manual review"}}]}.
MC cases require reference.correct_option (zero-based). Reference metadata stays
outside inference requests. Open-ended answers require human review and are never
silently counted as correct or wrong against a multiple-choice annotation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import tarfile
from datetime import UTC, datetime
from pathlib import Path

try:  # Direct CLI execution and namespace-package imports in focused tests.
    from evaluate_zerogpu_library import (
        OPTIONAL_SOURCE_FILES,
        SOURCE_FILES,
        load_local_source,
        local_pipeline,
        run_local_example,
        save_json,
    )
except ModuleNotFoundError:
    from scripts.evaluate_zerogpu_library import (
        OPTIONAL_SOURCE_FILES,
        SOURCE_FILES,
        load_local_source,
        local_pipeline,
        run_local_example,
        save_json,
    )


def load_cases(path: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads(path.read_text())
    rows = manifest.get("cases")
    if not isinstance(rows, list) or not 1 <= len(rows) <= 16:
        raise ValueError("The evaluation manifest must contain 1–16 cases.")
    cases, seen = [], set()
    for number, row in enumerate(rows, 1):
        case = copy.deepcopy(row)
        case_id = case.get("id", "")
        if (
            not isinstance(case_id, str)
            or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,99}", case_id)
            or case_id in seen
        ):
            raise ValueError("Case IDs must be unique lowercase letters/digits/hyphen slugs.")
        seen.add(case_id)
        video = Path(case["video"])
        if not video.is_absolute():
            video = path.parent / video
        if not video.is_file():
            raise ValueError(f"Missing case video: {video}")
        question, options = case.get("question"), case.get("options", [])
        if not isinstance(question, str) or not 2 <= len(question.strip()) <= 2000:
            raise ValueError(f"Invalid question for {case_id}.")
        if (
            not isinstance(options, list)
            or (options and not 2 <= len(options) <= 5)
            or any(not isinstance(s, str) or not 1 <= len(s.strip()) <= 300 for s in options)
            or len(set(options)) != len(options)
        ):
            raise ValueError(f"Options must be empty or 2–5 unique strings for {case_id}.")
        reference = case.get("reference", {})
        if not isinstance(reference, dict):
            raise ValueError(f"Invalid reference metadata for {case_id}.")
        if options:
            correct = reference.get("correct_option")
            if type(correct) is not int or not 0 <= correct < len(options):
                raise ValueError(f"MC reference.correct_option must be valid for {case_id}.")
            if "correct_choice" in reference and reference["correct_choice"] != options[correct]:
                raise ValueError(f"MC reference choice/index disagree for {case_id}.")
        case.update(
            number=number,
            video=str(video.resolve()),
            question=question,
            options=options,
            reference=reference,
            evaluation_mode="multiple_choice" if options else "open_ended",
        )
        with video.open("rb") as source:
            case["video_sha256"] = hashlib.file_digest(source, "sha256").hexdigest()
        cases.append(case)
    return manifest, cases


def holdout_result(record: dict) -> dict:
    case, summary = record["example"], record["summary"]
    trace = record.get("bundle", {}).get("trace", [])
    open_ended = not case["options"]
    choice_steps = [s for s in trace if s.get("stage") == "Choice selection"]
    answer_steps = [s for s in trace if s.get("stage") == "Answer"]
    final_answer_step = answer_steps[-1] if answer_steps else {}
    return {
        "id": case["id"],
        "mode": "open_ended" if open_ended else "multiple_choice",
        "completed": bool(summary.get("completed") and not record.get("error")),
        "answer": summary.get("answer", {}),
        "selected_option": summary.get("selected_option"),
        "selected_choice": summary.get("selected_choice"),
        "exact_correct": None if open_ended else summary.get("correct"),
        "manual_review": "required" if open_ended else "not an automatic grounding check",
        "accepted": summary.get("accepted"),
        "verifier": summary.get("verdict", {}),
        "choice_selection_calls": len(choice_steps),
        "open_ended_skipped_choice_selection": not choice_steps if open_ended else None,
        "raw_answer_attempts": final_answer_step.get("attempts", []),
        "retrieved_scene_ids": summary.get("retrieved_scene_ids", []),
        "refinements": [s["refinement"] for s in trace if s.get("refinement")],
        "cache_hit": summary.get("cache_hit"),
        "repair_count": summary.get("repair_count", 0),
        "wall_seconds": record.get("wall_seconds"),
        "error": record.get("error"),
    }


def summarize_records(records: list[dict]) -> dict:
    results = [holdout_result(record) for record in records]
    mc = [r for r in results if r["mode"] == "multiple_choice"]
    opened = [r for r in results if r["mode"] == "open_ended"]
    correct = sum(r["completed"] and r["exact_correct"] is True for r in mc)
    return {
        "attempted": len(results),
        "completed": sum(r["completed"] for r in results),
        "multiple_choice": {
            "attempted": len(mc),
            "completed": sum(r["completed"] for r in mc),
            "exact_correct": correct,
            "accuracy_over_mc_attempted": correct / len(mc) if mc else None,
        },
        "open_ended": {
            "attempted": len(opened),
            "completed": sum(r["completed"] for r in opened),
            "manual_review_required": len(opened),
            "unexpected_choice_selection_cases": [
                r["id"] for r in opened if r["choice_selection_calls"]
            ],
        },
        "wall_seconds": round(sum(r.get("wall_seconds") or 0 for r in results), 3),
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/evaluation/holdout-qa"))
    parser.add_argument("--source-archive", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--vlm-url", default="http://127.0.0.1:8205/v1")
    parser.add_argument("--embedding-url", default="http://127.0.0.1:8206/v1")
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--embedding-id", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    source_manifest, cases = load_cases(args.manifest)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    directory = args.output_dir / stamp
    directory.mkdir(parents=True, exist_ok=False)
    if not args.source_archive:
        args.source_archive = directory / "source_snapshot.tar.gz"
        with tarfile.open(args.source_archive, "w:gz") as target:
            for path in {**SOURCE_FILES, **OPTIONAL_SOURCE_FILES}.values():
                if Path(path).is_file():
                    target.add(path, arcname=path)
    core, system_prompt, hashes = load_local_source(args.source_archive)
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "input_manifest": str(args.manifest),
        "source_manifest": source_manifest,
        "cases": cases,
        "source_archive": str(args.source_archive),
        "source_sha256": hashes,
        "model_id": args.model_id,
        "embedding_id": args.embedding_id,
        "system_prompt": system_prompt,
        "dry_run": args.dry_run,
        "reference_policy": "Reference metadata is never passed to QA/model functions.",
        "latency_note": "Local vLLM wall-clock timings, not hosted ZeroGPU queue latency.",
    }
    save_json(directory / "manifest.json", manifest)
    print(f"Artifacts: {directory}", flush=True)
    if args.dry_run:
        print(f"Validated {len(cases)} cases; no inference or network requests.")
        return
    pipeline, http = local_pipeline(core, system_prompt, args)
    manifest["generation_settings"] = pipeline.generation_settings
    save_json(directory / "manifest.json", manifest)
    records = []
    try:
        for case in cases:
            print(f"Running {case['id']} ({case['evaluation_mode']}) …", flush=True)
            record = run_local_example(
                core, pipeline, case, directory, args.cache_root or directory / "cache"
            )
            record["holdout_evaluation"] = holdout_result(record)
            save_json(directory / f"{case['number']:02d}-{case['id']}.json", record)
            records.append(record)
            summary = summarize_records(records)
            save_json(directory / "summary.json", summary)
            print(
                json.dumps(
                    {
                        k: v
                        for k, v in summary["results"][-1].items()
                        if k not in {"raw_answer_attempts", "answer", "refinements", "verifier"}
                    }
                ),
                flush=True,
            )
    finally:
        http.close()
    if any(not holdout_result(record)["completed"] for record in records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
