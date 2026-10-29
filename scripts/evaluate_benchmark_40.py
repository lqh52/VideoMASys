"""Frozen, once-only 20 NExT-QA + 20 EgoSchema end-to-end evaluation.

Default is read-only preflight. --run admits paid inference with a $4 conservative
runner budget. --resume continues only never-started cases, not failures. Dataset
references are used only by the scorer; no application behavior is patched.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]

from benchmark_accounting import BenchmarkVLM, Budget  # noqa: E402
from evaluate_openai_prompts import (  # noqa: E402
    EvaluationError,
    read_owner_key,
    safe_failure,
    selected_model,
    write_json,
)

from videomasys.zerogpu.core import prepare_video  # noqa: E402
from videomasys.zerogpu.remote_runtime import (  # noqa: E402
    EMBEDDING_ID,
    GPT41_MINI_MODEL,
    CPUEmbedder,
    RemotePipeline,
)
from videomasys.zerogpu.structured_prompts import PROMPT_VERSION  # noqa: E402
from videomasys.zerogpu.usage import UsageGuard  # noqa: E402

ROOT = REPO / "artifacts/benchmark_40_2026-09-08"
VERSION = "structured-evidence-v3"
LIMITS = {"max_runs": 40, "max_requests": 1600, "max_requests_per_run": 40}
MAX_USD = 4.0
HELPERS = (
    "scripts/evaluate_benchmark_40.py", "scripts/prepare_benchmark_40.py",
    "scripts/benchmark_accounting.py", "scripts/evaluate_openai_prompts.py",
    "scripts/evaluate_prompt_tuning.py", "pyproject.toml",
)


def utc_now():
    return datetime.now(UTC).isoformat()


def file_sha(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def sources():
    paths = sorted((REPO / "src/videomasys").rglob("*.py"))
    paths += [REPO / name for name in HELPERS]
    if any(path.is_symlink() or not path.is_file() for path in paths):
        raise EvaluationError("Evaluation sources must be regular files.")
    return {str(path.relative_to(REPO)): file_sha(path) for path in paths}


def load_manifest():
    manifest = json.loads((ROOT / "manifest.json").read_text())
    cases = manifest["cases"]
    if Counter(case["dataset"] for case in cases) != {"nextqa": 20, "egoschema": 20}:
        raise EvaluationError("This evaluation requires exactly 20 cases per dataset.")
    ids = [case["id"] for case in cases]
    if len(set(ids)) != 40 or any(not re.fullmatch(r"[a-z0-9_-]+", name) for name in ids):
        raise EvaluationError("Invalid or duplicate case IDs.")
    for case in cases:
        options, correct = case["options"], case["reference"]["correct_option"]
        if (
            len(options) != 5 or len(set(options)) != 5
            or any(not isinstance(value, str) or not value.strip() or len(value) > 300
                   for value in options)
            or type(correct) is not int or not 0 <= correct < 5
            or options[correct] != case["reference"]["correct_choice"]
            or not 2 <= len(case["question"].strip()) <= 2000
            or not Path(case["video"]).is_file()
        ):
            raise EvaluationError("Invalid benchmark input or scoring reference.")
    return manifest


def atomic_json(path, value, key):
    # Only this runner's regenerable summary/checkpoint files use replacement.
    temporary = path.with_suffix(path.suffix + ".tmp")
    serialized = json.dumps(value, ensure_ascii=False, indent=2)
    if key:
        serialized = serialized.replace(key, "[REDACTED]")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(serialized + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


class LoggedVLM(BenchmarkVLM):
    """Persist each real call before continuing; never includes authorization headers."""

    def __init__(self, *args, output, **kwargs):
        super().__init__(*args, **kwargs)
        self.output = output
        self.sequence = 0

    def _generate(self, prompt, paths, tokens, *, schema=None):
        self.sequence += 1
        record = {"prompt": prompt, "images": list(paths), "schema": schema,
                  "max_output_tokens": tokens, "started_at": utc_now()}
        write_json(self.output / f"call-{self.sequence:03d}-started.json", record, self._token)
        try:
            value = super()._generate(prompt, paths, tokens, schema=schema)
            record["raw_output"] = value
            return value
        finally:
            record.update(finished_at=utc_now(), transport=self.calls[-1] if self.calls else {})
            write_json(self.output / f"call-{self.sequence:03d}.json", record, self._token)


def tokens_from_calls(calls):
    attempts = [a for call in calls for a in call.get("attempts", [])
                if a.get("request_attempted")]
    return {
        "http_attempts": len(attempts),
        "unknown_usage_attempts": sum(not a.get("usage") for a in attempts),
        "prompt_tokens": sum(a.get("usage", {}).get("prompt_tokens", 0) for a in attempts),
        "completion_tokens": sum(a.get("usage", {}).get("completion_tokens", 0) for a in attempts),
        "cached_tokens": sum(a.get("usage", {}).get("prompt_tokens_details", {}).get(
            "cached_tokens", 0) for a in attempts),
        "http_statuses": dict(Counter(str(a.get("http_status", "unknown")) for a in attempts)),
    }


def run_case(case, key, guard, budget, embed, frozen):
    if sources() != frozen:
        raise EvaluationError("Sources changed; refusing a mixed-version benchmark.")
    output = ROOT / "cases" / case["id"]
    output.mkdir(mode=0o700)  # Exclusive admission; never repeat a started case.
    write_json(output / "started.json", {"id": case["id"], "started_at": utc_now()}, key)
    started, phase, failure = time.monotonic(), "Preparation", None
    prepared_seconds = index_seconds = qa_seconds = None
    qa_started = None
    index_call_count = 0
    bundle = {"trace": [], "question": case["question"], "options": case["options"]}
    try:
        bundle.update(prepare_video(Path(case["video"]), output / "cache"))
        if bundle["key"] != case["video_sha256"]:
            raise EvaluationError("Video bytes differ from the frozen manifest.")
        prepared_seconds = time.monotonic() - started
        factory = partial(LoggedVLM, budget=budget, case_id=case["id"], output=output)
        pipeline = RemotePipeline(provider="openai", token=key, guard=guard, embed=embed,
                                  prompt_profile="structured", vlm_factory=factory)
        with pipeline.analysis_session(bundle) as active:
            phase = "Indexing"
            index_started = time.monotonic()
            active.index(bundle)
            if bundle["cached"]:
                raise EvaluationError("This benchmark requires a fresh video cache.")
            active.save_index(bundle)
            index_seconds = time.monotonic() - index_started
            index_call_count = len(bundle["inference"]["calls"])
            bundle["trace"].append({
                "stage": "Video cache", "status": "complete", "cache_hit": False,
                "scene_count": len(bundle["scenes"]), "scenes": bundle["scenes"],
                "source": "fresh captions and CPU embeddings", "latency_ms": round(
                    index_seconds * 1000),
            })
            atomic_json(output / "checkpoint.json", bundle, key)
            phase = "QA"
            qa_started = time.monotonic()
            iterator = active.qa(bundle, case["question"], case["options"])
            try:
                for bundle in iterator:
                    atomic_json(output / "checkpoint.json", bundle, key)
            finally:
                iterator.close()
                qa_seconds = time.monotonic() - qa_started
    except Exception as exc:
        failure = safe_failure(exc)
        bundle["error"] = failure[1]
        if qa_started is not None:
            qa_seconds = time.monotonic() - qa_started
    elapsed = time.monotonic() - started
    completed = failure is None and "answer" in bundle and "verification" in bundle
    answer = bundle.get("answer", {})
    selected = answer.get("selected_option")
    trace = bundle["trace"]
    choice = next((s.get("output", {}).get("selected_choice") for s in trace
                   if s.get("stage") == "Choice selection"), None)
    calls = bundle.get("inference", {}).get("calls", [])
    if phase == "Indexing":
        index_call_count = len(calls)
    report = {
        "id": case["id"], "dataset": case["dataset"], "split": case["split"],
        "question_type": case.get("question_type"), "completed": completed,
        "selected_option": selected, "selected_choice": answer.get("selected_choice"),
        "correct_option": case["reference"]["correct_option"],
        "correct": completed and selected == case["reference"]["correct_option"],
        "provisional_choice_correct": choice == case["reference"]["correct_choice"],
        "answer": answer.get("answer"), "status": bundle.get("status"),
        "review_accepted": bundle.get("verification", {}).get("accepted"),
        "resolution": bundle.get("resolution"), "error": failure,
        "failure_phase": phase if failure else None,
        "last_stage": trace[-1].get("stage") if trace else phase,
        "duration_seconds": case["duration_seconds"], "cold_cache": not bundle.get("cached"),
        "preparation_seconds": prepared_seconds, "index_seconds": index_seconds,
        "qa_seconds": qa_seconds, "total_seconds": elapsed,
        "selected_scenes": [s["index"] for s in bundle.get("selected", [])],
        "selected_seconds": sum(s["end"] - s["start"] for s in bundle.get("selected", [])),
        "format_repairs": sum(
            s.get("repair_count", 0) if s.get("stage") == "Evidence inspection"
            else max(0, len(s.get("attempts", [])) - 1) for s in trace
        ),
        "answer_revisions": max(0, sum(s.get("stage") == "Answer" for s in trace) - 1),
        "usage": tokens_from_calls(calls),
        "index_usage": tokens_from_calls(calls[:index_call_count]),
        "qa_usage": tokens_from_calls(calls[index_call_count:]),
        "sources_unchanged": sources() == frozen, "finished_at": utc_now(),
    }
    write_json(output / "bundle.json", bundle, key)
    write_json(output / "report.json", report, key)
    print(json.dumps({key: report[key] for key in (
        "id", "completed", "correct", "status", "total_seconds", "error",
    )}), flush=True)
    return report


def timing(values):
    import numpy as np

    values = [v for v in values if v is not None]
    return ({"n": len(values), "mean": round(float(np.mean(values)), 3),
             "median": round(float(np.median(values)), 3),
             "p95": round(float(np.percentile(values, 95)), 3)} if values else None)


def record_interrupted_cases(cases, key):
    """Called only on resume, before any workers; never restart an admitted case."""
    for case in cases:
        output = ROOT / "cases" / case["id"]
        if not output.exists() or (output / "report.json").exists():
            continue
        if output.is_symlink() or not (output / "started.json").is_file():
            raise EvaluationError("An incomplete case directory has no admission record.")
        checkpoint = output / "checkpoint.json"
        bundle = json.loads(checkpoint.read_text()) if checkpoint.is_file() else {}
        # Only completed journal files are included; pending reservations remain
        # conservatively charged in the separate durable budget ledger.
        calls = [json.loads(path.read_text()).get("transport", {})
                 for path in sorted(output.glob("call-[0-9][0-9][0-9].json"))]
        answer = bundle.get("answer", {})
        report = {
            "id": case["id"], "dataset": case["dataset"], "split": case["split"],
            "completed": False, "correct": False, "selected_option": answer.get("selected_option"),
            "selected_choice": answer.get("selected_choice"), "answer": answer.get("answer"),
            "correct_option": case["reference"]["correct_option"],
            "provisional_choice_correct": False, "status": "interrupted",
            "review_accepted": False, "resolution": "unknown_outcome",
            "error": ["interrupted", "Previously started case; outcome incomplete, not rerun."],
            "failure_phase": "Interrupted", "last_stage": "Interrupted",
            "index_seconds": None, "qa_seconds": None, "total_seconds": None,
            "selected_seconds": None, "format_repairs": 0, "answer_revisions": 0,
            "usage": tokens_from_calls(calls), "finished_at": utc_now(),
            "usage_may_be_incomplete": True,
        }
        write_json(output / "report.json", report, key)


def summarize(manifest, budget=None, guard=None):
    reports = []
    for case in manifest["cases"]:
        path = ROOT / "cases" / case["id"] / "report.json"
        if path.is_file():
            reports.append(json.loads(path.read_text()))
    datasets = {}
    for dataset in ("nextqa", "egoschema"):
        rows = [r for r in reports if r["dataset"] == dataset]
        complete = [r for r in rows if r["completed"]]
        correct = sum(r["correct"] for r in rows)
        verified = [r for r in rows if r["status"] == "verified" and r["completed"]]
        datasets[dataset] = {
            "planned": 20, "attempted": len(rows), "completed": len(complete),
            "correct": correct, "accuracy_all_attempted": correct / len(rows) if rows else None,
            "accuracy_all_planned": correct / 20 if len(rows) == 20 else None,
            "accuracy_completed": correct / len(complete) if complete else None,
            "failures": len(rows) - len(complete),
            "abstentions": sum(r["selected_option"] is None for r in complete),
            "verified": len(verified), "verified_correct": sum(r["correct"] for r in verified),
            "qa_seconds_completed": timing([r["qa_seconds"] for r in complete]),
            "index_seconds": timing([r["index_seconds"] for r in rows]),
            "total_seconds": timing([r["total_seconds"] for r in rows]),
            "mean_selected_seconds": timing([r["selected_seconds"] for r in complete]),
            "http_attempts": sum(r["usage"]["http_attempts"] for r in rows),
            "prompt_tokens": sum(r["usage"]["prompt_tokens"] for r in rows),
            "completion_tokens": sum(r["usage"]["completion_tokens"] for r in rows),
            "format_repairs": sum(r["format_repairs"] for r in rows),
            "answer_revisions": sum(r["answer_revisions"] for r in rows),
            "failure_stages": dict(Counter(r["last_stage"] for r in rows if not r["completed"])),
        }
    return {
        "updated_at": utc_now(), "model": GPT41_MINI_MODEL, "prompt_version": VERSION,
        "planned": 40, "attempted": len(reports), "finished": len(reports) == 40,
        "reference_policy": "Labels used only for exact option-index scoring after inference.",
        "accuracy_policy": "Failures/abstentions count incorrect; review is not ground truth.",
        "latency_note": "Fresh local preparation/indexing; QA includes retrieval and review. "
                        "Concurrent local workers, not public website queue latency.",
        "datasets": datasets, "results": reports,
        "budget": budget.snapshot() if budget else None,
        "allowance": guard.snapshot() if guard else None,
    }


def run(manifest, *, workers, resume):
    frozen = sources()
    config_path = ROOT / "run-config.json"
    manifest_sha = file_sha(ROOT / "manifest.json")
    if config_path.exists():
        if not resume:
            raise EvaluationError("Batch started. Use --resume for never-started cases only.")
        config = json.loads(config_path.read_text())
        if config["source_sha256"] != frozen or config["manifest_sha256"] != manifest_sha:
            raise EvaluationError("Frozen sources or manifest changed; cannot resume this batch.")
        if config["workers"] != workers:
            raise EvaluationError("Resume must preserve the original worker count for timings.")
        for name in ("usage.sqlite3", "budget.sqlite3"):
            path = ROOT / name
            if path.is_symlink() or not path.is_file():
                raise EvaluationError("An existing ledger is missing or invalid; no reset allowed.")
    elif resume:
        raise EvaluationError("There is no started batch to resume.")
    elif any((ROOT / name).exists() for name in ("usage.sqlite3", "budget.sqlite3", "cases")):
        raise EvaluationError("State exists without its configuration; no implicit reset allowed.")
    key = read_owner_key()
    guard = UsageGuard(ROOT / "usage.sqlite3", **LIMITS)
    budget = Budget(ROOT / "budget.sqlite3", max_usd=MAX_USD)
    if not config_path.exists():
        config = {
            "started_at": utc_now(), "model": GPT41_MINI_MODEL, "prompt_version": VERSION,
            "embedding_id": EMBEDDING_ID, "manifest_sha256": manifest_sha,
            "source_sha256": frozen, "limits": LIMITS, "max_usd": MAX_USD,
            "workers": workers,
            "versions": {name: importlib.metadata.version(name) for name in
                         ("pydantic", "numpy", "fastembed", "httpx")},
        }
        write_json(config_path, config, key)
        write_json(ROOT / "source-snapshots.json",
                   {name: (REPO / name).read_text() for name in frozen}, key)
    (ROOT / "cases").mkdir(mode=0o700, exist_ok=True)
    if resume:
        record_interrupted_cases(manifest["cases"], key)
    pending = [case for case in manifest["cases"]
               if not (ROOT / "cases" / case["id"]).exists()]
    # Interleave datasets rather than running all of one before the other.
    grouped = [[case for case in pending if case["dataset"] == name]
               for name in ("nextqa", "egoschema")]
    pending = [group[index] for index in range(max(map(len, grouped), default=0))
               for group in grouped if index < len(group)]
    embed = CPUEmbedder()
    embed(["Initialize the existing CPU search model."])
    stop = False
    with selected_model(GPT41_MINI_MODEL), ThreadPoolExecutor(max_workers=workers) as executor:
        active = {}
        while pending or active:
            while pending and len(active) < workers and not stop:
                if budget.snapshot()["budget_exhausted"]:
                    stop = True
                    break
                case = pending.pop(0)
                future = executor.submit(run_case, case, key, guard, budget, embed, frozen)
                active[future] = case["id"]
            if not active:
                break
            done, _ = wait(active, timeout=10, return_when=FIRST_COMPLETED)
            for future in done:
                active.pop(future)
                result = future.result()
                if result["error"] and result["error"][0] in {
                    "authentication", "credits", "model_mismatch", "redirect_blocked",
                }:
                    stop = True
            atomic_json(ROOT / "summary.json", summarize(manifest, budget, guard), key)
    result = summarize(manifest, budget, guard)
    result.update(sources_unchanged=sources() == frozen, stopped_early=stop)
    atomic_json(ROOT / "summary.json", result, key)
    print(json.dumps({"attempted": result["attempted"], "finished": result["finished"],
                      "budget": result["budget"]}, indent=2), flush=True)
    return 0 if result["finished"] else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    args = parser.parse_args(argv)
    if PROMPT_VERSION != VERSION:
        raise EvaluationError("The evaluator is pinned to structured-evidence-v3.")
    manifest = load_manifest()
    if not args.run:
        print(json.dumps({"dry_run": True, "cases": len(manifest["cases"]),
                          "datasets": dict(Counter(c["dataset"] for c in manifest["cases"])),
                          "model": GPT41_MINI_MODEL, "prompt_version": VERSION,
                          "source_sha256": sources(), "limits": LIMITS,
                          "max_usd": MAX_USD, "credential_read": False}, indent=2))
        return 0
    with (ROOT / ".runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return run(manifest, workers=args.workers, resume=args.resume)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except EvaluationError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from None
