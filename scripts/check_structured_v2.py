"""One owner-requested, two-case contract check. No tuning or visual accuracy grading.

Default is read-only preflight. --run uses a NEW exclusive batch, at most two
sessions / 24 HTTP attempts / 12 per case. All previous ledgers remain closed.
"""

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]

from evaluate_structured_evidence import (  # noqa: E402
    CASES,
    GPT41_MINI_MODEL,
    CPUEmbedder,
    EvaluationError,
    ProvenanceVLM,
    RemotePipeline,
    UsageGuard,
    cache_seed,
    current_sources,
    file_sha,
    historical_control,
    historical_unchanged,
    ledger_counts,
    read_owner_key,
    regular_input,
    safe_failure,
    seed_bundle,
    selected_model,
    write_json,
)

from videomasys.zerogpu import structured_prompts, svp_prompts  # noqa: E402

ROOT = REPO / "artifacts/structured_recheck_2026-09-08"
PREVIOUS = REPO / "artifacts/structured_evidence_2026-09-08"
VERSION = "structured-evidence-v2"
LIMITS = {"max_runs": 2, "max_requests": 24, "max_requests_per_run": 12}


def hashes():
    return {**current_sources(), str(Path(__file__).relative_to(REPO)): file_sha(Path(__file__))}


def preflight():
    if ROOT.exists() or ROOT.is_symlink():
        raise EvaluationError("This recheck already exists; no overwrite, rerun, or reset.")
    if structured_prompts.PROMPT_VERSION != VERSION:
        raise EvaluationError("This check is pinned to structured-evidence-v2.")
    if any(getattr(structured_prompts, name) is not getattr(svp_prompts, name)
           for name in ("caption_prompt", "planner_prompt", "tool_prompt")):
        raise EvaluationError("The hybrid planning/retrieval prompts changed.")
    previous_ledger = PREVIOUS / "usage.sqlite3"
    previous_counts = ledger_counts(previous_ledger, PREVIOUS)
    if previous_counts != {"runs_used": 2, "request_attempts_used": 16}:
        raise EvaluationError("The previous structured batch is not in its preserved closed state.")
    seeds = {case: cache_seed(case) for case in CASES}
    controls = {}
    for case, seed in seeds.items():
        control = historical_control(case, seed)
        control["closed_ledgers"].append({
            "path": str(previous_ledger), "sha256": file_sha(previous_ledger), **previous_counts,
        })
        directory = PREVIOUS / f"structured__structured-evidence-v1__{case}"
        for name in ("manifest.json", "fixed-inputs.json", "bundle.json", "raw-trace.json",
                     "report.json", "source-snapshots.json"):
            path = regular_input(directory / name, PREVIOUS)
            control["control_files_sha256"][str(path)] = file_sha(path)
        controls[case] = control
    return seeds, controls, hashes()


def check_case(case, seed, controls, frozen, key, guard, embed):
    if hashes() != frozen:
        raise EvaluationError("Sources changed after preflight; no mixed-version checks.")
    output = ROOT / case
    output.mkdir(mode=0o700)
    with selected_model(GPT41_MINI_MODEL):
        pipeline = RemotePipeline(provider="openai", token=key, guard=guard, embed=embed,
                                  prompt_profile="structured", vlm_factory=ProvenanceVLM)
    bundle = seed_bundle(seed, output, pipeline.signature)
    pipeline.index(bundle)
    if not bundle["cached"]:
        raise EvaluationError("The check requires the preserved cache; no re-captioning.")
    bundle["trace"].append({
        "stage": "Video cache", "status": "complete", "cache_hit": True,
        "source": "immutable historical input seed", "scene_count": len(bundle["scenes"]),
        "scenes": bundle["scenes"],
    })
    write_json(output / "fixed-inputs.json", seed, key)
    started, failure = time.monotonic(), None
    try:
        with pipeline.analysis_session(bundle) as active:
            iterator = active.qa(bundle, seed["question"], seed["options"])
            try:
                for bundle in iterator:
                    print(f"{case}: {bundle['trace'][-1]['stage']}", flush=True)
            finally:
                iterator.close()
    except (Exception, KeyboardInterrupt) as exc:
        failure = safe_failure(exc)
        bundle["error"] = failure[1]
    elapsed = round(time.monotonic() - started, 3)
    complete = failure is None and "answer" in bundle and "verification" in bundle
    usage = guard.snapshot()
    report = {
        "case": case, "model": GPT41_MINI_MODEL, "prompt_version": VERSION,
        "qa_completed": complete, "result_is_execution_not_visual_accuracy": True,
        "elapsed_seconds": elapsed, "error": failure,
        "answer": bundle.get("answer"), "review_accepted": bundle.get("verification", {}).get(
            "accepted"), "status": bundle.get("status"), "allowance": usage,
        "source_hashes_unchanged": hashes() == frozen,
        "historical_controls_unchanged": historical_unchanged(controls),
        "public_space_modified": False,
    }
    write_json(output / "bundle.json", bundle, key)
    write_json(output / "raw-trace.json", bundle["trace"], key)
    write_json(output / "report.json", report, key)
    print(f"{case}: {'completed' if complete else 'stopped'} in {elapsed}s", flush=True)
    if failure and failure[0] == "interrupted":
        raise EvaluationError("Owner interrupted; no next case was started.")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Run both bounded paid contract checks")
    args = parser.parse_args()
    seeds, controls, frozen = preflight()
    if not args.run:
        print(json.dumps({"dry_run": True, "version": VERSION, "model": GPT41_MINI_MODEL,
                          "cases": list(CASES), "limits": LIMITS,
                          "key_read": False, "model_called": False, "output_created": False}))
        return 0
    key = read_owner_key()
    ROOT.mkdir(mode=0o700)
    write_json(ROOT / "manifest.json", {
        "owner_request": "check it; model perception errors are out of scope",
        "version": VERSION, "model": GPT41_MINI_MODEL, "cases": list(CASES), "limits": LIMITS,
        "source_sha256": frozen, "historical_controls": controls,
        "caption_generation_enabled": False, "public_space_modified": False,
        "automatic_reruns": False, "model_fallback": False,
    }, key)
    write_json(ROOT / "source-snapshots.json",
               {name: (REPO / name).read_text() for name in frozen}, key)
    guard, embed = UsageGuard(ROOT / "usage.sqlite3", **LIMITS), CPUEmbedder()
    results = [check_case(case, seeds[case], controls[case], frozen, key, guard, embed)
               for case in CASES]
    write_json(ROOT / "summary.json", {"results": results, "allowance": guard.snapshot()}, key)
    return 0 if all(result["qa_completed"] for result in results) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except EvaluationError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from None
    except Exception:
        print("Recheck setup/output failed; no automatic rerun.", file=sys.stderr)
        raise SystemExit(2) from None
