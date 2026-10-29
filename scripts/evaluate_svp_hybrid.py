"""One local hybrid run per known case, against immutable historical controls.

Default is a read-only dry run: no credential, inference, embedding, or admission.
--run admits one of TWO sessions in a NEW, explicitly authorized 24-HTTP-attempt
batch. The previous 6-session/54-attempt batch stays closed and is never modified.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sqlite3
import sys
import time
from contextlib import closing
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]

from evaluate_openai_prompts import (  # noqa: E402
    CASES,
    EvaluationError,
    read_owner_key,
    safe_failure,
    selected_model,
    source_hashes,
    write_json,
)
from evaluate_prompt_tuning import ProvenanceVLM  # noqa: E402
from evaluate_svp_logic import cache_seed, file_sha, regular_input, seed_bundle  # noqa: E402

from videomasys.zerogpu.remote_runtime import (  # noqa: E402
    EMBEDDING_ID,
    GPT41_MINI_MODEL,
    CPUEmbedder,
    RemotePipeline,
)
from videomasys.zerogpu.usage import UsageGuard  # noqa: E402

ROOT = REPO / "artifacts/svp_hybrid_2026-09-08"
HISTORICAL_ROOT = REPO / "artifacts/svp_logic_2026-09-08"
VERSION = "svp-roles-v3"
LIMITS = {"max_runs": 2, "max_requests": 24, "max_requests_per_run": 12}
PROTOCOL_ID = "svp-hybrid-two-cases-new-authorized-batch-2026-09-08"


def ledger_counts(path, root):
    path = regular_input(path, root)
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        db.execute("PRAGMA query_only=ON")
        rows = db.execute("SELECT runs, requests FROM usage WHERE id=1").fetchall()
    if len(rows) != 1 or any(type(value) is not int or value < 0 for value in rows[0]):
        raise EvaluationError("The fixed evaluation ledger is invalid; it will not be replaced.")
    return {"runs_used": rows[0][0], "request_attempts_used": rows[0][1]}


def historical_controls(case, seed):
    """Read existing evidence only; never turn an old allowance into a new one."""
    ledger = HISTORICAL_ROOT / "usage.sqlite3"
    closed = ledger_counts(ledger, HISTORICAL_ROOT)
    if closed != {"runs_used": 6, "request_attempts_used": 54}:
        raise EvaluationError(
            "The previous batch is not the preserved closed 6-run/54-attempt batch."
        )
    controls = {}
    for profile, version, directory in (
        ("clean", "clean-roles-v2", f"clean__{case}"),
        ("svp", "svp-roles-v2", f"svp__svp-roles-v2__{case}"),
    ):
        paths = {name: regular_input(HISTORICAL_ROOT / directory / name, HISTORICAL_ROOT)
                 for name in ("manifest.json", "bundle.json", "raw-trace.json", "report.json")}
        manifest = json.loads(paths["manifest.json"].read_text())
        if (
            manifest.get("model") != GPT41_MINI_MODEL
            or manifest.get("case") != case or manifest.get("prompt_profile") != profile
            or manifest.get("prompt_version") != version
            or manifest.get("provenance", {}).get("fixed_input_sha256")
            != seed["provenance"]["fixed_input_sha256"]
        ):
            raise EvaluationError("A historical control does not match the fixed candidate input.")
        controls[version] = {
            "comparison": "historical control, not a fresh paired run",
            "files_sha256": {str(path): file_sha(path) for path in paths.values()},
        }
    return {
        "closed_batch": {"ledger": str(ledger), "sha256": file_sha(ledger), **closed},
        "controls": controls,
    }


def current_sources():
    hashes = source_hashes()
    for name in ("evaluate_svp_hybrid.py", "evaluate_svp_logic.py", "evaluate_prompt_tuning.py"):
        path = REPO / "scripts" / name
        hashes[str(path.relative_to(REPO))] = file_sha(path)
    return hashes


def batch_state(hashes):
    if not ROOT.exists():
        if ROOT.is_symlink():
            raise EvaluationError("The hybrid output root must not be a symlink.")
        return {"runs_used": 0, "request_attempts_used": 0}
    if ROOT.is_symlink() or not ROOT.is_dir():
        raise EvaluationError("The hybrid output root must be a regular directory.")
    usage = ledger_counts(ROOT / "usage.sqlite3", ROOT)
    if usage["runs_used"] >= 2 or usage["request_attempts_used"] >= 24:
        raise EvaluationError("The new hybrid batch is closed; no rerun, reset, or extension.")
    for case in CASES:
        directory = ROOT / f"svp__{VERSION}__{case}"
        if directory.exists():
            path = regular_input(directory / "manifest.json", ROOT)
            previous = json.loads(path.read_text())
            if previous.get("source_sha256") != hashes:
                raise EvaluationError(
                    "Candidate sources changed after the first run; no mixed batch."
                )
    return usage


def preflight(case):
    if case not in CASES:
        raise EvaluationError("Choose one of the two authorized hybrid cases.")
    module = importlib.import_module("videomasys.zerogpu.svp_prompts")
    if module.PROMPT_VERSION != VERSION:
        raise EvaluationError("Only the frozen svp-roles-v3 hybrid is authorized for this batch.")
    seed = cache_seed(case)
    historical = historical_controls(case, seed)
    hashes = current_sources()
    usage = batch_state(hashes)
    with selected_model(GPT41_MINI_MODEL):
        pipeline = RemotePipeline(
            provider="openai", token="", embed=CPUEmbedder(),
            prompt_profile="svp", vlm_factory=ProvenanceVLM,
        )
    if pipeline.prompt_version != VERSION or pipeline.embedding_id != EMBEDDING_ID:
        raise EvaluationError("The local pipeline identity does not match the hybrid batch.")
    return seed, pipeline, historical, hashes, usage


def evaluate(case, *, run=False):
    seed, pipeline, historical, hashes, usage = preflight(case)
    output = ROOT / f"svp__{VERSION}__{case}"
    if output.exists() or output.is_symlink():
        raise EvaluationError("This hybrid case already exists; no rerun or overwrite.")
    manifest = {
        "protocol_id": PROTOCOL_ID, "model": GPT41_MINI_MODEL,
        "provider": "OpenAI", "execution": "local", "case": case,
        "prompt_profile": "svp", "prompt_version": VERSION, "limits": LIMITS,
        "ledger": str(ROOT / "usage.sqlite3"), "allowance_before_run": usage,
        "source_sha256": hashes, "provenance": seed["provenance"],
        "historical_comparison": historical,
        "original_index_signature": seed["original_signature"],
        "evaluation_index_signature": pipeline.signature,
        "cache_rebound_explicitly": True, "caption_generation_enabled": False,
        "comparison_scope": "same historical caption/vector seed, newly selected pixels and QA",
        "validation_scope": "two known regression cases, not unseen holdouts",
        "automatic_reruns": False, "model_fallback": False, "public_space_modified": False,
        "previous_batch_modified": False,
    }
    if not run:
        print(json.dumps({
            "dry_run": True, "case": case, "model": GPT41_MINI_MODEL,
            "prompt_version": VERSION, "scene_count": len(seed["scenes"]),
            "fixed_input_sha256": seed["provenance"]["fixed_input_sha256"],
            "limits": LIMITS, "allowance_before_run": usage,
            "historical_comparison": historical,
            "credential_read": False, "model_called": False, "public_space_modified": False,
        }, indent=2), flush=True)
        return 0
    ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        output.mkdir(mode=0o700)
    except FileExistsError:
        raise EvaluationError("This hybrid case already exists; no rerun or overwrite.") from None
    bundle = seed_bundle(seed, output, pipeline.signature)
    pipeline.index(bundle)
    if not bundle["cached"]:
        raise EvaluationError("The hybrid evaluation requires the exact historical cache hit.")
    key = read_owner_key()
    snapshots = {name: (REPO / name).read_text() for name in hashes}
    if current_sources() != hashes:
        raise EvaluationError("Candidate sources changed during preflight; no run was admitted.")
    write_json(output / "manifest.json", manifest, key)
    write_json(output / "fixed-inputs.json", seed, key)
    write_json(output / "source-snapshots.json", snapshots, key)
    bundle["trace"].append({
        "stage": "Video cache", "status": "complete", "cache_hit": True,
        "cache_version": pipeline.signature, "scene_count": len(bundle["scenes"]),
        "scenes": bundle["scenes"], "source": "explicit immutable evaluation seed",
        "fixed_input_sha256": seed["provenance"]["fixed_input_sha256"],
        "segmentation": "uniform temporal windows; not learned scene boundaries",
    })
    completed, failure, guard = False, None, None
    started = time.monotonic()
    try:
        guard = UsageGuard(ROOT / "usage.sqlite3", **LIMITS)
        with selected_model(GPT41_MINI_MODEL):
            active_pipeline = RemotePipeline(
                provider="openai", token=key, guard=guard, embed=pipeline.embed,
                prompt_profile="svp", vlm_factory=ProvenanceVLM,
            )
        with active_pipeline.analysis_session(bundle) as active:
            if active.signature != manifest["evaluation_index_signature"]:
                raise EvaluationError("The admitted pipeline changed its evaluation identity.")
            iterator = active.qa(bundle, seed["question"], seed["options"])
            try:
                for bundle in iterator:
                    print(f"{bundle['trace'][-1]['stage']} finished", flush=True)
            finally:
                iterator.close()
            completed = "answer" in bundle and "verification" in bundle
    except (Exception, KeyboardInterrupt) as exc:
        failure = safe_failure(exc)
        bundle["error"] = failure[1]
    elapsed = round(time.monotonic() - started, 3)
    last_stage = bundle["trace"][-1]["stage"]
    if "inference" in bundle:
        bundle["trace"].append({"stage": "Hosted inference usage", **bundle["inference"]})
    report = {
        "model": GPT41_MINI_MODEL, "case": case, "prompt_profile": "svp",
        "prompt_version": VERSION, "protocol_id": PROTOCOL_ID,
        "qa_completed": completed, "result": "passed" if completed else "failed",
        "result_is_execution_not_correctness": True, "elapsed_seconds": elapsed,
        "last_pipeline_stage": last_stage, "answer": bundle.get("answer"),
        "verifier_accepted": bundle.get("verification", {}).get("accepted"),
        "error_code": failure[0] if failure else None, "error": failure[1] if failure else None,
        "raw_trace_file": "raw-trace.json", "bundle_file": "bundle.json",
        "fixed_input_sha256": seed["provenance"]["fixed_input_sha256"],
        "selected_frame_paths": [path for scene in bundle.get("selected", [])
                                 for path in scene.get("frames", [])],
        "inference_invoked": any(
            attempt.get("request_attempted") is True
            for call in bundle.get("inference", {}).get("calls", [])
            for attempt in call.get("attempts", [])
        ),
        "allowance": guard.snapshot() if guard is not None else None,
        "sources_unchanged_during_run": current_sources() == hashes,
        "previous_batch_modified": False, "public_space_modified": False,
    }
    write_json(output / "bundle.json", bundle, key)
    write_json(output / "raw-trace.json", bundle["trace"], key)
    write_json(output / "report.json", report, key)
    print(f"{'Complete' if completed else 'Stopped'}: {elapsed}s; artifacts: {output}", flush=True)
    return 0 if completed else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Admit one paid local hybrid case")
    parser.add_argument("--case", choices=CASES, required=True)
    args = parser.parse_args(argv)
    try:
        return evaluate(args.case, run=args.run)
    except EvaluationError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception:
        print("Hybrid setup/output failed; no automatic rerun was attempted.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
