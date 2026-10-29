"""Two bounded local structured-evidence runs against preserved hybrid controls.

Without --run, preflight is read-only: no credentials, embeddings, admission, or
inference. Prior ledgers remain closed. This new batch has exactly two sessions,
24 actual HTTP attempts total, and at most 12 attempts per session; no reruns.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]

from evaluate_openai_prompts import (  # noqa: E402
    CASES,
    EvaluationError,
    read_owner_key,
    safe_failure,
    selected_model,
    write_json,
)
from evaluate_prompt_tuning import ProvenanceVLM  # noqa: E402
from evaluate_svp_hybrid import current_sources as shared_sources  # noqa: E402
from evaluate_svp_hybrid import ledger_counts  # noqa: E402
from evaluate_svp_logic import cache_seed, file_sha, regular_input, seed_bundle  # noqa: E402

from videomasys.zerogpu.remote_runtime import (  # noqa: E402
    EMBEDDING_ID,
    GPT41_MINI_MODEL,
    CPUEmbedder,
    RemotePipeline,
)
from videomasys.zerogpu.usage import UsageGuard  # noqa: E402

ROOT = REPO / "artifacts/structured_evidence_2026-09-08"
HYBRID_ROOT = REPO / "artifacts/svp_hybrid_2026-09-08"
SVP_ROOT = REPO / "artifacts/svp_logic_2026-09-08"
PROFILE, VERSION = "structured", "structured-evidence-v1"
LIMITS = {"max_runs": 2, "max_requests": 24, "max_requests_per_run": 12}
PROTOCOL_ID = "structured-evidence-two-cases-new-authorized-batch-2026-09-08"
UNCHANGED_LOGIC = (
    "src/videomasys/zerogpu/svp_prompts.py", "src/videomasys/zerogpu/svp_retrieval.py",
)


def output_directory(case):
    if case not in CASES:
        raise EvaluationError("Choose one of the two authorized structured-evidence cases.")
    return ROOT / f"{PROFILE}__{VERSION}__{case}"


def historical_control(case, seed):
    """Verify closed ledgers and exact inputs; historical answers never enter prompts."""
    ledgers = []
    for root, expected in (
        (HYBRID_ROOT, {"runs_used": 2, "request_attempts_used": 17}),
        (SVP_ROOT, {"runs_used": 6, "request_attempts_used": 54}),
    ):
        path = root / "usage.sqlite3"
        counts = ledger_counts(path, root)
        if counts != expected:
            raise EvaluationError("A prior ledger no longer matches its preserved closed state.")
        ledgers.append({"path": str(path), "sha256": file_sha(path), **counts})
    directory = HYBRID_ROOT / f"svp__svp-roles-v3__{case}"
    paths = {name: regular_input(directory / name, HYBRID_ROOT) for name in (
        "manifest.json", "fixed-inputs.json", "bundle.json", "raw-trace.json", "report.json"
    )}
    manifest = json.loads(paths["manifest.json"].read_text())
    if (
        manifest.get("model") != GPT41_MINI_MODEL or manifest.get("case") != case
        or manifest.get("prompt_profile") != "svp"
        or manifest.get("prompt_version") != "svp-roles-v3"
        or manifest.get("provenance", {}).get("fixed_input_sha256")
        != seed["provenance"]["fixed_input_sha256"]
    ):
        raise EvaluationError("The historical hybrid control does not match the fixed input.")
    unchanged = {name: file_sha(REPO / name) for name in UNCHANGED_LOGIC}
    if any(manifest.get("source_sha256", {}).get(name) != value
           for name, value in unchanged.items()):
        raise EvaluationError(
            "The tested hybrid planner/retriever changed; this is not downstream-only."
        )
    return {
        "closed_ledgers": ledgers,
        "comparison": "historical hybrid-v3 control, not a fresh paired run or unseen holdout",
        "control_files_sha256": {str(path): file_sha(path) for path in paths.values()},
        "unchanged_hybrid_logic_sha256": unchanged,
    }


def current_sources():
    hashes = shared_sources()
    hashes[str(Path(__file__).relative_to(REPO))] = file_sha(Path(__file__))
    return hashes


def batch_state(hashes):
    if not ROOT.exists():
        if ROOT.is_symlink():
            raise EvaluationError("The structured output root must not be a symlink.")
        return {"runs_used": 0, "request_attempts_used": 0}
    if ROOT.is_symlink() or not ROOT.is_dir():
        raise EvaluationError("The structured output root must be a regular directory.")
    usage = ledger_counts(ROOT / "usage.sqlite3", ROOT)
    if (
        usage["runs_used"] >= LIMITS["max_runs"]
        or usage["request_attempts_used"] >= LIMITS["max_requests"]
    ):
        raise EvaluationError("The structured batch is closed; no rerun, reset, or extension.")
    for case in CASES:
        directory = output_directory(case)
        if directory.exists() or directory.is_symlink():
            path = regular_input(directory / "manifest.json", ROOT)
            if json.loads(path.read_text()).get("source_sha256") != hashes:
                raise EvaluationError(
                    "Candidate sources changed after the first run; no mixed batch."
                )
    return usage


def preflight(case):
    output_directory(case)
    module = importlib.import_module("videomasys.zerogpu.structured_prompts")
    hybrid = importlib.import_module("videomasys.zerogpu.svp_prompts")
    if module.PROMPT_VERSION != VERSION:
        raise EvaluationError("Only structured-evidence-v1 is authorized for this batch.")
    if any(getattr(module, name) is not getattr(hybrid, name) for name in (
        "planner_prompt", "tool_prompt", "caption_prompt"
    )):
        raise EvaluationError(
            "Structured evidence must reuse the unchanged hybrid retrieval prompts."
        )
    seed = cache_seed(case)
    historical = historical_control(case, seed)
    hashes = current_sources()
    usage = batch_state(hashes)
    with selected_model(GPT41_MINI_MODEL):
        pipeline = RemotePipeline(
            provider="openai", token="", embed=CPUEmbedder(),
            prompt_profile=PROFILE, vlm_factory=ProvenanceVLM,
        )
    if pipeline.prompt_version != VERSION or pipeline.embedding_id != EMBEDDING_ID:
        raise EvaluationError("The local pipeline identity does not match the structured batch.")
    return seed, pipeline, historical, hashes, usage


def historical_unchanged(historical):
    hashes = dict(historical["control_files_sha256"])
    hashes.update({item["path"]: item["sha256"] for item in historical["closed_ledgers"]})
    return all(file_sha(Path(path)) == value for path, value in hashes.items())


def evaluate(case, *, run=False):
    seed, pipeline, historical, hashes, usage = preflight(case)
    output = output_directory(case)
    if output.exists() or output.is_symlink():
        raise EvaluationError("This structured case already exists; no rerun or overwrite.")
    manifest = {
        "protocol_id": PROTOCOL_ID, "model": GPT41_MINI_MODEL, "provider": "OpenAI",
        "execution": "local", "case": case, "prompt_profile": PROFILE, "prompt_version": VERSION,
        "limits": LIMITS, "ledger": str(ROOT / "usage.sqlite3"), "allowance_before_run": usage,
        "source_sha256": hashes, "provenance": seed["provenance"],
        "historical_comparison": historical, "original_index_signature": seed["original_signature"],
        "evaluation_index_signature": pipeline.signature, "cache_rebound_explicitly": True,
        "caption_generation_enabled": False,
        "comparison_scope": "same cached input and hybrid retrieval; new downstream contracts",
        "refinement": "the inspector and answer/reviewer share the exact final six frame files",
        "validation_scope": "two known regression cases; historical control, not unseen holdouts",
        "automatic_reruns": False, "model_fallback": False, "public_space_modified": False,
        "previous_batches_modified": False,
    }
    if not run:
        print(json.dumps({
            "dry_run": True, "case": case, "model": GPT41_MINI_MODEL, "prompt_version": VERSION,
            "scene_count": len(seed["scenes"]), "limits": LIMITS, "allowance_before_run": usage,
            "fixed_input_sha256": seed["provenance"]["fixed_input_sha256"],
            "historical_comparison": historical, "credential_read": False, "model_called": False,
            "public_space_modified": False,
        }, indent=2), flush=True)
        return 0
    ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        output.mkdir(mode=0o700)
    except FileExistsError:
        raise EvaluationError(
            "This structured case already exists; no rerun or overwrite."
        ) from None
    bundle = seed_bundle(seed, output, pipeline.signature)
    pipeline.index(bundle)
    if not bundle["cached"]:
        raise EvaluationError("This evaluation requires the exact historical cache hit.")
    snapshots = {name: (REPO / name).read_text() for name in hashes}
    if current_sources() != hashes:
        raise EvaluationError("Candidate sources changed during preflight; no run was admitted.")
    key = read_owner_key()
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
                prompt_profile=PROFILE, vlm_factory=ProvenanceVLM,
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
        "model": GPT41_MINI_MODEL, "case": case, "prompt_profile": PROFILE,
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
        "public_space_modified": False,
    }
    for name, read in (
        ("allowance", lambda: guard.snapshot() if guard is not None else None),
        ("sources_unchanged_during_run", lambda: current_sources() == hashes),
        ("historical_controls_unchanged", lambda: historical_unchanged(historical)),
    ):
        try:
            report[name] = read()
        except Exception:
            report[name] = None
    write_json(output / "bundle.json", bundle, key)
    write_json(output / "raw-trace.json", bundle["trace"], key)
    write_json(output / "report.json", report, key)
    print(f"{'Complete' if completed else 'Stopped'}: {elapsed}s; artifacts: {output}", flush=True)
    return 0 if completed else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Admit one paid local structured case")
    parser.add_argument("--case", choices=CASES, required=True)
    args = parser.parse_args(argv)
    try:
        return evaluate(args.case, run=args.run)
    except EvaluationError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception:
        print("Structured setup/output failed; no automatic rerun was attempted.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
