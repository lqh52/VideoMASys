"""Compare local clean/SVP QA logic using one immutable historical video cache.

Dry-run is read-only and never opens credentials, admits a run, or calls a model.
--run authorizes exactly one case/profile in a shared, non-resettable 60-attempt
batch. It never calls or deploys the public Hugging Face Space.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sqlite3
import sys
import time
from contextlib import closing
from copy import deepcopy
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]

from evaluate_openai_prompts import (  # noqa: E402
    CASES,
    EvaluationError,
    case_input,
    read_owner_key,
    safe_failure,
    selected_model,
    source_hashes,
    write_json,
)
from evaluate_prompt_tuning import ProvenanceVLM, json_sha, sha  # noqa: E402

from videomasys.zerogpu.core import Scene, SceneCache  # noqa: E402
from videomasys.zerogpu.remote_runtime import (  # noqa: E402
    EMBEDDING_ID,
    GPT41_MINI_MODEL,
    CPUEmbedder,
    RemotePipeline,
)
from videomasys.zerogpu.usage import UsageGuard  # noqa: E402

ROOT = REPO / "artifacts/svp_logic_2026-09-08"
BASELINE = REPO / "artifacts/local_clean_v2_2026-09-07"
LIMITS = {"max_runs": 4, "max_requests": 60, "max_requests_per_run": 32}
PROFILES = {"clean": "clean-roles-v2", "svp": "svp-roles-v1"}
VERSIONS = {"clean": ("clean-roles-v2",), "svp": ("svp-roles-v1", "svp-roles-v2")}
AMENDED_LIMITS = {"max_runs": 6, "max_requests": 60, "max_requests_per_run": 12}
AMENDMENT = {
    "id": "svp-v2-two-local-runs-within-original-60-attempt-cap",
    "reason": "One revised SVP candidate per existing case; no original run is repeated.",
    "original_limits": LIMITS,
    "amended_limits": AMENDED_LIMITS,
    "usage_at_amendment": {"runs_used": 4, "request_attempts_used": 36},
    "ledger_reset": False, "additional_request_allowance": 0,
    "additional_sessions": 2, "public_space_allowance_changed": False,
}


def file_sha(path):
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def regular_input(path, root):
    """Only expected local evaluation artifacts, not arbitrary linked files."""
    path = Path(path)
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise EvaluationError("A fixed evaluation input is missing or outside its allowed root.")
    return path


def cache_seed(case):
    """Read-only validated baseline captions/vectors; no answer/label enters input."""
    if case not in CASES:
        raise EvaluationError("Choose one authorized evaluation case.")
    directory = BASELINE / f"{GPT41_MINI_MODEL}__{case}"
    files = {name: regular_input(directory / name, BASELINE)
             for name in ("bundle.json", "manifest.json", "raw-trace.json")}
    original = json.loads(files["bundle.json"].read_text())
    manifest = json.loads(files["manifest.json"].read_text())
    inputs = case_input(case)
    video = regular_input(inputs["video"], REPO / "space_library")
    if (
        manifest.get("model") != GPT41_MINI_MODEL
        or manifest.get("prompt_profile") != "clean"
        or manifest.get("prompt_version") != PROFILES["clean"]
        or original.get("question") != inputs["question"]
        or original.get("options") != inputs["options"]
        or Path(original.get("video_path", "")).resolve() != video.resolve()
    ):
        raise EvaluationError("The historical baseline no longer matches this fixed case.")
    clean_path = REPO / "src/videomasys/zerogpu/clean_prompts.py"
    if manifest.get("source_sha256", {}).get(str(clean_path.relative_to(REPO))) != file_sha(
        clean_path
    ):
        raise EvaluationError("The baseline clean prompts changed; this is not the fixed control.")
    key = file_sha(video)
    if original.get("key") != key:
        raise EvaluationError("The fixed video bytes no longer match the historical cache.")
    duration = original.get("duration")
    if (
        not isinstance(duration, (int, float)) or not math.isfinite(duration)
        or not 0 < duration <= 180
    ):
        raise EvaluationError("The historical video duration is invalid.")
    database = regular_input(original["database"], BASELINE / "cache" / GPT41_MINI_MODEL)
    with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        db.execute("PRAGMA query_only=ON")
        rows = db.execute("SELECT signature, captions, vectors FROM scene_cache").fetchall()
    if len(rows) != 1:
        raise EvaluationError("The historical cache must contain exactly one index row.")
    signature, encoded_captions, encoded_vectors = rows[0]
    expected_suffix = (
        f":{GPT41_MINI_MODEL}@openai-chat-v1-none-strict:{EMBEDDING_ID}"
        f":prompts={PROFILES['clean']}"
    )
    if not isinstance(signature, str) or not signature.endswith(expected_suffix):
        raise EvaluationError("The historical model/embedding/index identity does not match.")
    captions, vectors = json.loads(encoded_captions), np.asarray(
        json.loads(encoded_vectors), dtype=np.float32
    )
    source_scenes = original["scenes"]
    if (
        not 1 <= len(source_scenes) <= 9
        or captions != [item.get("caption") for item in source_scenes]
        or any(not isinstance(text, str) or not text.strip() for text in captions)
        or vectors.shape != (len(source_scenes), 384)
        or not np.isfinite(vectors).all()
        or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)
    ):
        raise EvaluationError("The historical captions or normalized 384-D vectors are invalid.")
    scenes, frames = [], {}
    for index, item in enumerate(source_scenes):
        # Explicit model fields prevent any old answer/annotation fields propagating.
        scene = Scene.model_validate(item).model_dump()
        start = duration * index / len(source_scenes)
        end = duration * (index + 1) / len(source_scenes)
        if (
            scene["index"] != index or scene["start"] != start or scene["end"] != end
            or len(scene["frames"]) != 4 or len(scene.get("frame_times", [])) != 4
            or any(not math.isfinite(t) or not start <= t <= end for t in scene["frame_times"])
        ):
            raise EvaluationError("The baseline scene boundaries or index-frame mapping changed.")
        for frame in scene["frames"]:
            path = regular_input(frame, database.parent)
            frames[str(path)] = file_sha(path)
        scenes.append(scene)
    seed = {
        "key": key, "duration": duration, "video_path": str(video.resolve()),
        "video_name": video.name, "question": inputs["question"], "options": inputs["options"],
        "scenes": scenes, "vectors": vectors.tolist(), "original_signature": signature,
    }
    seed["provenance"] = {
        "baseline_files": {str(path): file_sha(path) for path in files.values()},
        "baseline_database": {"path": str(database), "sha256": file_sha(database)},
        "video_sha256": key, "index_frame_sha256": frames,
        "captions_sha256": json_sha(captions), "vectors_sha256": json_sha(vectors.tolist()),
        "fixed_input_sha256": json_sha(seed), "original_index_signature": signature,
        "cache_source": "immutable historical clean run, not generated by the candidate",
        "baseline_prompt_sha256": file_sha(clean_path),
        "ground_truth_in_model_input": False,
    }
    return seed


def preflight(profile, case):
    """Validate immutable inputs and construct the real pipeline without admission."""
    if profile not in PROFILES or case not in CASES:
        raise EvaluationError("Choose one authorized profile and case.")
    module = importlib.import_module(f"videomasys.zerogpu.{profile}_prompts")
    if module.PROMPT_VERSION not in VERSIONS[profile]:
        raise EvaluationError("This prompt version is not authorized for the bounded batch.")
    seed = cache_seed(case)
    with selected_model(GPT41_MINI_MODEL):
        pipeline = RemotePipeline(
            provider="openai", token="", embed=CPUEmbedder(),
            prompt_profile=profile, vlm_factory=ProvenanceVLM,
        )
    if pipeline.prompt_version != module.PROMPT_VERSION or pipeline.embedding_id != EMBEDDING_ID:
        raise EvaluationError("The local pipeline identity does not match this batch.")
    return seed, pipeline


def evaluation_directory(profile, version, case):
    if version not in VERSIONS.get(profile, ()) or case not in CASES:
        raise EvaluationError("This profile/version/case is not authorized for this batch.")
    prefix = f"{profile}__{version}" if version == "svp-roles-v2" else profile
    return ROOT / f"{prefix}__{case}"


def protocol(version, case):
    """V2 can spend only remaining requests in the already-used original ledger."""
    if version != "svp-roles-v2":
        return dict(LIMITS), None
    ledger = regular_input(ROOT / "usage.sqlite3", ROOT)
    with closing(sqlite3.connect(ledger.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        db.execute("PRAGMA query_only=ON")
        rows = db.execute("SELECT runs, requests FROM usage WHERE id=1").fetchall()
    if len(rows) != 1:
        raise EvaluationError("The existing batch ledger is invalid; it will not be replaced.")
    runs, requests = rows[0]
    if (
        type(runs) is not int or type(requests) is not int
        or not 4 <= runs < 6 or not 36 <= requests < 60
    ):
        raise EvaluationError(
            "The existing batch allowance is exhausted or does not match the amendment; "
            "no ledger reset or new batch is permitted."
        )
    previous = regular_input(ROOT / f"svp__{case}" / "manifest.json", ROOT)
    manifest = json.loads(previous.read_text())
    if manifest.get("prompt_version") != "svp-roles-v1" or manifest.get("case") != case:
        raise EvaluationError("The preserved SVP v1 case manifest does not match this amendment.")
    remaining = 60 - requests
    limits = {**AMENDED_LIMITS, "max_requests_per_run": min(12, remaining)}
    return limits, {
        **AMENDMENT,
        "allowance_before_run": {"runs_used": runs, "request_attempts_used": requests},
        "remaining_global_request_attempts": remaining,
        "max_attempts_this_run": limits["max_requests_per_run"],
        "prior_candidate_manifest": {"path": str(previous), "sha256": file_sha(previous)},
        "prior_candidate_source_sha256": manifest["source_sha256"],
    }


def seed_bundle(seed, output, signature):
    """Copy captions/vectors into an isolated index; never update the source cache."""
    frames = output / "frames"
    frames.mkdir(mode=0o700)
    bundle = {name: deepcopy(seed[name]) for name in (
        "key", "duration", "video_path", "video_name", "question", "options", "scenes"
    )}
    bundle.update(database=str(output / "index.sqlite"), frame_directory=str(frames),
                  trace=[], cached=True)
    cache = SceneCache(bundle["database"], signature)
    cache.save(bundle["scenes"], np.asarray(seed["vectors"], dtype=np.float32))
    restored = cache.load(bundle["scenes"])
    if restored is None or restored[0] != bundle["scenes"] or not np.array_equal(
        restored[1], np.asarray(seed["vectors"], dtype=np.float32)
    ):
        raise EvaluationError("The isolated caption/vector cache did not round-trip exactly.")
    return bundle


def evaluate(profile, case, *, run=False):
    seed, pipeline = preflight(profile, case)
    version = pipeline.prompt_version
    output = evaluation_directory(profile, version, case)
    if output.exists():
        raise EvaluationError("This profile/case already exists; no rerun or overwrite.")
    limits, amendment = protocol(version, case)
    if not run:
        print(json.dumps({
            "dry_run": True, "case": case, "profile": profile, "model": GPT41_MINI_MODEL,
            "prompt_version": version, "scene_count": len(seed["scenes"]),
            "fixed_input_sha256": seed["provenance"]["fixed_input_sha256"],
            "original_index_signature": seed["original_signature"],
            "evaluation_index_signature": pipeline.signature,
            "limits": limits, "protocol_amendment": amendment,
            "credential_read": False, "model_called": False,
            "public_space_modified": False,
        }, indent=2), flush=True)
        return 0
    ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        output.mkdir(mode=0o700)
    except FileExistsError:
        raise EvaluationError("This profile/case already exists; no rerun or overwrite.") from None
    # All index work happens outside an admitted session: a cache miss cannot
    # silently turn this comparison into paid recaptioning.
    bundle = seed_bundle(seed, output, pipeline.signature)
    pipeline.index(bundle)
    if not bundle["cached"]:
        raise EvaluationError("This evaluation requires the exact historical cache hit.")
    key = read_owner_key()
    hashes = source_hashes()
    hashes.update({str(path.relative_to(REPO)): sha(path.read_bytes()) for path in (
        Path(__file__), REPO / "scripts/evaluate_prompt_tuning.py",
    )})
    manifest = {
        "model": GPT41_MINI_MODEL, "provider": "OpenAI", "execution": "local",
        "case": case, "prompt_profile": profile, "prompt_version": version,
        "limits": limits, "protocol_amendment": amendment, "ledger": str(ROOT / "usage.sqlite3"),
        "source_sha256": hashes, "provenance": seed["provenance"],
        "original_index_signature": seed["original_signature"],
        "evaluation_index_signature": pipeline.signature,
        "cache_rebound_explicitly": True, "caption_generation_enabled": False,
        "comparison_scope": "same cached index; retrieval, selected pixels and QA logic may differ",
        "validation_scope": "known regression cases, not unseen holdouts",
        "automatic_reruns": False, "model_fallback": False, "public_space_modified": False,
    }
    write_json(output / "manifest.json", manifest, key)
    write_json(output / "fixed-inputs.json", seed, key)
    write_json(output / "prompt-sources.json", {
        name: (REPO / "src/videomasys/zerogpu" / name).read_text()
        for name in ("clean_prompts.py", "svp_prompts.py")
    }, key)
    if amendment is not None:
        write_json(output / "source-snapshots.json", {
            str(path.relative_to(REPO)): path.read_text()
            for path in (REPO / "src/videomasys/zerogpu/core.py",
                         REPO / "src/videomasys/zerogpu/svp_retrieval.py")
        }, key)
    bundle["trace"].append({
        "stage": "Video cache", "status": "complete", "cache_hit": True,
        "cache_version": pipeline.signature, "scene_count": len(bundle["scenes"]),
        "scenes": bundle["scenes"], "source": "explicit immutable evaluation seed",
        "original_index_signature": seed["original_signature"],
        "fixed_input_sha256": seed["provenance"]["fixed_input_sha256"],
        "segmentation": "uniform temporal windows; not learned scene boundaries",
    })
    completed, failure, guard = False, None, None
    started = time.monotonic()
    try:
        guard = UsageGuard(ROOT / "usage.sqlite3", **limits)
        with selected_model(GPT41_MINI_MODEL):
            pipeline = RemotePipeline(
                provider="openai", token=key, guard=guard, embed=pipeline.embed,
                prompt_profile=profile, vlm_factory=ProvenanceVLM,
            )
        with pipeline.analysis_session(bundle) as active:
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
        "model": GPT41_MINI_MODEL, "case": case, "prompt_profile": profile,
        "prompt_version": version, "protocol_amendment": amendment,
        "qa_completed": completed, "result": "passed" if completed else "failed",
        "result_is_execution_not_correctness": True, "elapsed_seconds": elapsed,
        "last_pipeline_stage": last_stage, "answer": bundle.get("answer"),
        "verifier_accepted": bundle.get("verification", {}).get("accepted"),
        "error_code": failure[0] if failure else None, "error": failure[1] if failure else None,
        "raw_trace_file": "raw-trace.json", "bundle_file": "bundle.json", "allowance": None,
        "fixed_input_sha256": seed["provenance"]["fixed_input_sha256"],
        "selected_frame_paths": [path for scene in bundle.get("selected", [])
                                 for path in scene.get("frames", [])],
        "inference_invoked": any(
            attempt.get("request_attempted") is True
            for call in bundle.get("inference", {}).get("calls", [])
            for attempt in call.get("attempts", [])
        ),
    }
    if guard is not None:
        try:
            report["allowance"] = guard.snapshot()
        except Exception:
            report["allowance"] = {"status": "Usage summary unavailable"}
    # Session and generator have closed before writing outputs or reporting usage.
    write_json(output / "bundle.json", bundle, key)
    write_json(output / "raw-trace.json", bundle["trace"], key)
    write_json(output / "report.json", report, key)
    print(f"{'Complete' if completed else 'Stopped'}: {elapsed}s; artifacts: {output}", flush=True)
    return 0 if completed else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Admit one paid local evaluation")
    parser.add_argument("--profile", choices=tuple(PROFILES), required=True)
    parser.add_argument("--case", choices=CASES, required=True)
    args = parser.parse_args(argv)
    try:
        return evaluate(args.profile, args.case, run=args.run)
    except EvaluationError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception:
        print("Evaluation setup/output failed; no automatic rerun was attempted.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
