"""One owner-triggered, bounded local QA evaluation; never auto-rerun or fall back."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import stat
import sys
import time
from contextlib import contextmanager
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dotenv import dotenv_values  # noqa: E402

from videomasys.zerogpu.clean_prompts import PROMPT_VERSION  # noqa: E402
from videomasys.zerogpu.core import prepare_video  # noqa: E402
from videomasys.zerogpu.remote import HostedVLMError  # noqa: E402
from videomasys.zerogpu.remote_runtime import (  # noqa: E402
    GPT41_MINI_MODEL,
    GPT54_MINI_MODEL,
    CPUEmbedder,
    RemotePipeline,
)
from videomasys.zerogpu.usage import UsageGuard, UsageLimitError  # noqa: E402

MODELS = (GPT41_MINI_MODEL, GPT54_MINI_MODEL)
CASES = ("sofa_mc", "kitchen_open")
KITCHEN_QUESTION = (
    "What is the person mainly trying to accomplish across the video? "
    "Describe the visible actions that support your answer."
)


class EvaluationError(RuntimeError):
    """Only fixed owner-safe messages are raised by runner configuration checks."""


def read_owner_key():
    path = REPO / ".env"
    try:
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
            raise EvaluationError("The repository .env must be an owner-owned regular file.")
        if info.st_mode & 0o077:
            raise EvaluationError("The repository .env must be owner-only (chmod 600 .env).")
        # Do not interpolate, source the file, or load any settings into os.environ.
        logger = logging.getLogger("dotenv.main")
        previous = logger.disabled
        logger.disabled = True
        try:
            key = dotenv_values(path, interpolate=False).get("OPENAI_API_KEY")
        finally:
            logger.disabled = previous
    except EvaluationError:
        raise
    except Exception:
        raise EvaluationError("Cannot read the repository owner credential file.") from None
    if (
        not isinstance(key, str) or not key or len(key) > 8192
        or any(c.isspace() or ord(c) < 32 for c in key)
    ):
        raise EvaluationError("Add a valid OPENAI_API_KEY to the repository .env.")
    return key


@contextmanager
def selected_model(model):
    """RemotePipeline currently reads its model at construction; restore ambient state."""
    previous = os.environ.get("VLM_MODEL")
    os.environ["VLM_MODEL"] = model
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("VLM_MODEL", None)
        else:
            os.environ["VLM_MODEL"] = previous


def case_input(case):
    library = json.loads((REPO / "space_library/library.json").read_text())
    entry = library[0 if case == "sofa_mc" else 2]
    filename = "nextqa-4882821564.mp4" if case == "sofa_mc" else "egoschema-0074f737.mp4"
    if entry["file"] != filename:
        raise EvaluationError("The fixed evaluation library entry has changed.")
    return {
        "video": REPO / "space_library" / filename,
        "question": entry["question"] if case == "sofa_mc" else KITCHEN_QUESTION,
        "options": entry["options"] if case == "sofa_mc" else [],
    }


def source_hashes():
    # Explicit source allowlist: no environment files, credentials, or secret hashes.
    paths = [Path(__file__), REPO / "space_library/library.json"]
    paths += sorted((REPO / "src/videomasys").rglob("*.py"))
    return {
        str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths if not path.is_symlink()
    }


def write_json(path, value, key):
    # Defensive exact-key removal even from unusual downstream exception text.
    def redact(item):
        if isinstance(item, str):
            return item.replace(key, "[REDACTED]")
        if isinstance(item, list):
            return [redact(child) for child in item]
        if isinstance(item, dict):
            return {redact(name): redact(child) for name, child in item.items()}
        return item

    serialized = json.dumps(redact(value), ensure_ascii=False, indent=2)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(serialized + "\n")


def safe_failure(exc):
    if isinstance(exc, HostedVLMError):
        return exc.code, str(exc)
    if isinstance(exc, UsageLimitError):
        return "allowance", "The bounded evaluation allowance has been reached."
    if isinstance(exc, KeyboardInterrupt):
        return "interrupted", "The owner interrupted this evaluation; it was not restarted."
    return "pipeline_error", "The evaluation stopped; inspect the retained stage trace."


def evaluate(model, case, output_root):
    if model not in MODELS or case not in CASES:
        raise EvaluationError("Choose one supported pinned model and evaluation case.")
    key = read_owner_key()
    inputs = case_input(case)
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    output = root / f"{model}__{case}"
    try:
        output.mkdir(mode=0o700)
    except FileExistsError:
        raise EvaluationError(
            "This model/case already has an evaluation; no rerun allowed."
        ) from None
    manifest = {
        "model": model, "case": case, "provider": "OpenAI", "prompt_profile": "clean",
        "prompt_version": PROMPT_VERSION, "source_sha256": source_hashes(),
        "limits": {"max_runs": 4, "max_requests": 120, "max_requests_per_run": 32},
        "ledger": str(root / "usage.sqlite3"), "automatic_reruns": False,
        "model_fallback": False, "execution": "local CPU orchestration with hosted API",
    }
    write_json(output / "manifest.json", manifest, key)
    started = time.monotonic()
    bundle = {"trace": [], "question": inputs["question"], "options": inputs["options"]}
    completed = False
    failure = None
    guard = None
    stage = "Video preparation"
    try:
        print("Video preparation started", flush=True)
        bundle = prepare_video(inputs["video"], root / "cache" / model)
        bundle.update(question=inputs["question"], options=inputs["options"],
                      video_name=inputs["video"].name)
        guard = UsageGuard(root / "usage.sqlite3", max_runs=4, max_requests=120,
                           max_requests_per_run=32)
        with selected_model(model):
            pipeline = RemotePipeline(provider="openai", token=key, guard=guard,
                                      embed=CPUEmbedder(), prompt_profile="clean")
        with pipeline.analysis_session(bundle) as active:
            stage = "Scene captioning"
            print("Scene captioning started", flush=True)
            active.index(bundle)
            if not bundle["cached"]:
                active.save_index(bundle)
            bundle["trace"].append({
                "stage": "Video cache", "status": "complete", "cache_hit": bundle["cached"],
                "cache_version": active.signature, "scene_count": len(bundle["scenes"]),
                "segmentation": "uniform temporal windows; not learned scene boundaries",
                "scenes": bundle["scenes"],
            })
            print("Video cache complete", flush=True)
            iterator = active.qa(bundle, inputs["question"], inputs["options"])
            try:
                for update in iterator:
                    bundle = update
                    stage = bundle["trace"][-1]["stage"]
                    print(f"{stage} finished", flush=True)
            finally:
                iterator.close()
            completed = "answer" in bundle and "verification" in bundle
    except (Exception, KeyboardInterrupt) as exc:
        completed = False
        failure = safe_failure(exc)
        bundle["error"] = failure[1]
    # The analysis context and generator have closed before usage/trace publication.
    elapsed = round(time.monotonic() - started, 3)
    if bundle["trace"]:
        stage = bundle["trace"][-1].get("stage", stage)
    if "inference" in bundle:
        bundle["trace"].append({"stage": "Hosted inference usage", **bundle["inference"]})
    report = {
        "fixture_only": False, "execution": "local", "model": model,
        "inference_invoked": any(
            attempt.get("request_attempted") is True
            for call in bundle.get("inference", {}).get("calls", [])
            for attempt in call.get("attempts", [])
        ),
        "expected_model": model, "question": inputs["question"], "options": inputs["options"],
        "open_ended": not inputs["options"], "qa_completed": completed,
        "result": "passed" if completed else "failed",
        "qa_state": "complete" if completed else "failed", "trace_captured": True,
        "ui_checks_passed": None, "last_pipeline_stage": stage,
        "verifier_accepted": bundle.get("verification", {}).get("accepted"),
        "answer": bundle.get("answer"), "elapsed_seconds": elapsed,
        "status": f"{'Analysis complete' if completed else 'Analysis stopped'} · {elapsed}s total",
        "error": failure[1] if failure else None,
        "error_code": failure[0] if failure else None,
        "raw_trace_file": "raw-trace.json", "bundle_file": "bundle.json",
        "selected_frame_paths": [p for scene in bundle.get("selected", [])
                                 for p in scene.get("frames", [])],
    }
    if guard is not None:
        try:
            report["allowance"] = guard.snapshot()
        except Exception:
            report["allowance"] = {"status": "Usage summary unavailable"}
    write_json(output / "bundle.json", bundle, key)
    write_json(output / "raw-trace.json", bundle["trace"], key)
    write_json(output / "report.json", report, key)
    print(f"{'Complete' if completed else 'Stopped'}: {elapsed}s; artifacts: {output}", flush=True)
    return 0 if completed else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", required=True)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        return evaluate(args.model, args.case, args.output_root)
    except EvaluationError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception:
        print("Evaluation setup/output failed; no automatic rerun was attempted.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
