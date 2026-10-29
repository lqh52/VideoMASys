"""Explicit, bounded real API smoke test; no benchmark labels or cached answers.

Uses the local authenticated HF credential only for this owner-run test. The
deployment instead requires an inference-only HF_INFERENCE_TOKEN Space secret.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

from huggingface_hub import get_token

from videomasys.zerogpu.core import prepare_video
from videomasys.zerogpu.remote_runtime import RemotePipeline
from videomasys.zerogpu.usage import UsageGuard


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Actually submit paid API requests")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.run:
        parser.error("Explicit --run is required for real, potentially billable inference.")
    args.output.mkdir(parents=True, exist_ok=True)
    guard = UsageGuard(args.output / "usage.sqlite3", max_runs=2, max_requests=64)
    pipeline = RemotePipeline(
        token=os.environ.get("HF_INFERENCE_TOKEN") or get_token(), guard=guard,
    )
    library = json.loads(Path("space_library/library.json").read_text())
    cases = [("sofa_mc", library[0], library[0]["options"]),
             ("kitchen_open", library[2], [])]
    manifest = {
        "started_at": datetime.now(UTC).isoformat(), "model": pipeline.model_id,
        "embedding": pipeline.embedding_id,
        "cases": [case[0] for case in cases],
        "purpose": "Live API integration smoke tests on existing development library examples",
        "billing_note": "Provider-reported tokens may omit image tokens; not a billing estimate.",
        "sources": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(Path("src/videomasys/zerogpu").glob("*.py"))},
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    for name, example, options in cases:
        started = time.monotonic()
        bundle = prepare_video(Path("space_library") / example["file"], args.output / "cache")
        bundle.update(question=example["question"], options=options,
                      video_name=example["file"])
        print(f"{name}: starting", flush=True)
        try:
            with pipeline.analysis_session(bundle) as active:
                active.index(bundle)
                active.save_index(bundle)
                print(f"{name}: indexed {len(bundle['scenes'])} windows", flush=True)
                for update in active.qa(bundle, example["question"], options):
                    bundle = update
                    print(f"{name}: {bundle['trace'][-1]['stage']}", flush=True)
        except Exception as exc:
            bundle["error"] = str(exc)
            print(f"{name}: failed: {exc}", flush=True)
        bundle["elapsed_seconds"] = round(time.monotonic() - started, 2)
        (args.output / f"{name}.json").write_text(json.dumps(bundle, indent=2))
        print(json.dumps({"case": name, "seconds": bundle["elapsed_seconds"],
                          "answer": bundle.get("answer"), "status": bundle.get("status"),
                          "error": bundle.get("error")}), flush=True)
        if bundle.get("error"):
            # Do not keep spending after a failed integration check.
            break


if __name__ == "__main__":
    main()
