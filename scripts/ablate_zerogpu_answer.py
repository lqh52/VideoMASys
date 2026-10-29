"""Three bounded, real answer-prompt ablations from an existing evaluation trace.

Uses only its question, presented options, actual observations and selected frames.
Reference answers are deliberately not copied into this diagnostic's requests or
artifacts. No retrieval, captioning, answer repair, or additional inference runs.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from evaluate_zerogpu_library import load_local_source, local_pipeline, save_json


def exact_choice(raw, options):
    candidate = raw.strip()
    if candidate in options:
        return candidate
    try:
        decoded = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return decoded if isinstance(decoded, str) and decoded in options else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vlm-url", default="http://127.0.0.1:8205/v1")
    parser.add_argument("--embedding-url", default="http://127.0.0.1:8206/v1")
    args = parser.parse_args()
    record = json.loads(args.record.read_text())
    provenance = json.loads((args.record.parent / "manifest.json").read_text())
    core, system_prompt, hashes = load_local_source(Path(provenance["source_archive"]))
    runtime_args = SimpleNamespace(
        source_archive=Path(provenance["source_archive"]),
        timeout=180,
        vlm_url=args.vlm_url,
        embedding_url=args.embedding_url,
        model_id=provenance["model_id"],
        embedding_id=provenance["embedding_id"],
    )
    pipeline, http = local_pipeline(core, system_prompt, runtime_args)
    question, options = record["example"]["question"], record["example"]["options"]
    selected = record["bundle"]["selected"]
    images = [frame for scene in selected for frame in scene["frames"]]
    observation = next(
        step["output"]
        for step in reversed(record["bundle"]["trace"])
        if step.get("stage") == "Evidence inspection" and step.get("status") == "complete"
    )
    context = f"Question: {question}\nChoices: {json.dumps(options)}\n"
    visual_prompt = (
        context + "Choose the best answer based on the chronological frames. "
        "Return the exact choice only."
    )
    variants = [
        ("compact_visual", visual_prompt, images),
        (
            "compact_observations_text_only",
            context
            + "Observed evidence: "
            + json.dumps(observation)
            + "\nChoose the best answer based on these observations. "
            "Return the exact choice only.",
            [],
        ),
        (
            "compact_visual_function_rule",
            context
            + "For purpose, distinguish the task/function achieved from the material/resource "
            "supplied; choose the functional explanation when observed use supports it. "
            "Choose the best answer based on the chronological frames. "
            "Return the exact choice only.",
            images,
        ),
    ]
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    directory = args.output_dir / stamp
    directory.mkdir(parents=True, exist_ok=False)
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "source_record": str(args.record),
        "source_sha256": hashes,
        "source_archive": provenance["source_archive"],
        "model_id": pipeline.model_id,
        "system_prompt": system_prompt,
        "generation_settings": pipeline.generation_settings,
        "max_tokens": 128,
        "question": question,
        "options": options,
        "selected_frames": images,
        "observations": observation,
        "reference_policy": "No benchmark labels loaded into requests or copied into artifacts.",
    }
    save_json(directory / "manifest.json", manifest)
    print(f"Artifacts: {directory}", flush=True)
    results = []
    try:
        for name, prompt, paths in variants:
            started = time.monotonic()
            result = {"variant": name, "prompt": prompt, "images": paths}
            try:
                raw = pipeline.generate(prompt, paths, 128)
                result.update(raw_response=raw, exact_selected_choice=exact_choice(raw, options))
            except Exception as exc:
                result["error"] = f"{type(exc).__name__}: {exc}"
            result["wall_seconds"] = round(time.monotonic() - started, 3)
            results.append(result)
            save_json(directory / f"{name}.json", result)
            save_json(directory / "results.json", results)
            print(
                json.dumps({k: v for k, v in result.items() if k not in {"prompt", "images"}}),
                flush=True,
            )
    finally:
        http.close()


if __name__ == "__main__":
    main()
