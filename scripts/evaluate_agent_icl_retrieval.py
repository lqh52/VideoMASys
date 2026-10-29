"""Offline CPU retrieval audit of actual saved planner/compiler generations.

No VLM, GPU, video decoding, HTTP service, or answer scoring is performed. Only
the saved scene captions and generated locate queries enter MiniLM. Missing cache
assets fail explicitly; this script never downloads weights or overwrites results.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

from evaluate_agent_icl import ROOT, file_sha, import_live, write_json

CASE_IDS = ("sofa_mc", "kitchen_mc", "kitchen_open")


def validated_case(path, core):
    from videomasys.agents.tool_sequence import ToolProgramValidator

    document = json.loads(path.read_text())
    case = document["case"]
    if case["id"] not in CASE_IDS or case["kind"] != "real_cached_video_captions":
        raise ValueError("Only the three authorized actual-caption cases are supported")
    if document.get("error") or not document.get("contracts_passed"):
        raise ValueError("Saved generation did not complete its structural contracts")
    stages = {}
    for name in ("Planner", "Tool agent"):
        entries = [entry for entry in document["trace"] if entry["stage"] == name]
        if len(entries) != 1 or entries[0]["status"] != "complete":
            raise ValueError(f"Expected one completed actual {name} stage")
        stages[name] = entries[0]
    plan = core.QAPlan.model_validate(stages["Planner"]["output"])
    program = core.ToolProgram.model_validate(stages["Tool agent"]["output"])
    ToolProgramValidator().validate(program, plan)
    scenes = [
        {key: scene[key] for key in ("index", "start", "end", "caption")}
        for scene in case["scenes"]
    ]
    return document, case, plan, program, scenes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--hf-home", type=Path, default=Path("/vast/s224013995/cache"))
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("--output-dir must not exist; previous results are preserved")
    os.environ.update(
        CUDA_VISIBLE_DEVICES="",
        HF_HOME=str(args.hf_home),
        HF_HUB_CACHE=str(args.hf_home / "hub"),
        HUGGINGFACE_HUB_CACHE=str(args.hf_home / "hub"),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        TOKENIZERS_PARALLELISM="false",
    )
    core, _ = import_live()
    from videomasys.zerogpu.retrieval import execute_program

    generation_manifest_path = args.input_dir / "manifest.json"
    generation_manifest = json.loads(generation_manifest_path.read_text())
    input_paths = [args.input_dir / f"{name}.json" for name in CASE_IDS]
    prepared = [(path, validated_case(path, core)) for path in input_paths]
    source_files = (
        "src/videomasys/zerogpu/core.py",
        "src/videomasys/zerogpu/prompts.py",
        "src/videomasys/zerogpu/retrieval.py",
        "src/videomasys/agents/tool_sequence.py",
        "src/videomasys/domain/schemas.py",
        "scripts/serve_eval_embeddings.py",
        "scripts/evaluate_agent_icl_retrieval.py",
    )
    hashes = {name: file_sha(ROOT / name) for name in source_files}
    generation_matches = {
        name: current == generation_manifest["source_sha256"][name]
        for name, current in hashes.items()
        if name in generation_manifest["source_sha256"]
    }
    if not all(generation_matches.values()):
        raise ValueError("Production source differs from the actual generation manifest")
    args.output_dir.mkdir(parents=True)
    write_json(
        args.output_dir / "manifest.json",
        {
            "started_at": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
            "mode": "offline_cpu_actual_caption_retrieval",
            "python": platform.python_version(),
            "generation_manifest": str(generation_manifest_path),
            "generation_manifest_sha256": file_sha(generation_manifest_path),
            "source_sha256": hashes,
            "generation_source_matches_current": generation_matches,
            "inputs_sha256": {str(path): file_sha(path) for path in input_paths},
            "gpu_used": False,
            "network_used": False,
            "images_inspected": False,
            "answer_accuracy_evaluated": False,
            "limitations": "Retrieval candidates only. Cached captions are fallible, and "
            "retaining a window does not establish what its pixels show.",
        },
    )
    started = time.monotonic()
    try:
        from serve_eval_embeddings import build_app

        # Reuse the exact existing pooling/tokenization implementation directly;
        # registering FastAPI routes does not bind a socket or start a service.
        app = build_app()
        embed = next(route.endpoint for route in app.routes if route.path == "/v1/embeddings")
        health = next(route.endpoint for route in app.routes if route.path == "/health")
        metadata = health()
    except Exception as exc:
        write_json(
            args.output_dir / "setup_failure.json",
            {"error_type": type(exc).__name__, "error": str(exc), "offline": True},
        )
        raise
    write_json(
        args.output_dir / "embedding_setup.json",
        {"metadata": metadata, "load_seconds": round(time.monotonic() - started, 4)},
    )
    summaries = []
    for path, (document, case, plan, program, scenes) in prepared:
        report = {
            "case_id": case["id"],
            "input_artifact": str(path),
            "input_sha256": file_sha(path),
            "question": case["question"],
            "options": case["options"],
            "actual_plan": plan.model_dump(),
            "actual_program": program.model_dump(),
            "actual_generation_repairs": document["repairs"],
            "scenes": scenes,
            "embedding_model": metadata,
            "image_qa_performed": False,
            "answer_accuracy_evaluated": False,
        }
        started = time.monotonic()
        try:
            captions = [scene["caption"] for scene in scenes]
            queries = [call.query for call in program.locate]
            scene_result = embed({"input": captions})
            query_result = embed({"input": queries})
            vectors = [row["embedding"] for row in scene_result["data"]]
            query_vectors = [row["embedding"] for row in query_result["data"]]
            report["embedding_seconds"] = round(time.monotonic() - started, 4)
            report["embedding_inputs"] = {"scene_captions": captions, "locate_queries": queries}
            report["scene_vectors"] = vectors
            report["query_vectors"] = query_vectors
            retrieval_started = time.monotonic()
            retrieval = execute_program(program, scenes, vectors, query_vectors, scope=plan.scope)
            report["retrieval_seconds"] = round(time.monotonic() - retrieval_started, 6)
            report["retrieval"] = retrieval
            report["completed"] = True
        except Exception as exc:
            report.update(completed=False, error_type=type(exc).__name__, error=str(exc))
        report["wall_seconds"] = round(time.monotonic() - started, 4)
        write_json(args.output_dir / f"{case['id']}.json", report)
        summary = {
            key: report[key]
            for key in ("case_id", "completed", "wall_seconds", "image_qa_performed")
        }
        if "retrieval" in report:
            summary.update(
                scope=plan.scope,
                relation_count=len(program.relate),
                selected_spans=[
                    {key: scene[key] for key in ("index", "start", "end")}
                    for scene in retrieval["selected"]
                ],
                temporal_coverage=retrieval.get("temporal_coverage"),
                uncovered_evidence_needs=retrieval["uncovered_evidence_needs"],
            )
        else:
            summary["error"] = report["error"]
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    write_json(args.output_dir / "summary.json", summaries)


if __name__ == "__main__":
    main()
