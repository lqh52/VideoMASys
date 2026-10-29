from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import httpx

from videomasys.agents.qa import AnswerComposer, AnswerVerifier, EvidenceInspector, QAPlanner
from videomasys.agents.scene_analyst import SceneAnalyst
from videomasys.domain.schemas import SceneAnalysis
from videomasys.evaluation.datasets import BenchmarkExample, load_egoschema, load_nextqa
from videomasys.media.ffmpeg import (
    Interval,
    detect_scene_intervals,
    extract_proxy_clip,
    extract_representative_frames,
    probe,
)
from videomasys.model_gateway.embeddings import HashEmbeddingGateway
from videomasys.model_gateway.vlm import VLMGateway
from videomasys.settings import Settings
from videomasys.tasks.qa.pipeline import (
    canonical_multiple_choice_answer,
    compact_observation,
)


async def wait_for_state(
    client: httpx.AsyncClient,
    path: str,
    terminal_states: set[str],
    timeout_seconds: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while True:
        response = await client.get(path)
        response.raise_for_status()
        payload = response.json()
        if payload["status"] in terminal_states:
            return payload
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for {path}")
        await asyncio.sleep(2.0)


async def evaluate_example(
    client: httpx.AsyncClient, example: BenchmarkExample, timeout_seconds: float
) -> dict[str, Any]:
    started = time.monotonic()
    with example.video_path.open("rb") as source:
        response = await client.post(
            "/v1/videos",
            files={"file": (example.video_path.name, source, "video/mp4")},
        )
    response.raise_for_status()
    video = await wait_for_state(
        client,
        f"/v1/videos/{response.json()['id']}",
        {"ready", "failed"},
        timeout_seconds,
    )
    if video["status"] != "ready":
        raise RuntimeError(f"ingestion failed: {video.get('error')}")

    response = await client.post(
        "/v1/qa",
        json={
            "video_id": video["id"],
            "question": example.question,
            "options": example.options,
        },
    )
    response.raise_for_status()
    task = await wait_for_state(
        client,
        f"/v1/tasks/{response.json()['id']}",
        {"succeeded", "failed", "cancelled"},
        timeout_seconds,
    )
    selected = (task.get("result") or {}).get("selected_option")
    return {
        "dataset": example.dataset,
        "example_id": example.example_id,
        "video_path": str(example.video_path),
        "question": example.question,
        "correct_option": example.correct_option,
        "selected_option": selected,
        "correct": selected == example.correct_option,
        "status": task["status"],
        "result": task.get("result"),
        "evidence": task.get("evidence"),
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


async def evaluate_standalone(
    example: BenchmarkExample, args: argparse.Namespace
) -> dict[str, Any]:
    """Exercise the real agents/media path without PostgreSQL for workstation smoke tests."""
    started = time.monotonic()
    settings = Settings(
        artifact_root=Path(args.artifact_root),
        vlm_base_url=args.vlm_url,
        vlm_model=args.vlm_model,
        embedding_provider="hash",
    )
    settings.ensure_directories()
    gateway = VLMGateway(settings)
    embedder = HashEmbeddingGateway(settings.embedding_dimension)
    try:
        media_info = await probe(example.video_path)
        cache_identity = (
            f"{example.video_path.resolve()}|scene={settings.scene_threshold}|"
            f"max={settings.max_scene_seconds}|frames={settings.frames_per_scene}|schema=2"
        )
        cache_key = uuid5(NAMESPACE_URL, cache_identity).hex
        cache_root = settings.artifact_root / "standalone" / cache_key
        interval_manifest = cache_root / "intervals.json"
        if interval_manifest.is_file():
            raw_intervals = await asyncio.to_thread(
                interval_manifest.read_text, encoding="utf-8"
            )
            intervals = [Interval(**item) for item in json.loads(raw_intervals)]
        else:
            intervals = await detect_scene_intervals(
                example.video_path,
                media_info.duration_seconds,
                settings.scene_threshold,
                settings.min_scene_seconds,
                settings.max_scene_seconds,
                settings.max_scenes_per_video,
            )
            await asyncio.to_thread(cache_root.mkdir, parents=True, exist_ok=True)
            interval_payload = [asdict(interval) for interval in intervals]
            await asyncio.to_thread(
                interval_manifest.write_text,
                json.dumps(interval_payload, indent=2),
                encoding="utf-8",
            )
        scene_analyst = SceneAnalyst(gateway)

        async def analyze_scene(index: int, interval: Interval) -> dict[str, Any]:
            scene_root = cache_root / f"scene_{index:04d}"
            frames = sorted(scene_root.glob("frame_*.jpg"))
            if len(frames) != settings.frames_per_scene:
                frames = await extract_representative_frames(
                    example.video_path, interval, scene_root, settings.frames_per_scene
                )
            analysis_path = scene_root / "analysis.json"
            if analysis_path.is_file():
                raw = await asyncio.to_thread(analysis_path.read_text, encoding="utf-8")
                analysis = SceneAnalysis.model_validate_json(raw)
            else:
                analysis = await scene_analyst.analyze(
                    frames, interval.start_seconds, interval.end_seconds
                )
                await asyncio.to_thread(
                    analysis_path.write_text,
                    analysis.model_dump_json(indent=2),
                    encoding="utf-8",
                )
            return {"interval": interval, "frames": frames, "analysis": analysis}

        scenes = await asyncio.gather(
            *(analyze_scene(index, interval) for index, interval in enumerate(intervals))
        )
        planner = QAPlanner(gateway)
        plan = await planner.plan(example.question, media_info.duration_seconds, example.options)
        query_vectors = await embedder.embed(plan.search_queries)
        caption_vectors = await embedder.embed(
            [scene["analysis"].caption for scene in scenes]
        )

        def relevance(vector: list[float]) -> float:
            return max(
                sum(left * right for left, right in zip(vector, query, strict=True))
                for query in query_vectors
            )

        ranked = sorted(
            zip(scenes, caption_vectors, strict=True),
            key=lambda pair: relevance(pair[1]),
            reverse=True,
        )[: plan.max_intervals]
        inspector = EvidenceInspector(gateway)

        async def inspect(index: int, scene: dict[str, Any]) -> dict[str, Any]:
            interval = scene["interval"]
            interval_key = (
                f"{round(interval.start_seconds * 1000):012d}_"
                f"{round(interval.end_seconds * 1000):012d}"
            )
            evidence_root = cache_root / "evidence" / interval_key
            evidence_frames = sorted(evidence_root.glob("frame_*.jpg"))
            if len(evidence_frames) != settings.qa_frames_per_interval:
                evidence_frames = await extract_representative_frames(
                    example.video_path,
                    interval,
                    evidence_root,
                    settings.qa_frames_per_interval,
                )
            proxy_path: Path | None = None
            if settings.qa_use_video_clip:
                proxy_path = evidence_root / "motion_proxy.mp4"
                if not proxy_path.is_file():
                    proxy_path = await extract_proxy_clip(
                        example.video_path,
                        interval,
                        proxy_path,
                        settings.qa_clip_fps,
                    )
            assessment = await inspector.inspect(
                question=example.question,
                options=example.options,
                caption=scene["analysis"].caption,
                start_seconds=interval.start_seconds,
                end_seconds=interval.end_seconds,
                frame_paths=evidence_frames,
                video_path=proxy_path,
            )
            assessment.observation = compact_observation(assessment.observation)
            node_id = uuid5(
                NAMESPACE_URL,
                f"{example.video_path.resolve()}#"
                f"{interval.start_seconds:.6f}-{interval.end_seconds:.6f}",
            )
            return {
                "node_id": node_id,
                "interval": interval,
                "assessment": assessment,
                "frames": evidence_frames,
            }

        inspected = await asyncio.gather(
            *(inspect(index, pair[0]) for index, pair in enumerate(ranked))
        )
        evidence = [item for item in inspected if item["assessment"].relevant]
        evidence_text = "\n".join(
            f"- node={item['node_id']} interval={item['interval'].start_seconds:.3f}–"
            f"{item['interval'].end_seconds:.3f}s confidence="
            f"{item['assessment'].confidence:.2f}: {item['assessment'].observation}"
            for item in evidence
        ) or "No relevant visual evidence was retrieved."
        answer = await AnswerComposer(gateway).compose(
            example.question, evidence_text, example.options
        )
        verification = await AnswerVerifier(gateway).verify(
            example.question, evidence_text, answer, example.options
        )
        if not verification.accepted and not verification.issues:
            verification.issues.append(
                "Verifier rejected the candidate without an actionable reason; "
                "the grounded candidate was retained."
            )
        selected = answer.selected_option
        if selected is not None and selected >= len(example.options):
            selected = None
        final_answer = answer.answer
        correction_is_complete = (
            verification.corrected_answer is not None
            and verification.corrected_selected_option is not None
            and verification.corrected_selected_option < len(example.options)
        )
        correction_applied = False
        if not verification.accepted and correction_is_complete:
            final_answer = verification.corrected_answer
            selected = verification.corrected_selected_option
            correction_applied = True
        if selected is not None:
            final_answer = canonical_multiple_choice_answer(selected, example.options)
        return {
            "dataset": example.dataset,
            "example_id": example.example_id,
            "video_path": str(example.video_path),
            "question": example.question,
            "correct_option": example.correct_option,
            "selected_option": selected,
            "correct": selected == example.correct_option,
            "status": "succeeded",
            "result": {
                "answer": final_answer,
                "selected_option": selected,
                "confidence": min(answer.confidence, verification.confidence),
                "verified": verification.accepted,
                "corrected_by_verifier": correction_applied,
                "verification_issues": verification.issues,
            },
            "plan": plan.model_dump(mode="json"),
            "scene_count": len(scenes),
            "inspected_intervals": len(inspected),
            "evidence": [
                {
                    "node_id": str(item["node_id"]),
                    "start_seconds": item["interval"].start_seconds,
                    "end_seconds": item["interval"].end_seconds,
                    "observation": item["assessment"].observation,
                    "confidence": item["assessment"].confidence,
                }
                for item in evidence
            ],
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
    finally:
        await gateway.close()
        await embedder.close()


async def async_main(args: argparse.Namespace) -> None:
    root = Path(args.root)
    examples = (
        load_nextqa(root, args.split)
        if args.dataset == "nextqa"
        else load_egoschema(root, args.annotations)
    )
    selected = examples[args.offset : args.offset + args.limit]
    if not selected:
        raise RuntimeError("no examples resolved from the requested range")

    if args.dry_run:
        for example in selected:
            info = await probe(example.video_path)
            payload = asdict(example)
            payload["video_path"] = str(example.video_path)
            payload["duration_seconds"] = info.duration_seconds
            print(json.dumps(payload, ensure_ascii=False))
        return

    results: list[dict[str, Any]] = []
    if args.standalone:
        for example in selected:
            try:
                result = await evaluate_standalone(example, args)
            except Exception as exc:
                result = {
                    "dataset": example.dataset,
                    "example_id": example.example_id,
                    "status": "runner_error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "correct": False,
                }
            results.append(result)
            print(json.dumps(result, ensure_ascii=False))
    else:
        async with httpx.AsyncClient(
            base_url=args.api_url.rstrip("/"), timeout=httpx.Timeout(args.timeout)
        ) as client:
            for example in selected:
                try:
                    result = await evaluate_example(client, example, args.timeout)
                except Exception as exc:
                    result = {
                        "dataset": example.dataset,
                        "example_id": example.example_id,
                        "status": "runner_error",
                        "error": f"{type(exc).__name__}: {exc}",
                        "correct": False,
                    }
                results.append(result)
                print(json.dumps(result, ensure_ascii=False))

    summary = {
        "dataset": args.dataset,
        "examples": len(results),
        "completed": sum(item["status"] == "succeeded" for item in results),
        "correct": sum(bool(item["correct"]) for item in results),
        "accuracy": sum(bool(item["correct"]) for item in results) / len(results),
    }
    print(json.dumps({"summary": summary}))
    if args.output:
        output_path = Path(args.output)
        await asyncio.to_thread(output_path.parent.mkdir, parents=True, exist_ok=True)
        output_text = (
            "\n".join(json.dumps(item, ensure_ascii=False) for item in results) + "\n"
        )
        await asyncio.to_thread(output_path.write_text, output_text, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run VideoMASys on local QA benchmarks")
    parser.add_argument("--dataset", choices=["nextqa", "egoschema"], required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--split", default="val", help="NExT-QA CSV split")
    parser.add_argument(
        "--annotations",
        default="subset_first10_with_gt.json",
        help="EgoSchema annotation filename relative to root",
    )
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--standalone", action="store_true")
    parser.add_argument("--vlm-url", default="http://localhost:8001/v1")
    parser.add_argument("--vlm-model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--artifact-root", default="./artifacts/evaluation")
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--output")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def run() -> None:
    asyncio.run(async_main(build_parser().parse_args()))


if __name__ == "__main__":
    run()
