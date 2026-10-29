from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

from sqlalchemy import delete

from videomasys.media.ffmpeg import probe
from videomasys.model_gateway.embeddings import HashEmbeddingGateway
from videomasys.settings import get_settings
from videomasys.storage.db import session_factory
from videomasys.storage.models import Artifact, Evidence, Task, TaskStep, Video, VideoNode

DEMO_VIDEO_ID = UUID("dd621b89-04b1-5fa2-980f-c60d2acb4777")
DEMO_TASK_ID = UUID("b90a77ff-f6fc-5db5-bd65-863677e33dbe")
SOURCE_VIDEO = Path(
    "/scratch/Projects/3DInfant/SVP_data/EgoSchema/competitions/egoschema-public/"
    "videos/videos/0074f737-11cb-497d-8d07-77c3a8127391.mp4"
)
CACHE_ROOT = Path("artifacts/evaluation/standalone/7ffeae1731bd5b31a16341dcf35b13cf")
RESULT_PATH = Path("artifacts/evaluation/egoschema_results_warm_tuned.jsonl")


async def seed() -> None:
    settings = get_settings()
    await asyncio.to_thread(settings.ensure_directories)
    target_video, digest, benchmark, analyses, scene_frames = await asyncio.to_thread(
        _prepare_demo_data, settings.artifact_root
    )
    media = await probe(target_video)
    embedder = HashEmbeddingGateway(settings.embedding_dimension)
    caption_embeddings = await embedder.embed([analysis["caption"] for analysis in analyses])
    await embedder.close()

    async with session_factory() as session:
        existing_task = await session.get(Task, DEMO_TASK_ID)
        if existing_task is not None:
            await session.execute(delete(Task).where(Task.id == DEMO_TASK_ID))
        existing_video = await session.get(Video, DEMO_VIDEO_ID)
        if existing_video is not None:
            await session.execute(delete(Video).where(Video.id == DEMO_VIDEO_ID))
        await session.commit()

        video = Video(
            id=DEMO_VIDEO_ID,
            filename="EgoSchema · Kitchen cleanup.mp4",
            source_path=str(target_video.resolve()),
            content_hash=digest,
            status="ready",
            duration_seconds=media.duration_seconds,
            width=media.width,
            height=media.height,
            media_info={"source": "EgoSchema", "demo": True},
        )
        session.add(video)
        nodes: dict[float, VideoNode] = {}
        for index in range(9):
            start = float(index * 20)
            end = float((index + 1) * 20)
            analysis = analyses[index]
            node = VideoNode(
                id=uuid5(NAMESPACE_URL, f"videomasys-demo#{start:.3f}-{end:.3f}"),
                video_id=DEMO_VIDEO_ID,
                kind="scene",
                ordinal=index,
                start_seconds=start,
                end_seconds=end,
                caption=analysis["caption"],
                node_metadata={
                    "visible_entities": analysis.get("visible_entities", []),
                    "actions": analysis.get("actions", []),
                    "searchable_tags": analysis.get("searchable_tags", []),
                },
                embedding=caption_embeddings[index],
            )
            nodes[start] = node
            session.add(node)
            for frame_index, frame_path in enumerate(scene_frames[index]):
                session.add(
                    Artifact(
                        video_id=DEMO_VIDEO_ID,
                        node_id=node.id,
                        kind="representative_frame",
                        path=str(frame_path.resolve()),
                        artifact_metadata={"ordinal": frame_index, "demo": True},
                    )
                )
        await session.flush()

        result = benchmark["result"] | {
            "verified": False,
            "corrected_by_verifier": False,
            "verification_issues": [
                "Verifier rejected the candidate without an actionable reason; "
                "the grounded candidate was retained."
            ],
            "citations": [
                {
                    "node_id": str(nodes[float(item["start_seconds"])].id),
                    "start_seconds": item["start_seconds"],
                    "end_seconds": item["end_seconds"],
                }
                for item in benchmark["evidence"]
            ],
        }
        task = Task(
            id=DEMO_TASK_ID,
            video_id=DEMO_VIDEO_ID,
            task_type="qa",
            status="succeeded",
            request={
                "question": benchmark["question"],
                "options": [
                    "O1: C is cooking.",
                    "O2: C is doing laundry.",
                    "O3: C is cleaning the kitchen.",
                    "O4: C is cleaning dishes.",
                    "O5: C is cleaning the bathroom.",
                ],
            },
            plan=benchmark["plan"],
            result=result,
        )
        session.add(task)
        await session.flush()

        for item in benchmark["evidence"]:
            start = float(item["start_seconds"])
            node = nodes[start]
            scene_index = round(start / 20)
            frame_paths = scene_frames[scene_index]
            session.add(
                Evidence(
                    task_id=DEMO_TASK_ID,
                    node_id=node.id,
                    start_seconds=start,
                    end_seconds=float(item["end_seconds"]),
                    observation=item["observation"],
                    confidence=float(item["confidence"]),
                    frame_paths=[str(path.resolve()) for path in frame_paths],
                )
            )

        started = datetime.now(UTC) - timedelta(seconds=48.163)
        steps = [
            ("plan", 2110),
            ("retrieve", 23),
            ("inspect_evidence", 38240),
            ("compose_answer", 4360),
            ("verify_answer", 3430),
        ]
        cursor = started
        for name, latency_ms in steps:
            finished = cursor + timedelta(milliseconds=latency_ms)
            session.add(
                TaskStep(
                    task_id=DEMO_TASK_ID,
                    name=name,
                    status="succeeded",
                    started_at=cursor,
                    finished_at=finished,
                    latency_ms=latency_ms,
                    details={"demo": True},
                )
            )
            cursor = finished
        await session.commit()

    print(f"Seeded demo video {DEMO_VIDEO_ID} and task {DEMO_TASK_ID}")


def _prepare_demo_data(
    artifact_root: Path,
) -> tuple[Path, str, dict[str, Any], list[dict[str, Any]], list[list[Path]]]:
    if not SOURCE_VIDEO.is_file() or not CACHE_ROOT.is_dir() or not RESULT_PATH.is_file():
        raise RuntimeError("EgoSchema source video or cached evaluation artifacts are missing")
    target_video = artifact_root / "uploads" / "demo-egoschema.mp4"
    if not target_video.is_file():
        shutil.copyfile(SOURCE_VIDEO, target_video)
    benchmark = json.loads(RESULT_PATH.read_text(encoding="utf-8").splitlines()[0])
    analyses = [
        json.loads(
            (CACHE_ROOT / f"scene_{index:04d}" / "analysis.json").read_text(encoding="utf-8")
        )
        for index in range(9)
    ]
    scene_frames = [
        sorted((CACHE_ROOT / f"scene_{index:04d}").glob("frame_*.jpg"))
        for index in range(9)
    ]
    return target_video, _sha256(target_video), benchmark, analyses, scene_frames


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    asyncio.run(seed())
