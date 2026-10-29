from __future__ import annotations

import hashlib
import os
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated
from uuid import UUID

import uvicorn
from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from videomasys.domain.schemas import (
    EvidenceRead,
    QATaskCreate,
    TaskRead,
    TaskStatus,
    TaskStepRead,
    VideoRead,
    VideoStatus,
)
from videomasys.settings import get_settings
from videomasys.storage.db import database_ready, get_session, init_db
from videomasys.storage.models import Evidence, Task, TaskStep, Video, VideoNode

settings = get_settings()
SessionDep = Annotated[AsyncSession, Depends(get_session)]


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    settings.ensure_directories()
    if settings.auto_create_schema:
        await init_db()
    yield


app = FastAPI(
    title="VideoMASys",
    version="0.1.0",
    description="QA-first cache-centric video understanding prototype",
    lifespan=lifespan,
)
web_root = Path(__file__).resolve().parent.parent / "web"
app.mount("/static", StaticFiles(directory=web_root), name="static")


@app.get("/", include_in_schema=False)
async def web_app() -> FileResponse:
    return FileResponse(web_root / "index.html")


@app.get("/healthz")
async def health() -> dict[str, object]:
    ready = await database_ready()
    return {"status": "ok" if ready else "degraded", "database": ready}


_BROWSER_DEMO_EXAMPLES = [
    {
        "id": "nextqa_gift_reason",
        "dataset": "NExT-QA",
        "label": "Gift: infer the boy's purpose",
        "video": "nextqa",
        "question": "Why did the boy pick up one present and move to the sofa?",
        "options": [
            "Share with the girl",
            "Approach the lady sitting there",
            "Unwrap it",
            "Play with the toy train",
            "Gesture something",
        ],
    },
    {
        "id": "nextqa_after_woman",
        "dataset": "NExT-QA",
        "label": "Gift: action after approaching woman",
        "video": "nextqa",
        "question": "What did the boy do after he walked toward the woman with a present?",
        "options": [
            "Walk to the sofa",
            "Go away",
            "Look at the camera",
            "Place it around the house",
            "Stand behind the wall",
        ],
    },
    {
        "id": "nextqa_return_tree",
        "dataset": "NExT-QA",
        "label": "Gift: reason for returning to tree",
        "video": "nextqa",
        "question": "Why did the boy return to the Christmas tree after unwrapping one present?",
        "options": [
            "Blow the candles",
            "Decorate the tree",
            "Continue unwrapping presents",
            "Look at the tree",
            "Put the wrapper aside",
        ],
    },
    {
        "id": "nextqa_torn_paper",
        "dataset": "NExT-QA",
        "label": "Gift: explain torn paper at the end",
        "video": "nextqa",
        "question": "Why was there torn wrapping paper near the sofa at the end?",
        "options": [
            "A grey cat pulled it down",
            "The lady put it there",
            "The boy threw it there",
            "The man tore it down",
            "A baby was playing",
        ],
    },
    {
        "id": "nextqa_make_space",
        "dataset": "NExT-QA",
        "label": "Gift: reason for putting present down",
        "video": "nextqa",
        "question": "Why did the boy place the unwrapped present on the ground?",
        "options": [
            "Make space on the sofa for the next present",
            "Let the dog play with it",
            "Watch his steps",
            "Keep the toy",
            "Teach a baby how to play",
        ],
    },
    {
        "id": "egoschema_dishes",
        "dataset": "EgoSchema",
        "label": "Kitchen: infer the overall objective",
        "video": "egoschema",
        "question": (
            "Taking into account all actions performed by C, what is the primary "
            "objective of the video?"
        ),
        "options": [
            "C is cooking",
            "C is doing laundry",
            "C is cleaning the kitchen",
            "C is cleaning dishes",
            "C is cleaning the bathroom",
        ],
    },
]

_BROWSER_DEMO_VIDEOS = {
    "nextqa": {
        "filename": "NExT-QA 4882821564 · Christmas presents",
        "source": (
            "49dd433b47f9e3c88c272332d5ae739fec1d9b4d96f5b93b8e648d2b45428b41.mp4"
        ),
        "duration": 90.023756,
        "width": 640,
        "height": 480,
        "segments": {
            0: (0.0, 20.0),
            1: (20.0, 40.0),
            2: (40.0, 60.0),
            3: (60.0, 80.0),
            4: (80.0, 90.023756),
        },
    },
    "egoschema": {
        "filename": "EgoSchema 0074f737 · Dishwashing routine",
        "source": "demo-egoschema.mp4",
        "duration": 180.0,
        "width": 640,
        "height": 360,
        "segments": {index: (index * 20.0, (index + 1) * 20.0) for index in range(9)},
    },
}

@app.get("/v1/browser-demo/examples", include_in_schema=False)
async def browser_demo_examples() -> dict[str, object]:
    return {
        "default": _BROWSER_DEMO_EXAMPLES[0]["id"],
        "videos": [
            {
                "key": key,
                "label": video["filename"],
                "dataset": next(
                    example["dataset"]
                    for example in _BROWSER_DEMO_EXAMPLES
                    if example["video"] == key
                ),
                "question_count": sum(
                    example["video"] == key for example in _BROWSER_DEMO_EXAMPLES
                ),
            }
            for key, video in _BROWSER_DEMO_VIDEOS.items()
        ],
        "examples": [
            {
                key: example[key]
                for key in ("id", "dataset", "label", "question", "video")
            }
            for example in _BROWSER_DEMO_EXAMPLES
        ],
    }


@app.get("/v1/browser-demo", include_in_schema=False)
async def browser_demo(
    example: str = Query(default="nextqa_gift_reason"),
) -> dict[str, object]:
    """Serve real media and benchmark questions; inference runs in the browser."""
    selected = next(
        (item for item in _BROWSER_DEMO_EXAMPLES if item["id"] == example),
        None,
    )
    if selected is None:
        raise HTTPException(status_code=404, detail="browser demo example not found")
    video_key = str(selected["video"])
    video = _BROWSER_DEMO_VIDEOS[video_key]
    source = settings.artifact_root / "uploads" / str(video["source"])
    if not source.is_file():
        raise HTTPException(status_code=404, detail="browser demo video is unavailable")

    task_id = uuid.uuid5(uuid.NAMESPACE_URL, f"videomasys-browser-demo-{example}")
    video_id = uuid.uuid5(uuid.NAMESPACE_URL, f"videomasys-browser-demo-{video_key}")
    options = list(selected["options"])
    return {
        "example_id": example,
        "video": {
            "id": str(video_id),
            "filename": video["filename"],
            "content_hash": f"browser-demo-{video_key}",
            "status": "ready",
            "duration_seconds": video["duration"],
            "width": video["width"],
            "height": video["height"],
            "node_count": len(video["segments"]),
            "content_url": f"/v1/browser-demo/assets/{video_key}/content",
        },
        "task": {
            "id": str(task_id),
            "video_id": str(video_id),
            "task_type": "qa",
            "status": "idle",
            "request": {"question": selected["question"], "options": options},
            "plan": None,
            "result": None,
            "evidence": [],
            "steps": [],
        },
    }


def _browser_demo_frame_paths(video_key: str, segment: int) -> list[Path]:
    if video_key == "nextqa" and segment in range(5):
        directory = (
            settings.artifact_root
            / "cache"
            / "ef584765-bd06-41df-a4e1-5115e5848ea3"
            / f"scene_{segment:04d}"
        )
    elif video_key == "egoschema" and segment in range(9):
        directory = (
            settings.artifact_root
            / "evaluation"
            / "standalone"
            / "7ffeae1731bd5b31a16341dcf35b13cf"
            / f"scene_{segment:04d}"
        )
    else:
        raise HTTPException(status_code=404, detail="browser demo segment not found")
    matches = sorted(directory.glob("frame_*.jpg"))
    if not matches:
        raise HTTPException(status_code=404, detail="browser demo frames are unavailable")
    return matches


@app.get("/v1/browser-demo/assets/{video_key}/scenes", include_in_schema=False)
async def browser_demo_scenes(video_key: str) -> dict[str, object]:
    video = _BROWSER_DEMO_VIDEOS.get(video_key)
    if video is None:
        raise HTTPException(status_code=404, detail="browser demo video not found")
    scenes = []
    for segment, (start, end) in video["segments"].items():
        frames = _browser_demo_frame_paths(video_key, int(segment))
        scenes.append(
            {
                "index": segment,
                "start_seconds": start,
                "end_seconds": end,
                "frame_urls": [
                    f"/v1/browser-demo/assets/{video_key}/frames/{segment}/{index}"
                    for index in range(len(frames))
                ],
            }
        )
    return {"video_key": video_key, "scenes": scenes}


@app.get(
    "/v1/browser-demo/assets/{video_key}/content",
    include_in_schema=False,
)
async def browser_demo_content(video_key: str) -> FileResponse:
    video = _BROWSER_DEMO_VIDEOS.get(video_key)
    if video is None:
        raise HTTPException(status_code=404, detail="browser demo video not found")
    path = settings.artifact_root / "uploads" / str(video["source"])
    return FileResponse(_safe_artifact_path(str(path)), content_disposition_type="inline")


@app.get(
    "/v1/browser-demo/assets/{video_key}/frames/{segment}/{frame_index}",
    include_in_schema=False,
)
async def browser_demo_frame(video_key: str, segment: int, frame_index: int) -> FileResponse:
    frames = _browser_demo_frame_paths(video_key, segment)
    if frame_index < 0 or frame_index >= len(frames):
        raise HTTPException(status_code=404, detail="browser demo frame not found")
    return FileResponse(_safe_artifact_path(str(frames[frame_index])))


@app.post("/v1/videos", response_model=VideoRead, status_code=status.HTTP_202_ACCEPTED)
async def upload_video(
    file: Annotated[UploadFile, File()], session: SessionDep
) -> VideoRead:
    filename = Path(file.filename or "video.bin").name
    temp_path = settings.artifact_root / "uploads" / f"upload-{uuid.uuid4()}.part"
    digest = hashlib.sha256()
    total = 0
    try:
        with temp_path.open("wb") as target:
            while chunk := await file.read(1024 * 1024):
                total += len(chunk)
                if total > settings.max_upload_bytes:
                    raise HTTPException(status_code=413, detail="video exceeds MAX_UPLOAD_BYTES")
                digest.update(chunk)
                target.write(chunk)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    finally:
        await file.close()

    content_hash = digest.hexdigest()
    existing = await session.scalar(select(Video).where(Video.content_hash == content_hash))
    if existing is not None:
        temp_path.unlink(missing_ok=True)
        return await _video_read(session, existing)

    suffix = Path(filename).suffix.casefold()[:16] or ".bin"
    final_path = settings.artifact_root / "uploads" / f"{content_hash}{suffix}"
    os.replace(temp_path, final_path)
    video = Video(
        filename=filename,
        source_path=str(final_path.resolve()),
        content_hash=content_hash,
        status=VideoStatus.QUEUED.value,
    )
    session.add(video)
    try:
        await session.commit()
    except Exception:
        await session.rollback()
        # Keep the content-addressed upload: a concurrent request may own the same path.
        raise
    await session.refresh(video)
    return await _video_read(session, video)


@app.get("/v1/videos", response_model=list[VideoRead])
async def list_videos(
    session: SessionDep, limit: int = Query(default=20, ge=1, le=100)
) -> list[VideoRead]:
    videos = list(
        await session.scalars(select(Video).order_by(Video.created_at.desc()).limit(limit))
    )
    return [await _video_read(session, video) for video in videos]


@app.get("/v1/videos/{video_id}", response_model=VideoRead)
async def get_video(video_id: UUID, session: SessionDep) -> VideoRead:
    video = await session.get(Video, video_id)
    if video is None:
        raise HTTPException(status_code=404, detail="video not found")
    return await _video_read(session, video)


@app.get("/v1/videos/{video_id}/content", include_in_schema=False)
async def stream_video(video_id: UUID, session: SessionDep) -> FileResponse:
    video = await session.get(Video, video_id)
    if video is None:
        raise HTTPException(status_code=404, detail="video not found")
    source_path = _safe_artifact_path(video.source_path)
    return FileResponse(source_path, filename=video.filename, content_disposition_type="inline")


@app.post("/v1/qa", response_model=TaskRead, status_code=status.HTTP_202_ACCEPTED)
async def create_qa_task(
    request: QATaskCreate, session: SessionDep
) -> TaskRead:
    video = await session.get(Video, request.video_id)
    if video is None:
        raise HTTPException(status_code=404, detail="video not found")
    if video.status != VideoStatus.READY.value:
        raise HTTPException(status_code=409, detail=f"video is {video.status}, not ready")
    task = Task(
        video_id=video.id,
        task_type="qa",
        status=TaskStatus.QUEUED.value,
        request={"question": request.question, "options": request.options},
    )
    session.add(task)
    await session.commit()
    await session.refresh(task)
    return _task_read(task, [], [])


@app.get("/v1/tasks", response_model=list[TaskRead])
async def list_tasks(
    session: SessionDep,
    limit: int = Query(default=20, ge=1, le=100),
    video_id: UUID | None = None,
) -> list[TaskRead]:
    statement = select(Task).options(selectinload(Task.evidence), selectinload(Task.steps))
    if video_id is not None:
        statement = statement.where(Task.video_id == video_id)
    statement = statement.order_by(Task.created_at.desc()).limit(limit)
    tasks = list(await session.scalars(statement))
    return [_task_read(task, task.evidence, task.steps) for task in tasks]


@app.get("/v1/tasks/{task_id}", response_model=TaskRead)
async def get_task(task_id: UUID, session: SessionDep) -> TaskRead:
    statement = (
        select(Task)
        .options(selectinload(Task.evidence), selectinload(Task.steps))
        .where(Task.id == task_id)
    )
    task = await session.scalar(statement)
    if task is None:
        raise HTTPException(status_code=404, detail="task not found")
    return _task_read(task, task.evidence, task.steps)


@app.get(
    "/v1/tasks/{task_id}/evidence/{node_id}/frames/{frame_index}",
    include_in_schema=False,
)
async def evidence_frame(
    task_id: UUID,
    node_id: UUID,
    frame_index: int,
    session: SessionDep,
) -> FileResponse:
    evidence = await session.scalar(
        select(Evidence).where(Evidence.task_id == task_id, Evidence.node_id == node_id)
    )
    if evidence is None or frame_index < 0 or frame_index >= len(evidence.frame_paths):
        raise HTTPException(status_code=404, detail="evidence frame not found")
    return FileResponse(_safe_artifact_path(evidence.frame_paths[frame_index]))


@app.post("/v1/tasks/{task_id}/cancel", response_model=TaskRead)
async def cancel_task(task_id: UUID, session: SessionDep) -> TaskRead:
    statement = (
        select(Task)
        .options(selectinload(Task.evidence), selectinload(Task.steps))
        .where(Task.id == task_id)
    )
    task = await session.scalar(statement)
    if task is None:
        raise HTTPException(status_code=404, detail="task not found")
    if task.status == TaskStatus.QUEUED.value:
        task.status = TaskStatus.CANCELLED.value
        await session.commit()
        await session.refresh(task)
    elif task.status == TaskStatus.PROCESSING.value:
        task.status = TaskStatus.CANCEL_REQUESTED.value
        await session.commit()
        await session.refresh(task)
    return _task_read(task, task.evidence, task.steps)


async def _video_read(session: AsyncSession, video: Video) -> VideoRead:
    count = await session.scalar(
        select(func.count(VideoNode.id)).where(VideoNode.video_id == video.id)
    )
    return VideoRead(
        id=video.id,
        filename=video.filename,
        content_hash=video.content_hash,
        status=video.status,
        duration_seconds=video.duration_seconds,
        width=video.width,
        height=video.height,
        error=video.error,
        created_at=video.created_at,
        updated_at=video.updated_at,
        node_count=count or 0,
    )


def _task_read(task: Task, evidence: list[Evidence], steps: list[TaskStep]) -> TaskRead:
    return TaskRead(
        id=task.id,
        video_id=task.video_id,
        task_type=task.task_type,
        status=task.status,
        request=task.request,
        plan=task.plan,
        result=task.result,
        error=task.error,
        evidence=[
            EvidenceRead(
                node_id=item.node_id,
                start_seconds=item.start_seconds,
                end_seconds=item.end_seconds,
                observation=item.observation,
                confidence=item.confidence,
                frame_paths=item.frame_paths,
            )
            for item in evidence
        ],
        steps=[
            TaskStepRead(
                name=item.name,
                status=item.status,
                started_at=item.started_at,
                finished_at=item.finished_at,
                latency_ms=item.latency_ms,
                details=item.details,
            )
            for item in sorted(steps, key=lambda value: value.started_at)
        ],
        created_at=task.created_at,
        updated_at=task.updated_at,
    )


def _safe_artifact_path(raw_path: str) -> Path:
    artifact_root = settings.artifact_root.resolve()
    path = Path(raw_path).resolve()
    if not path.is_relative_to(artifact_root) or not path.is_file():
        raise HTTPException(status_code=404, detail="artifact not found")
    return path


def run() -> None:
    uvicorn.run("videomasys.api.app:app", host=settings.api_host, port=settings.api_port)
