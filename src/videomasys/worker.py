from __future__ import annotations

import asyncio
import logging
import signal
import traceback
from uuid import UUID

from sqlalchemy import func, select, update

from videomasys.agents.qa import (
    AnswerComposer,
    AnswerVerifier,
    EvidenceInspector,
    MultipleChoiceJudge,
    ReasoningPlanner,
)
from videomasys.agents.scene_analyst import SceneAnalyst
from videomasys.agents.tool_sequence import ToolSequenceAgent
from videomasys.domain.schemas import TaskStatus, VideoStatus
from videomasys.ingestion.pipeline import IngestionPipeline
from videomasys.model_gateway import VLMGateway, build_embedding_gateway
from videomasys.retrieval import VideoRetriever
from videomasys.settings import get_settings
from videomasys.storage.db import init_db, session_factory
from videomasys.storage.models import Task, TaskStep, Video
from videomasys.tasks.qa import QAPipeline
from videomasys.tasks.qa.pipeline import TaskCancelled

logger = logging.getLogger("videomasys.worker")


class Worker:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.stop_event = asyncio.Event()
        self.vlm = VLMGateway(self.settings)
        self.embedder = build_embedding_gateway(self.settings)
        self.ingestion = IngestionPipeline(
            self.settings, SceneAnalyst(self.vlm), self.embedder
        )
        self.qa = QAPipeline(
            self.settings,
            ReasoningPlanner(self.vlm),
            ToolSequenceAgent(self.vlm),
            VideoRetriever(self.embedder),
            EvidenceInspector(self.vlm),
            MultipleChoiceJudge(self.vlm),
            AnswerComposer(self.vlm),
            AnswerVerifier(self.vlm),
        )

    async def serve(self) -> None:
        if self.settings.auto_create_schema:
            await init_db()
        logger.info("worker started")
        try:
            while not self.stop_event.is_set():
                video_id = await self._claim_video()
                if video_id is not None:
                    await self._process_video(video_id)
                    continue
                task_id = await self._claim_task()
                if task_id is not None:
                    await self._process_task(task_id)
                    continue
                try:
                    await asyncio.wait_for(
                        self.stop_event.wait(), timeout=self.settings.worker_poll_seconds
                    )
                except TimeoutError:
                    pass
        finally:
            await self.vlm.close()
            await self.embedder.close()
            logger.info("worker stopped")

    async def _claim_video(self) -> UUID | None:
        async with session_factory() as session:
            statement = (
                select(Video)
                .where(Video.status == VideoStatus.QUEUED.value)
                .order_by(Video.created_at)
                .with_for_update(skip_locked=True)
                .limit(1)
            )
            video = await session.scalar(statement)
            if video is None:
                return None
            video.status = VideoStatus.PROCESSING.value
            await session.commit()
            return video.id

    async def _claim_task(self) -> UUID | None:
        async with session_factory() as session:
            statement = (
                select(Task)
                .where(Task.status == TaskStatus.QUEUED.value, Task.task_type == "qa")
                .order_by(Task.created_at)
                .with_for_update(skip_locked=True)
                .limit(1)
            )
            task = await session.scalar(statement)
            if task is None:
                return None
            task.status = TaskStatus.PROCESSING.value
            task.attempt += 1
            await session.commit()
            return task.id

    async def _process_video(self, video_id: UUID) -> None:
        async with session_factory() as session:
            video = await session.get(Video, video_id)
            if video is None:
                return
            try:
                logger.info("ingesting video %s", video_id)
                await self.ingestion.run(session, video)
                logger.info("video %s ready", video_id)
            except Exception as exc:
                await session.rollback()
                video = await session.get(Video, video_id)
                if video is not None:
                    video.status = VideoStatus.FAILED.value
                    video.error = self._error_text(exc)
                    await session.commit()
                logger.exception("video %s failed", video_id)

    async def _process_task(self, task_id: UUID) -> None:
        async with session_factory() as session:
            task = await session.get(Task, task_id)
            if task is None:
                return
            video = await session.get(Video, task.video_id)
            try:
                if video is None or video.status != VideoStatus.READY.value:
                    raise RuntimeError("video cache is unavailable")
                logger.info("running QA task %s", task_id)
                await self.qa.run(session, task, video)
                logger.info("QA task %s succeeded", task_id)
            except TaskCancelled:
                logger.info("QA task %s cancelled", task_id)
            except Exception as exc:
                await session.rollback()
                task = await session.get(Task, task_id)
                if task is not None:
                    task.status = TaskStatus.FAILED.value
                    task.error = self._error_text(exc)
                    await session.execute(
                        update(TaskStep)
                        .where(TaskStep.task_id == task_id, TaskStep.status == "running")
                        .values(
                            status="failed",
                            finished_at=func.now(),
                            details={"error": task.error},
                        )
                    )
                    await session.commit()
                logger.exception("QA task %s failed", task_id)

    @staticmethod
    def _error_text(exc: Exception) -> str:
        message = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        return message[-4000:]


async def async_main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    worker = Worker()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, worker.stop_event.set)
    await worker.serve()


def run() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    run()
