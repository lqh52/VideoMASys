from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from videomasys.agents.scene_analyst import SceneAnalyst
from videomasys.domain.schemas import NodeKind, SceneAnalysis, VideoStatus
from videomasys.media.ffmpeg import (
    Interval,
    detect_scene_intervals,
    extract_representative_frames,
    probe,
)
from videomasys.model_gateway.embeddings import EmbeddingGateway
from videomasys.settings import Settings
from videomasys.storage.models import Artifact, Video, VideoNode


@dataclass
class AnalyzedScene:
    interval: Interval
    frame_paths: list[Path]
    analysis: SceneAnalysis


class IngestionPipeline:
    def __init__(
        self,
        settings: Settings,
        scene_analyst: SceneAnalyst,
        embedder: EmbeddingGateway,
    ) -> None:
        self.settings = settings
        self.scene_analyst = scene_analyst
        self.embedder = embedder

    async def run(self, session: AsyncSession, video: Video) -> None:
        video_path = Path(video.source_path)
        info = await probe(video_path)
        intervals = await detect_scene_intervals(
            video_path,
            info.duration_seconds,
            self.settings.scene_threshold,
            self.settings.min_scene_seconds,
            self.settings.max_scene_seconds,
            self.settings.max_scenes_per_video,
        )
        cache_dir = self.settings.artifact_root / "cache" / str(video.id)
        media_semaphore = asyncio.Semaphore(self.settings.media_max_concurrency)

        async def analyze_one(index: int, interval: Interval) -> AnalyzedScene:
            async with media_semaphore:
                frames = await extract_representative_frames(
                    video_path,
                    interval,
                    cache_dir / f"scene_{index:04d}",
                    self.settings.frames_per_scene,
                )
            analysis = await self.scene_analyst.analyze(
                frames, interval.start_seconds, interval.end_seconds
            )
            return AnalyzedScene(interval=interval, frame_paths=frames, analysis=analysis)

        scenes = await asyncio.gather(
            *(analyze_one(index, interval) for index, interval in enumerate(intervals))
        )
        event_groups = self._group_events(scenes)
        caption_texts = [self._event_caption(group) for group in event_groups]
        caption_texts.extend(scene.analysis.caption for scene in scenes)
        embeddings = await self.embedder.embed(caption_texts)
        event_embeddings = embeddings[: len(event_groups)]
        scene_embeddings = embeddings[len(event_groups) :]

        # A retry replaces partial metadata atomically; content-addressed frame paths are reusable.
        await session.execute(delete(VideoNode).where(VideoNode.video_id == video.id))
        root = VideoNode(
            video_id=video.id,
            kind=NodeKind.VIDEO.value,
            ordinal=0,
            start_seconds=0.0,
            end_seconds=info.duration_seconds,
            caption=f"Video file {video.filename}",
            node_metadata={"source": "ffprobe"},
        )
        session.add(root)
        await session.flush()

        scene_embedding_index = 0
        for event_index, group in enumerate(event_groups):
            event_caption = caption_texts[event_index]
            event = VideoNode(
                video_id=video.id,
                parent_id=root.id,
                kind=NodeKind.EVENT.value,
                ordinal=event_index,
                start_seconds=group[0].interval.start_seconds,
                end_seconds=group[-1].interval.end_seconds,
                caption=event_caption,
                node_metadata={"scene_count": len(group), "source": "qwen3-vl-grouping"},
                embedding=event_embeddings[event_index],
            )
            session.add(event)
            await session.flush()
            for scene in group:
                node = VideoNode(
                    video_id=video.id,
                    parent_id=event.id,
                    kind=NodeKind.SCENE.value,
                    ordinal=scene_embedding_index,
                    start_seconds=scene.interval.start_seconds,
                    end_seconds=scene.interval.end_seconds,
                    caption=scene.analysis.caption,
                    node_metadata=scene.analysis.model_dump(mode="json"),
                    embedding=scene_embeddings[scene_embedding_index],
                )
                session.add(node)
                await session.flush()
                for frame_index, path in enumerate(scene.frame_paths):
                    session.add(
                        Artifact(
                            video_id=video.id,
                            node_id=node.id,
                            kind="representative_frame",
                            path=str(path.resolve()),
                            artifact_metadata={"ordinal": frame_index},
                        )
                    )
                scene_embedding_index += 1

        video.duration_seconds = info.duration_seconds
        video.width = info.width
        video.height = info.height
        video.media_info = info.raw
        video.status = VideoStatus.READY.value
        video.error = None
        await session.commit()

    def _group_events(self, scenes: list[AnalyzedScene]) -> list[list[AnalyzedScene]]:
        groups: list[list[AnalyzedScene]] = []
        current: list[AnalyzedScene] = []
        for scene in scenes:
            current.append(scene)
            analysis = scene.analysis
            model_boundary = analysis.boundary_confidence >= 0.6
            size_boundary = len(current) >= self.settings.max_scenes_per_event
            duration_boundary = (
                current[-1].interval.end_seconds - current[0].interval.start_seconds
                >= self.settings.max_event_seconds
            )
            if model_boundary or size_boundary or duration_boundary:
                groups.append(current)
                current = []
        if current:
            groups.append(current)
        return groups

    @staticmethod
    def _event_caption(group: list[AnalyzedScene]) -> str:
        captions = " ".join(scene.analysis.caption for scene in group)
        tags = list(
            dict.fromkeys(
                tag
                for scene in group
                for tag in (scene.analysis.searchable_tags + scene.analysis.visible_entities)
            )
        )
        suffix = f" Tags: {', '.join(tags[:24])}." if tags else ""
        return captions + suffix
