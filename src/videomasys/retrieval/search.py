from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from videomasys.domain.schemas import NodeKind
from videomasys.model_gateway.embeddings import EmbeddingGateway
from videomasys.storage.models import Artifact, VideoNode


@dataclass(frozen=True)
class RetrievalHit:
    node: VideoNode
    score: float


class VideoRetriever:
    def __init__(self, embedder: EmbeddingGateway) -> None:
        self.embedder = embedder

    async def search(
        self,
        session: AsyncSession,
        *,
        video_id: UUID,
        queries: list[str],
        limit: int,
    ) -> list[RetrievalHit]:
        vectors = await self.embedder.embed(queries)
        by_id: dict[UUID, RetrievalHit] = {}
        for query, vector in zip(queries, vectors, strict=True):
            distance = VideoNode.embedding.cosine_distance(vector)
            statement = (
                select(VideoNode, distance.label("distance"))
                .where(
                    VideoNode.video_id == video_id,
                    VideoNode.kind.in_([NodeKind.EVENT.value, NodeKind.SCENE.value]),
                    VideoNode.embedding.is_not(None),
                )
                .order_by(distance)
                .limit(limit * 4)
            )
            rows = (await session.execute(statement)).all()
            query_tokens = {token.casefold() for token in query.split() if len(token) > 2}
            for node, raw_distance in rows:
                semantic = max(0.0, 1.0 - float(raw_distance))
                caption_tokens = set(node.caption.casefold().split())
                lexical = len(query_tokens & caption_tokens) / max(len(query_tokens), 1)
                score = 0.85 * semantic + 0.15 * lexical
                previous = by_id.get(node.id)
                if previous is None or score > previous.score:
                    by_id[node.id] = RetrievalHit(node=node, score=score)
        ranked = sorted(by_id.values(), key=lambda item: item.score, reverse=True)
        # Event and scene nodes may share the same interval and caption. Deduplicate
        # before applying the caller's beam; otherwise limit=2 can still yield only
        # one distinct moment after event-to-scene expansion.
        by_interval: dict[tuple[float, float], RetrievalHit] = {}
        for hit in ranked:
            key = (hit.node.start_seconds, hit.node.end_seconds)
            current = by_interval.get(key)
            if current is None or (
                hit.score == current.score
                and hit.node.kind == NodeKind.SCENE.value
                and current.node.kind != NodeKind.SCENE.value
            ):
                by_interval[key] = hit
        return list(by_interval.values())[:limit]

    async def expand_to_scenes(
        self,
        session: AsyncSession,
        *,
        hits: list[RetrievalHit],
        question: str,
        limit: int,
    ) -> list[RetrievalHit]:
        vector = (await self.embedder.embed([question]))[0]
        selected: dict[UUID, RetrievalHit] = {}
        for hit in hits:
            if hit.node.kind == NodeKind.SCENE.value:
                selected[hit.node.id] = hit
                continue
            distance = VideoNode.embedding.cosine_distance(vector)
            statement = (
                select(VideoNode, distance.label("distance"))
                .where(
                    VideoNode.parent_id == hit.node.id,
                    VideoNode.kind == NodeKind.SCENE.value,
                )
                .order_by(distance)
                .limit(2)
            )
            for node, raw_distance in (await session.execute(statement)).all():
                child_score = 0.6 * hit.score + 0.4 * max(0.0, 1.0 - float(raw_distance))
                previous = selected.get(node.id)
                if previous is None or child_score > previous.score:
                    selected[node.id] = RetrievalHit(node=node, score=child_score)
        return sorted(selected.values(), key=lambda item: item.score, reverse=True)[:limit]

    async def temporal_neighbors(
        self,
        session: AsyncSession,
        *,
        hits: list[RetrievalHit],
        direction: str,
        horizon_seconds: float,
        limit: int,
    ) -> list[RetrievalHit]:
        """Resolve bounded neighboring scenes without sending additional video to a model."""
        neighbors: dict[UUID, RetrievalHit] = {}
        # Match StructuredVP relation semantics: relate from the highest-ranked
        # segment, while locate may retain a wider recall beam.
        for hit in hits[:1]:
            statement = select(VideoNode).where(
                VideoNode.video_id == hit.node.video_id,
                VideoNode.kind == NodeKind.SCENE.value,
                VideoNode.id != hit.node.id,
            )
            if direction == "after":
                statement = statement.where(
                    VideoNode.start_seconds >= hit.node.end_seconds - 0.001,
                    VideoNode.start_seconds <= hit.node.end_seconds + horizon_seconds,
                ).order_by(VideoNode.start_seconds)
            elif direction == "before":
                statement = statement.where(
                    VideoNode.end_seconds <= hit.node.start_seconds + 0.001,
                    VideoNode.end_seconds >= hit.node.start_seconds - horizon_seconds,
                ).order_by(VideoNode.end_seconds.desc())
            else:
                statement = statement.where(
                    VideoNode.start_seconds <= hit.node.end_seconds + horizon_seconds,
                    VideoNode.end_seconds >= hit.node.start_seconds - horizon_seconds,
                ).order_by(VideoNode.start_seconds)
            candidates = list(await session.scalars(statement.limit(limit)))
            for candidate in candidates:
                gap = min(
                    abs(candidate.start_seconds - hit.node.end_seconds),
                    abs(hit.node.start_seconds - candidate.end_seconds),
                )
                proximity = max(0.0, 1.0 - gap / max(horizon_seconds, 1.0))
                score = 0.65 * hit.score + 0.35 * proximity
                previous = neighbors.get(candidate.id)
                if previous is None or score > previous.score:
                    neighbors[candidate.id] = RetrievalHit(node=candidate, score=score)
        return sorted(neighbors.values(), key=lambda item: item.score, reverse=True)[:limit]

    async def frame_paths(self, session: AsyncSession, node_id: UUID) -> list[Path]:
        statement = (
            select(Artifact)
            .where(
                Artifact.node_id == node_id,
                Artifact.kind.in_(["qa_evidence_frame", "representative_frame"]),
            )
            .order_by(Artifact.artifact_metadata["ordinal"].as_integer())
        )
        artifacts = (await session.scalars(statement)).all()
        evidence_frames = [item for item in artifacts if item.kind == "qa_evidence_frame"]
        selected = evidence_frames or list(artifacts)
        return [Path(artifact.path) for artifact in selected]
