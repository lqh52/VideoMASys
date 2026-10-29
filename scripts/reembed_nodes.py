from __future__ import annotations

import asyncio

from sqlalchemy import select

from videomasys.model_gateway.embeddings import build_embedding_gateway
from videomasys.settings import get_settings
from videomasys.storage.db import session_factory
from videomasys.storage.models import VideoNode


async def reembed(batch_size: int = 32) -> None:
    settings = get_settings()
    gateway = build_embedding_gateway(settings)
    try:
        async with session_factory() as session:
            nodes = list(
                await session.scalars(
                    select(VideoNode)
                    .where(VideoNode.kind.in_(["scene", "event"]))
                    .order_by(VideoNode.video_id, VideoNode.start_seconds)
                )
            )
            for offset in range(0, len(nodes), batch_size):
                batch = nodes[offset : offset + batch_size]
                vectors = await gateway.embed([node.caption for node in batch])
                for node, vector in zip(batch, vectors, strict=True):
                    node.embedding = vector
                await session.commit()
                print(f"Embedded {min(offset + len(batch), len(nodes))}/{len(nodes)} nodes")
    finally:
        await gateway.close()


if __name__ == "__main__":
    asyncio.run(reembed())
