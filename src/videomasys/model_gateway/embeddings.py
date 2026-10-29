from __future__ import annotations

import hashlib
import math
from typing import Protocol

import httpx

from videomasys.settings import Settings


class EmbeddingGateway(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...

    async def close(self) -> None: ...


class OpenAIEmbeddingGateway:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client = httpx.AsyncClient(
            base_url=settings.embedding_base_url.rstrip("/") + "/",
            headers={"Authorization": f"Bearer {settings.embedding_api_key}"},
            timeout=httpx.Timeout(120.0),
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = await self._client.post(
            "embeddings", json={"model": self.settings.embedding_model, "input": texts}
        )
        response.raise_for_status()
        ordered = sorted(response.json()["data"], key=lambda item: item["index"])
        vectors = [item["embedding"] for item in ordered]
        expected = self.settings.embedding_dimension
        if any(len(vector) != expected for vector in vectors):
            raise ValueError(f"embedding endpoint must return {expected}-dimensional vectors")
        return vectors

    async def close(self) -> None:
        await self._client.aclose()


class HashEmbeddingGateway:
    """Deterministic plumbing fallback. It intentionally has no semantic meaning."""

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        tokens = text.casefold().split()
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            vector[index] += -1.0 if digest[4] & 1 else 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    async def close(self) -> None:
        return None


def build_embedding_gateway(settings: Settings) -> EmbeddingGateway:
    if settings.embedding_provider.casefold() == "hash":
        return HashEmbeddingGateway(settings.embedding_dimension)
    if settings.embedding_provider.casefold() == "openai":
        return OpenAIEmbeddingGateway(settings)
    raise ValueError(f"unsupported embedding provider: {settings.embedding_provider}")

