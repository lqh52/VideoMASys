import math

from videomasys.model_gateway.embeddings import HashEmbeddingGateway


async def test_hash_embedder_is_deterministic_and_normalized() -> None:
    gateway = HashEmbeddingGateway(32)
    first, second = await gateway.embed(["A red car", "A red car"])
    assert first == second
    assert math.isclose(sum(value * value for value in first), 1.0)


async def test_hash_embedder_has_requested_dimension() -> None:
    gateway = HashEmbeddingGateway(17)
    result = await gateway.embed(["scene"])
    assert len(result[0]) == 17

