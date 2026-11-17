"""Índice versionado contra um Redis 8 de verdade (Vector Sets).

Roda com `AI_TEST_REDIS_URL=redis://localhost:6379/15`. Usa um prefixo único
por execução e apaga só as chaves desse prefixo no final.
"""

from __future__ import annotations

import os
import uuid

import pytest
from redis.asyncio import Redis

from tcc_docs_ai.indexing import IndexingService
from tcc_docs_ai.provider import HashingEmbeddingProvider
from tcc_docs_ai.schemas import IndexStatus
from tcc_docs_ai.vectorstore import RedisVectorIndex
from tests.rag_fixtures import POLICY_V1, POLICY_V2, page_version

REDIS_URL = os.getenv("AI_TEST_REDIS_URL")
pytestmark = pytest.mark.skipif(not REDIS_URL, reason="AI_TEST_REDIS_URL is required")


@pytest.fixture
async def redis_index():
    client = Redis.from_url(REDIS_URL or "", decode_responses=False)
    prefix = f"tcc-docs-ai-test-{uuid.uuid4().hex}"
    embedder = HashingEmbeddingProvider(dimensions=128)
    index = RedisVectorIndex(client, key_prefix=prefix, model=embedder.model)
    service = IndexingService(
        index=index,
        embedder=embedder,
        model=embedder.model,
        chunk_max_chars=1_600,
        max_embedding_inputs=100,
        max_text_bytes=2_000_000,
    )
    try:
        yield client, index, service, embedder
    finally:
        keys = [key async for key in client.scan_iter(match=f"{prefix}:*")]
        if keys:
            await client.delete(*keys)
        await client.aclose()


async def test_redis_index_is_append_only_and_versioned(redis_index) -> None:
    client, index, service, embedder = redis_index

    first = await service.index("w1", [page_version("p1", 1, "Férias", POLICY_V1)])
    cardinality_v1 = await client.execute_command("VCARD", f"{index._root('w1')}:vectors")  # noqa: SLF001
    repeat = await service.index("w1", [page_version("p1", 1, "Férias", POLICY_V1)])
    cardinality_repeat = await client.execute_command(
        "VCARD",
        f"{index._root('w1')}:vectors",  # noqa: SLF001
    )
    second = await service.index("w1", [page_version("p1", 2, "Férias", POLICY_V2)])
    older = await service.index("w1", [page_version("p1", 1, "Férias", POLICY_V1)])

    assert first[0].status is IndexStatus.INDEXED
    assert repeat[0].status is IndexStatus.ALREADY_INDEXED
    assert cardinality_repeat == cardinality_v1  # nada foi reescrito
    assert second[0].status is IndexStatus.INDEXED
    assert older[0].status is IndexStatus.ALREADY_INDEXED

    state = (await index.page_states("w1", ["p1"]))["p1"]
    assert state.latest_version == 2
    assert state.versions == (1, 2)

    query = (await embedder.embed(["prazo do pedido de férias"])).embeddings[0]
    hits = await index.search("w1", query, count=10)
    assert {hit.chunk.version for hit in hits} == {1, 2}
    assert all(0.0 <= hit.score <= 1.0 for hit in hits)
    assert hits[0].chunk.page_id == "p1"


async def test_redis_index_is_isolated_by_tenant(redis_index) -> None:
    _, index, service, embedder = redis_index
    await service.index("w1", [page_version("p1", 1, "Férias", POLICY_V1)])
    query = (await embedder.embed(["férias"])).embeddings[0]
    assert await index.search("w2", query, count=5) == []
    assert await index.page_states("w2", ["p1"]) == {}
