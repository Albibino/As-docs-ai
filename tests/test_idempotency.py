from __future__ import annotations

import os
import uuid

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import SecretStr
from redis.asyncio import Redis

from tcc_docs_ai.app import create_app
from tcc_docs_ai.config import Settings
from tcc_docs_ai.errors import IdempotencyConflictError
from tcc_docs_ai.idempotency import InMemoryIdempotencyStore, RedisIdempotencyStore


async def test_in_memory_store_requires_the_active_lease_to_mutate_a_record() -> None:
    store = InMemoryIdempotencyStore()
    started = await store.start("docs-app", "key", "hash")
    assert started.lease_token

    await store.abandon("docs-app", "key", "stale-lease")
    await store.append("docs-app", "key", started.lease_token, b"event")
    await store.complete("docs-app", "key", started.lease_token)

    replay = await store.start("docs-app", "key", "hash")
    assert replay.lease_token is None
    assert replay.replay is not None
    assert replay.replay.events == [b"event"]


async def test_redis_store_shares_atomic_replay_between_replicas() -> None:
    redis_url = os.getenv("REDIS_URL_TEST")
    if not redis_url:
        pytest.skip("REDIS_URL_TEST is required for the Redis integration test")

    first_client = Redis.from_url(redis_url, decode_responses=False)
    second_client = Redis.from_url(redis_url, decode_responses=False)
    prefix = f"tcc-docs-ai:test:idempotency:{uuid.uuid4().hex}"
    first_replica = RedisIdempotencyStore(
        first_client,
        key_prefix=prefix,
        in_progress_ttl_seconds=30,
        completed_ttl_seconds=60,
    )
    second_replica = RedisIdempotencyStore(
        second_client,
        key_prefix=prefix,
        in_progress_ttl_seconds=30,
        completed_ttl_seconds=60,
    )

    try:
        started = await first_replica.start("docs-app", "same-key", "request-hash")
        assert started.lease_token

        with pytest.raises(IdempotencyConflictError, match="in progress"):
            await second_replica.start("docs-app", "same-key", "request-hash")

        # Um worker antigo não pode apagar a posse da réplica ativa.
        await second_replica.abandon("docs-app", "same-key", "stale-lease")
        await first_replica.append(
            "docs-app", "same-key", started.lease_token, b"event: done\ndata: {}\n\n"
        )
        await first_replica.complete("docs-app", "same-key", started.lease_token)

        replay = await second_replica.start("docs-app", "same-key", "request-hash")
        assert replay.lease_token is None
        assert replay.replay is not None
        assert replay.replay.events == [b"event: done\ndata: {}\n\n"]

        with pytest.raises(IdempotencyConflictError, match="different request"):
            await second_replica.start("docs-app", "same-key", "other-hash")

        expired = await first_replica.start("docs-app", "expired-key", "request-hash")
        assert expired.lease_token
        expired_keys = [key async for key in first_client.scan_iter(match=f"{prefix}:*")]
        await first_client.delete(*expired_keys)
        replacement = await second_replica.start("docs-app", "expired-key", "request-hash")
        assert replacement.lease_token

        # A finalização tardia do lease expirado não remove o novo dono.
        await first_replica.abandon("docs-app", "expired-key", expired.lease_token)
        await second_replica.append(
            "docs-app", "expired-key", replacement.lease_token, b"new-owner"
        )
        await second_replica.complete("docs-app", "expired-key", replacement.lease_token)
    finally:
        keys = [key async for key in first_client.scan_iter(match=f"{prefix}:*")]
        if keys:
            await first_client.delete(*keys)
        await first_client.aclose()
        await second_client.aclose()


async def test_readiness_checks_the_configured_redis() -> None:
    redis_url = os.getenv("REDIS_URL_TEST")
    if not redis_url:
        pytest.skip("REDIS_URL_TEST is required for the Redis integration test")

    settings = Settings(
        openai_api_key=SecretStr("test-key"),
        openai_model="test-model",
        docs_ai_hmac_secret=SecretStr("test-secret-with-at-least-32-bytes"),
        redis_url=SecretStr(redis_url),
    )
    app = create_app(settings=settings)
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/health/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}
