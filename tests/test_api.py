from __future__ import annotations

from httpx import ASGITransport, AsyncClient
from pydantic import SecretStr

from tcc_docs_ai.app import create_app
from tcc_docs_ai.config import Settings
from tests.conftest import FakeProvider, post_signed


async def test_streams_validated_operations_and_usage(
    client: AsyncClient, payload: dict[str, object]
) -> None:
    response = await post_signed(client, payload)
    assert response.status_code == 200
    assert response.headers["x-accel-buffering"] == "no"
    assert "event: meta" in response.text
    assert "event: operation" in response.text
    assert '"blockId":"b1"' in response.text
    assert "event: usage" in response.text
    assert '"cachedInputTokens":1100' in response.text
    assert "event: done" in response.text


async def test_replays_completed_idempotent_stream_without_new_model_call(
    client: AsyncClient, payload: dict[str, object], provider: FakeProvider
) -> None:
    first = await post_signed(client, payload, key="same-key")
    second = await post_signed(client, payload, key="same-key")
    assert first.text == second.text
    assert provider.calls == 1


async def test_rejects_same_idempotency_key_for_different_body(
    client: AsyncClient, payload: dict[str, object]
) -> None:
    assert (await post_signed(client, payload, key="reused-key")).status_code == 200
    payload["request"] = "Outro pedido"
    response = await post_signed(client, payload, key="reused-key")
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "idempotency_conflict"


async def test_scopes_the_same_idempotency_key_by_tenant(
    client: AsyncClient, payload: dict[str, object], provider: FakeProvider
) -> None:
    assert (await post_signed(client, payload, key="shared-key")).status_code == 200
    payload["tenantId"] = "tenant-2"
    assert (await post_signed(client, payload, key="shared-key")).status_code == 200
    assert provider.calls == 2


async def test_rejects_duplicate_block_ids(client: AsyncClient, payload: dict[str, object]) -> None:
    payload["blocks"] = [payload["blocks"][0], payload["blocks"][0]]  # type: ignore[index]
    response = await post_signed(client, payload, key="duplicate-blocks")
    assert response.status_code == 422


async def test_readiness_fails_closed_when_generation_has_no_redis() -> None:
    settings = Settings(
        openai_api_key=SecretStr("test-key"),
        openai_model="test-model",
        docs_ai_hmac_secret=SecretStr("test-secret-with-at-least-32-bytes"),
        redis_url=None,
    )
    async with AsyncClient(
        transport=ASGITransport(app=create_app(settings=settings)),
        base_url="http://test",
    ) as client:
        response = await client.get("/health/ready")

    assert response.status_code == 503
    assert response.json() == {"status": "not_ready"}
