from __future__ import annotations

import json
import time

from httpx import ASGITransport, AsyncClient
from pydantic import SecretStr

from tcc_docs_ai.app import create_app
from tcc_docs_ai.auth import sign_request
from tcc_docs_ai.config import Settings
from tests.rag_fixtures import POLICY_V2, RecordingChatModel, build_stack, parse_sse

SECRET = "test-secret-with-at-least-32-bytes"


def settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "openai_model": "test-model",
        "docs_ai_hmac_secret": SecretStr(SECRET),
        "redis_url": None,
    }
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


async def signed_post(client: AsyncClient, path: str, payload: dict, key: str = "k-1"):
    body = json.dumps(payload, separators=(",", ":")).encode()
    timestamp = str(int(time.time()))
    headers = {
        "content-type": "application/json",
        "x-ai-consumer": "docs-app",
        "x-ai-timestamp": timestamp,
        "idempotency-key": key,
        "x-ai-signature": sign_request(
            SECRET,
            timestamp=timestamp,
            method="POST",
            path=path,
            idempotency_key=key,
            body=body,
        ),
    }
    return await client.post(path, content=body, headers=headers)


def client_for(**services: object) -> AsyncClient:
    return AsyncClient(
        transport=ASGITransport(app=create_app(settings=settings(), **services)),  # type: ignore[arg-type]
        base_url="http://test",
    )


async def test_index_then_chat_over_http() -> None:
    model = RecordingChatModel(reply="São 45 dias [1].")
    _, indexing, chat, _ = build_stack(model=model)
    async with client_for(indexing_service=indexing, chat_service=chat) as client:
        indexed = await signed_post(
            client,
            "/v1/index/pages",
            {
                "tenantId": "w1",
                "pages": [{"pageId": "p1", "version": 2, "title": "Férias", "text": POLICY_V2}],
            },
        )
        again = await signed_post(
            client,
            "/v1/index/pages",
            {
                "tenantId": "w1",
                "pages": [{"pageId": "p1", "version": 2, "title": "Férias", "text": POLICY_V2}],
            },
            key="k-2",
        )
        status = await signed_post(
            client, "/v1/index/status", {"tenantId": "w1", "pageIds": ["p1", "nope"]}, key="k-3"
        )
        answer = await signed_post(
            client,
            "/v1/chat/stream",
            {"tenantId": "w1", "sessionId": "s1", "question": "Qual o prazo do pedido de férias?"},
            key="k-4",
        )

    assert indexed.status_code == 200
    # Contrato com o tcc-docs-app (AiIndexingService.markIndexed): camelCase.
    assert indexed.json()["results"][0] == {
        "pageId": "p1",
        "version": 2,
        "status": "indexed",
        "chunks": indexed.json()["results"][0]["chunks"],
    }
    assert indexed.json()["results"][0]["status"] == "indexed"
    assert again.json()["results"][0]["status"] == "already_indexed"
    assert status.json()["pages"] == [
        {"pageId": "p1", "latestVersion": 2, "deleted": False, "versions": [2]}
    ]
    assert answer.status_code == 200
    events = parse_sse(answer.content)
    assert [name for name, _ in events][0] == "meta"
    assert "".join(data["text"] for name, data in events if name == "delta") == "São 45 dias [1]."


async def test_routes_require_signature() -> None:
    _, indexing, chat, _ = build_stack()
    async with client_for(indexing_service=indexing, chat_service=chat) as client:
        response = await client.post(
            "/v1/chat/stream", json={"tenantId": "w", "sessionId": "s", "question": "q"}
        )
    assert response.status_code == 401


async def test_chat_without_backend_is_unavailable() -> None:
    async with client_for() as client:
        response = await signed_post(
            client, "/v1/chat/stream", {"tenantId": "w", "sessionId": "s", "question": "q"}
        )
    assert response.status_code == 503


async def test_ready_is_fail_closed_without_redis() -> None:
    async with client_for() as client:
        response = await client.get("/health/ready")
    assert response.status_code == 503


def test_temperature_other_than_zero_is_rejected() -> None:
    try:
        settings(openai_temperature=0.7)
    except ValueError:
        return
    raise AssertionError("temperature must be fixed at zero")


def test_settings_build_without_any_api_key() -> None:
    built = settings(ai_embedding_provider="hashing", ai_chat_provider="extractive")
    assert built.openai_api_key is None
    assert built.openai_temperature == 0.0
