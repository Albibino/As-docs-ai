from __future__ import annotations

import json
import time

from httpx import ASGITransport, AsyncClient

from tcc_docs_ai.app import create_app
from tcc_docs_ai.auth import sign_request
from tcc_docs_ai.provider import EmbeddingBatch
from tcc_docs_ai.service import EmbeddingService


class FakeEmbeddingProvider:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> EmbeddingBatch:
        self.calls.append(texts)
        return EmbeddingBatch(
            model="embedding-test-v1",
            dimensions=2,
            embeddings=[[float(index), 1.0] for index, _ in enumerate(texts)],
            input_tokens=sum(len(text.split()) for text in texts),
        )


async def post_embeddings(client: AsyncClient, payload: dict[str, object]):
    body = json.dumps(payload, separators=(",", ":")).encode()
    timestamp = str(int(time.time()))
    key = "embedding-batch-1"
    headers = {
        "content-type": "application/json",
        "x-ai-consumer": "docs-app",
        "x-ai-timestamp": timestamp,
        "idempotency-key": key,
        "x-ai-signature": sign_request(
            "test-secret-with-at-least-32-bytes",
            timestamp=timestamp,
            method="POST",
            path="/v1/embeddings",
            idempotency_key=key,
            body=body,
        ),
    }
    return await client.post("/v1/embeddings", content=body, headers=headers)


async def test_returns_ordered_embeddings_and_usage(settings, service) -> None:
    provider = FakeEmbeddingProvider()
    embedding_service = EmbeddingService(provider=provider, max_inputs=10)

    async with AsyncClient(
        transport=ASGITransport(
            app=create_app(
                settings=settings,
                service=service,
                embedding_service=embedding_service,
            )
        ),
        base_url="http://test",
    ) as client:
        response = await post_embeddings(
            client,
            {"tenantId": "workspace-1", "texts": ["primeiro texto", "segundo"]},
        )

    assert response.status_code == 200
    assert response.json() == {
        "model": "embedding-test-v1",
        "dimensions": 2,
        "embeddings": [[0.0, 1.0], [1.0, 1.0]],
        "usage": {"inputTokens": 3},
    }
    assert provider.calls == [["primeiro texto", "segundo"]]


async def test_rejects_unsigned_embedding_request(client: AsyncClient) -> None:
    response = await client.post(
        "/v1/embeddings",
        json={"tenantId": "workspace-1", "texts": ["texto"]},
    )
    assert response.status_code == 401


async def test_rejects_embedding_batch_over_service_limit(settings, service) -> None:
    provider = FakeEmbeddingProvider()
    embedding_service = EmbeddingService(provider=provider, max_inputs=1)
    async with AsyncClient(
        transport=ASGITransport(
            app=create_app(
                settings=settings,
                service=service,
                embedding_service=embedding_service,
            )
        ),
        base_url="http://test",
    ) as client:
        response = await post_embeddings(
            client,
            {"tenantId": "workspace-1", "texts": ["um", "dois"]},
        )

    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "input_budget_exceeded"
    assert provider.calls == []
