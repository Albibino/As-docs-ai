from __future__ import annotations

import json
import os
import time
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from pydantic import SecretStr

from tcc_docs_ai.app import create_app
from tcc_docs_ai.auth import sign_request
from tcc_docs_ai.config import Settings
from tcc_docs_ai.idempotency import InMemoryIdempotencyStore
from tcc_docs_ai.provider import ProviderEvent
from tcc_docs_ai.service import GenerationService


class FixedTokenCounter:
    def __init__(self, value: int = 100) -> None:
        self.value = value

    def count(self, _value: str) -> int:
        return self.value


class FakeProvider:
    def __init__(self, events: list[ProviderEvent] | None = None) -> None:
        self.events = events or [
            ProviderEvent(
                "delta",
                '{"action":"replace","blockId":"b1","baseHash":"h1","content":[{"type":"text","text":"Novo"}]}\n',
            ),
            ProviderEvent(
                "usage", {"inputTokens": 1200, "cachedInputTokens": 1100, "outputTokens": 20}
            ),
        ]
        self.calls = 0

    async def stream(
        self, *, input_messages: list[dict[str, str]], cache_key: str
    ) -> AsyncIterator[ProviderEvent]:
        self.calls += 1
        for event in self.events:
            yield event


@pytest.fixture(autouse=True)
def isolated_settings_env(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Testes unitários não podem depender do `.env` nem do ambiente local.

    Com uma chave real configurada no `.env` do desenvolvedor, `Settings()`
    passaria a ler `OPENAI_API_KEY`, o modelo e os provedores de lá, e os
    testes mudariam de comportamento. Só a pasta `tests/integration` usa o
    ambiente real (chave da OpenAI, Redis).
    """
    if "integration" in request.node.path.parts:
        return
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    for name in list(os.environ):
        if name.startswith(("OPENAI_", "AI_", "DOCS_AI_")) or name in (
            "REDIS_URL",
            "ENVIRONMENT",
        ):
            monkeypatch.delenv(name, raising=False)


@pytest.fixture
def settings() -> Settings:
    return Settings(
        openai_api_key=SecretStr("test-key"),
        openai_model="gpt-5.6-luna",
        docs_ai_hmac_secret=SecretStr("test-secret-with-at-least-32-bytes"),
        ai_max_input_tokens=8_000,
    )


@pytest.fixture
def provider() -> FakeProvider:
    return FakeProvider()


@pytest.fixture
def service(provider: FakeProvider) -> GenerationService:
    return GenerationService(
        provider=provider,
        token_counter=FixedTokenCounter(),
        idempotency=InMemoryIdempotencyStore(),
        max_input_tokens=8_000,
        max_operations=40,
    )


@pytest_asyncio.fixture
async def client(settings: Settings, service: GenerationService):
    async with AsyncClient(
        transport=ASGITransport(app=create_app(settings=settings, service=service)),
        base_url="http://test",
    ) as async_client:
        yield async_client


@pytest.fixture
def payload() -> dict[str, object]:
    return {
        "tenantId": "tenant-1",
        "sessionId": "session-1",
        "request": "Melhore o parágrafo",
        "blocks": [
            {
                "blockId": "b1",
                "type": "paragraph",
                "hash": "h1",
                "content": {"text": "Original"},
                "editable": True,
            }
        ],
    }


def signed_headers(
    payload: dict[str, object],
    *,
    secret: str = "test-secret-with-at-least-32-bytes",
    key: str = "idem-1",
    timestamp: str | None = None,
) -> dict[str, str]:
    body = json.dumps(payload, separators=(",", ":")).encode()
    timestamp = timestamp or str(int(time.time()))
    return {
        "content-type": "application/json",
        "x-ai-consumer": "docs-app",
        "x-ai-timestamp": timestamp,
        "idempotency-key": key,
        "x-ai-signature": sign_request(
            secret,
            timestamp=timestamp,
            method="POST",
            path="/v1/generations/stream",
            idempotency_key=key,
            body=body,
        ),
    }


async def post_signed(
    client: AsyncClient, payload: dict[str, object], *, key: str = "idem-1", **kwargs: object
):
    body = json.dumps(payload, separators=(",", ":"))
    return await client.post(
        "/v1/generations/stream",
        content=body,
        headers=signed_headers(payload, key=key, **kwargs),
    )
