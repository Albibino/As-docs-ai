from __future__ import annotations

import json

import pytest

from tcc_docs_ai.errors import BudgetExceededError, UpstreamUnavailableError
from tcc_docs_ai.idempotency import InMemoryIdempotencyStore
from tcc_docs_ai.provider import ProviderEvent
from tcc_docs_ai.schemas import GenerationRequest
from tcc_docs_ai.service import GenerationService
from tests.conftest import FakeProvider, FixedTokenCounter


def request_model() -> GenerationRequest:
    return GenerationRequest.model_validate(
        {
            "tenantId": "t1",
            "sessionId": "s1",
            "request": "edite",
            "blocks": [
                {
                    "blockId": "b1",
                    "type": "paragraph",
                    "hash": "h1",
                    "content": "original",
                }
            ],
        }
    )


@pytest.mark.asyncio
async def test_authoritative_budget_fails_before_provider_call() -> None:
    provider = FakeProvider()
    service = GenerationService(
        provider=provider,
        token_counter=FixedTokenCounter(8_001),
        idempotency=InMemoryIdempotencyStore(),
        max_input_tokens=8_000,
        max_operations=40,
    )
    with pytest.raises(BudgetExceededError) as raised:
        await service.prepare(
            request_model(), consumer="docs-app", idempotency_key="k", raw_body=b"{}"
        )
    assert raised.value.details == {"inputTokens": 8001, "maxInputTokens": 8000}
    assert provider.calls == 0


@pytest.mark.asyncio
async def test_invalid_model_line_becomes_typed_sse_error() -> None:
    provider = FakeProvider([ProviderEvent("delta", "not-json\n")])
    service = GenerationService(
        provider=provider,
        token_counter=FixedTokenCounter(),
        idempotency=InMemoryIdempotencyStore(),
        max_input_tokens=8_000,
        max_operations=40,
    )
    prepared = await service.prepare(
        request_model(), consumer="docs-app", idempotency_key="k", raw_body=b"{}"
    )
    body = b"".join([event async for event in service.stream(prepared)]).decode()
    assert "event: error" in body
    assert "invalid_model_proposal" in body
    assert "event: done" not in body


@pytest.mark.asyncio
async def test_operation_limit_is_enforced() -> None:
    line = json.dumps(
        {"action": "delete", "blockId": "b1", "baseHash": "h1"}, separators=(",", ":")
    )
    provider = FakeProvider([ProviderEvent("delta", f"{line}\n{line}\n")])
    service = GenerationService(
        provider=provider,
        token_counter=FixedTokenCounter(),
        idempotency=InMemoryIdempotencyStore(),
        max_input_tokens=8_000,
        max_operations=1,
    )
    prepared = await service.prepare(
        request_model(), consumer="docs-app", idempotency_key="k", raw_body=b"{}"
    )
    body = b"".join([event async for event in service.stream(prepared)]).decode()
    assert body.count("event: operation") == 1
    assert "event: error" in body


@pytest.mark.asyncio
async def test_partial_upstream_failure_is_typed_and_never_marked_done() -> None:
    line = json.dumps(
        {"action": "delete", "blockId": "b1", "baseHash": "h1"}, separators=(",", ":")
    )

    class PartialFailureProvider:
        async def stream(self, *, input_messages, cache_key):
            yield ProviderEvent("delta", f"{line}\n")
            raise UpstreamUnavailableError("connection lost", details={"partialResponse": True})

    service = GenerationService(
        provider=PartialFailureProvider(),
        token_counter=FixedTokenCounter(),
        idempotency=InMemoryIdempotencyStore(),
        max_input_tokens=8_000,
        max_operations=40,
    )
    prepared = await service.prepare(
        request_model(), consumer="docs-app", idempotency_key="k", raw_body=b"{}"
    )
    body = b"".join([event async for event in service.stream(prepared)]).decode()
    assert body.count("event: operation") == 1
    assert "event: error" in body
    assert '"partialResponse":true' in body
    assert "event: done" not in body
