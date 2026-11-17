from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Any

import pytest
from mcp import ClientSession
from mcp.client._memory import InMemoryTransport
from mcp.server.mcpserver import MCPServer

from tcc_docs_ai.errors import IdempotencyConflictError, McpUnavailableError
from tcc_docs_ai.idempotency import InMemoryIdempotencyStore
from tcc_docs_ai.mcp import McpBudget, McpReadSession
from tcc_docs_ai.provider import PlanStep, ProviderEvent, ToolCall
from tcc_docs_ai.retrieval import UNTRUSTED_CHANNEL, McpRetrievalOrchestrator
from tcc_docs_ai.schemas import GenerationRequest
from tcc_docs_ai.service import GenerationService
from tests.conftest import FakeProvider, FixedTokenCounter

CREDENTIAL = "delegated-read-only-token"
SECRET_CHUNK = "Texto recuperado apenas pelo MCP."


def request_model(**overrides: object) -> GenerationRequest:
    payload: dict[str, object] = {
        "tenantId": "t1",
        "sessionId": "s1",
        "request": "resuma a política",
        "blocks": [{"blockId": "b1", "type": "paragraph", "hash": "h1", "content": "original"}],
    }
    payload.update(overrides)
    return GenerationRequest.model_validate(payload)


def mcp_request() -> GenerationRequest:
    return request_model(mode="mcp_read", mcpCredential=CREDENTIAL)


class RecordingMcpClient:
    """Cliente MCP de teste sobre transporte em memória.

    Registra a credencial recebida para provar que ela chega à conexão — e só
    a ela, jamais ao prompt.
    """

    def __init__(self, server: MCPServer, budget: McpBudget | None = None) -> None:
        self._server = server
        self._budget = budget or McpBudget()
        self.credentials: list[str] = []

    @asynccontextmanager
    async def session(self, credential: str):
        self.credentials.append(credential)
        async with InMemoryTransport(self._server) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as client:
                await client.initialize()
                yield McpReadSession(client, budget=self._budget)


class ScriptedPlanner:
    def __init__(self, steps: list[PlanStep]) -> None:
        self._steps = list(steps)

    async def plan(self, *, input_messages, tools, cache_key) -> PlanStep:
        return self._steps.pop(0) if self._steps else PlanStep(tool_calls=())


def search_server() -> MCPServer:
    server = MCPServer(name="docs-test", version="1.0.0")

    @server.tool(name="docs.search_pages")
    def search_pages(query: str) -> str:
        return json.dumps({"hits": [{"chunkId": "c1", "snippet": SECRET_CHUNK}]})

    return server


def build_service(
    *,
    provider: Any,
    mcp_client: Any = None,
    planner: Any = None,
    idempotency: InMemoryIdempotencyStore | None = None,
) -> GenerationService:
    return GenerationService(
        provider=provider,
        token_counter=FixedTokenCounter(),
        idempotency=idempotency or InMemoryIdempotencyStore(),
        max_input_tokens=8_000,
        max_operations=40,
        mcp_client=mcp_client,
        retrieval=McpRetrievalOrchestrator(planner) if planner else None,
    )


@pytest.mark.asyncio
async def test_mcp_read_feeds_the_generation_and_never_leaks_the_credential() -> None:
    client = RecordingMcpClient(search_server())
    planner = ScriptedPlanner(
        [
            PlanStep(
                tool_calls=(
                    ToolCall(call_id="1", name="docs.search_pages", arguments='{"query":"a"}'),
                )
            ),
            PlanStep(tool_calls=()),
        ]
    )
    service = build_service(provider=FakeProvider(), mcp_client=client, planner=planner)

    prepared = await service.prepare(
        mcp_request(), consumer="docs-app", idempotency_key="k", raw_body=b"{}"
    )

    serialized = json.dumps(prepared.input_messages, ensure_ascii=False)
    # O conteúdo recuperado alimenta a geração...
    assert SECRET_CHUNK in serialized
    assert UNTRUSTED_CHANNEL in serialized
    # ...mas a credencial só existe na conexão MCP.
    assert client.credentials == [CREDENTIAL]
    assert CREDENTIAL not in serialized
    assert prepared.retrieval is not None
    assert prepared.retrieval.tool_calls == 1
    assert prepared.retrieval.tools == {"docs.search_pages": 1}


@pytest.mark.asyncio
async def test_the_meta_event_reports_retrieval_cost_without_content() -> None:
    client = RecordingMcpClient(search_server())
    planner = ScriptedPlanner(
        [
            PlanStep(
                tool_calls=(
                    ToolCall(call_id="1", name="docs.search_pages", arguments='{"query":"a"}'),
                )
            ),
            PlanStep(tool_calls=()),
        ]
    )
    service = build_service(provider=FakeProvider(), mcp_client=client, planner=planner)

    prepared = await service.prepare(
        mcp_request(), consumer="docs-app", idempotency_key="k", raw_body=b"{}"
    )
    body = b"".join([event async for event in service.stream(prepared)]).decode()

    meta_line = next(
        line for line in body.splitlines() if line.startswith("data:") and "inputTokens" in line
    )
    meta = json.loads(meta_line.removeprefix("data:"))
    assert meta["mode"] == "mcp_read"
    assert meta["retrieval"] == {
        "toolCalls": 1,
        "resultBytes": len(json.dumps({"hits": [{"chunkId": "c1", "snippet": SECRET_CHUNK}]})),
        "truncated": False,
        "tools": {"docs.search_pages": 1},
    }
    assert SECRET_CHUNK not in meta_line
    assert CREDENTIAL not in body


@pytest.mark.asyncio
async def test_inline_mode_never_touches_the_mcp_client() -> None:
    client = RecordingMcpClient(search_server())
    service = build_service(provider=FakeProvider(), mcp_client=client, planner=ScriptedPlanner([]))

    prepared = await service.prepare(
        request_model(), consumer="docs-app", idempotency_key="k", raw_body=b"{}"
    )

    assert client.credentials == []
    assert prepared.retrieval is None
    assert prepared.mode.value == "inline"


@pytest.mark.asyncio
async def test_mcp_read_is_refused_when_the_client_flag_is_off() -> None:
    # Rollback: com a flag desligada o modo não degrada para leitura mais ampla,
    # ele simplesmente não existe.
    service = build_service(provider=FakeProvider())

    with pytest.raises(McpUnavailableError, match="not enabled"):
        await service.prepare(
            mcp_request(), consumer="docs-app", idempotency_key="k", raw_body=b"{}"
        )


@pytest.mark.asyncio
async def test_a_failed_retrieval_releases_the_idempotency_lease() -> None:
    # Sem devolver o lease, o consumidor ficaria travado até o TTL e não
    # conseguiria repetir o mesmo pedido depois de corrigir a causa.
    idempotency = InMemoryIdempotencyStore()
    service = build_service(provider=FakeProvider(), idempotency=idempotency)

    with pytest.raises(McpUnavailableError):
        await service.prepare(
            mcp_request(), consumer="docs-app", idempotency_key="k", raw_body=b"{}"
        )

    started = await idempotency.start("docs-app:t1", "k", "hash")
    assert started.lease_token


@pytest.mark.asyncio
async def test_a_concurrent_retry_cannot_spend_the_retrieval_budget_twice() -> None:
    client = RecordingMcpClient(search_server())
    planner = ScriptedPlanner(
        [
            PlanStep(
                tool_calls=(
                    ToolCall(call_id="1", name="docs.search_pages", arguments='{"query":"a"}'),
                )
            ),
            PlanStep(tool_calls=()),
        ]
    )
    idempotency = InMemoryIdempotencyStore()
    service = build_service(
        provider=FakeProvider(), mcp_client=client, planner=planner, idempotency=idempotency
    )

    await service.prepare(mcp_request(), consumer="docs-app", idempotency_key="k", raw_body=b"{}")
    with pytest.raises(IdempotencyConflictError):
        await service.prepare(
            mcp_request(), consumer="docs-app", idempotency_key="k", raw_body=b"{}"
        )

    # A segunda tentativa nem abriu sessão MCP: o lease barrou antes da leitura.
    assert client.credentials == [CREDENTIAL]


@pytest.mark.asyncio
async def test_a_replayed_generation_does_not_retrieve_again() -> None:
    client = RecordingMcpClient(search_server())
    planner = ScriptedPlanner(
        [
            PlanStep(
                tool_calls=(
                    ToolCall(call_id="1", name="docs.search_pages", arguments='{"query":"a"}'),
                )
            ),
            PlanStep(tool_calls=()),
            PlanStep(tool_calls=()),
        ]
    )
    idempotency = InMemoryIdempotencyStore()
    service = build_service(
        provider=FakeProvider([ProviderEvent("delta", "")]),
        mcp_client=client,
        planner=planner,
        idempotency=idempotency,
    )

    first = await service.prepare(
        mcp_request(), consumer="docs-app", idempotency_key="k", raw_body=b"{}"
    )
    first_body = b"".join([event async for event in service.stream(first)])

    replay = await service.prepare(
        mcp_request(), consumer="docs-app", idempotency_key="k", raw_body=b"{}"
    )
    replay_body = b"".join([event async for event in service.stream(replay)])

    assert replay.replay is not None
    assert replay_body == first_body
    # Replay não relê o índice: uma repetição não pode custar um segundo
    # orçamento de leitura nem reexpor fontes já entregues.
    assert client.credentials == [CREDENTIAL]
