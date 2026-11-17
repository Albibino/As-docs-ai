from __future__ import annotations

import asyncio
import json
import socket
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest
import uvicorn
from mcp.server.mcpserver import MCPServer
from starlette.middleware.base import BaseHTTPMiddleware

from tcc_docs_ai.errors import McpUnavailableError
from tcc_docs_ai.mcp import ALLOWED_TOOLS, McpBudget, StreamableHttpMcpClient

# O servidor MCP do app roda o SDK 1.x: a era do protocolo é a de handshake.
CONNECT_MODE = "legacy"
CREDENTIAL = "delegated-read-only-token"
#: O que os handlers viram na requisição. É onde provamos que a credencial
#: delegada chega ao servidor como cabeçalho e nada mais viaja com ela.
seen_headers: list[dict[str, str]] = []


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def build_app():
    server = MCPServer(name="docs-test", version="1.0.0")

    @server.tool(name="docs.search_pages")
    def search_pages(query: str) -> str:
        return json.dumps({"hits": [{"chunkId": "c1", "snippet": f"resultado de {query}"}]})

    @server.tool(name="docs.apply_proposal")
    def apply_proposal(proposalId: str) -> str:  # noqa: N803 - schema do servidor
        raise AssertionError("uma tool de escrita jamais deve ser exposta ao modelo")

    app = server.streamable_http_app(stateless_http=True, json_response=True)

    async def capture(request, call_next):
        seen_headers.append(dict(request.headers))
        return await call_next(request)

    app.add_middleware(BaseHTTPMiddleware, dispatch=capture)
    return app


@asynccontextmanager
async def running_server() -> AsyncIterator[str]:
    port = free_port()
    config = uvicorn.Config(build_app(), host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    try:
        for _ in range(100):
            if server.started:
                break
            await asyncio.sleep(0.05)
        else:  # pragma: no cover - servidor de teste não subiu
            raise RuntimeError("test MCP server did not start")
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        server.should_exit = True
        await task


def client_for(url: str, budget: McpBudget | None = None) -> StreamableHttpMcpClient:
    return StreamableHttpMcpClient(
        url=url,
        connect_mode=CONNECT_MODE,
        budget=budget or McpBudget(),
    )


@pytest.mark.asyncio
async def test_the_real_transport_carries_the_credential_on_every_request() -> None:
    seen_headers.clear()
    async with running_server() as url:
        async with client_for(url).session(CREDENTIAL) as session:
            specs = await session.available_tools()
            result = await session.call("docs.search_pages", {"query": "reembolso"})

    assert [spec.name for spec in specs] == [ALLOWED_TOOLS[0]]
    assert "resultado de reembolso" in result.text
    assert not result.is_error

    assert seen_headers
    for headers in seen_headers:
        # Toda requisição da sessão carrega a credencial delegada; nenhuma cai
        # para uma conexão anônima ou privilegiada no meio do caminho.
        assert headers["authorization"] == f"Bearer {CREDENTIAL}"


@pytest.mark.asyncio
async def test_the_allowlist_holds_against_a_server_advertising_a_write_tool() -> None:
    # O servidor de teste anuncia docs.apply_proposal de propósito: o gate é do
    # cliente, então uma tool nova no app não vira capacidade nova no modelo.
    seen_headers.clear()
    async with running_server() as url:
        async with client_for(url).session(CREDENTIAL) as session:
            names = [spec.name for spec in await session.available_tools()]

    assert "docs.apply_proposal" not in names


@pytest.mark.asyncio
async def test_an_unreachable_server_fails_closed() -> None:
    client = client_for(f"http://127.0.0.1:{free_port()}/mcp")

    with pytest.raises(McpUnavailableError):
        async with client.session(CREDENTIAL):
            pass  # pragma: no cover - a sessão não deve abrir
