"""Peças compartilhadas pelos testes de índice versionado e conversa."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from mcp import ClientSession
from mcp.client._memory import InMemoryTransport
from mcp.server.mcpserver import MCPServer

from tcc_docs_ai.chat import ChatService, ChatTuning
from tcc_docs_ai.idempotency import InMemoryIdempotencyStore
from tcc_docs_ai.indexing import IndexingService
from tcc_docs_ai.mcp import McpBudget, McpReadSession
from tcc_docs_ai.provider import HashingEmbeddingProvider, ProviderEvent
from tcc_docs_ai.schemas import IndexPageVersion
from tcc_docs_ai.vectorstore import InMemoryVectorIndex
from tests.conftest import FixedTokenCounter

TENANT = "workspace-1"
CREDENTIAL = "delegated-read-only-token"

POLICY_V1 = """# Política de férias

## Prazo de solicitação
O pedido de férias deve ser feito com 30 dias de antecedência pelo gestor.

## Fracionamento
As férias podem ser divididas em até três períodos.
"""

POLICY_V2 = """# Política de férias

## Prazo de solicitação
O pedido de férias deve ser feito com 45 dias de antecedência pelo gestor.

## Fracionamento
As férias podem ser divididas em até três períodos.
"""

DEPLOY = """# Guia de deploy

## Ambiente
O deploy usa Docker e Railway com Postgres e Redis gerenciados.
"""


@dataclass
class AppPage:
    page_id: str
    title: str
    version: int
    markdown: str
    deleted: bool = False


@dataclass
class FakeDocsApp:
    """Estado da aplicação de documentos visto pelo servidor MCP de teste."""

    pages: dict[str, AppPage] = field(default_factory=dict)
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def put(self, page: AppPage) -> None:
        self.pages[page.page_id] = page

    def server(self) -> MCPServer:
        server = MCPServer(name="tcc-docs-app-test", version="1.0.0")

        @server.tool(name="docs.search_pages")
        def search_pages(query: str, limit: int = 5) -> str:
            self.calls.append(("docs.search_pages", {"query": query}))
            words = {word for word in query.lower().split() if len(word) > 3}
            hits = [
                {"pageId": page.page_id, "title": page.title, "version": page.version}
                for page in self.pages.values()
                if not page.deleted and words & set(page.markdown.lower().split())
            ]
            return json.dumps({"pages": hits[:limit]})

        @server.tool(name="docs.get_page_context")
        def get_page_context(pageIds: list[str]) -> str:  # noqa: N803 - schema do servidor
            self.calls.append(("docs.get_page_context", {"pageIds": pageIds}))
            pages = [
                {
                    "pageId": page.page_id,
                    "title": page.title,
                    "version": page.version,
                    "deleted": page.deleted,
                }
                for page_id in pageIds
                if (page := self.pages.get(page_id))
            ]
            return json.dumps({"pages": pages})

        @server.tool(name="docs.read_page")
        def read_page(pageId: str) -> str:  # noqa: N803 - schema do servidor
            self.calls.append(("docs.read_page", {"pageId": pageId}))
            page = self.pages[pageId]
            return json.dumps(
                {
                    "pageId": page.page_id,
                    "title": page.title,
                    "version": page.version,
                    "markdown": page.markdown,
                }
            )

        return server

    def tool_names(self) -> list[str]:
        return [name for name, _ in self.calls]


class InMemoryMcpClient:
    def __init__(self, app: FakeDocsApp, budget: McpBudget | None = None) -> None:
        self._app = app
        self._budget = budget or McpBudget(max_tool_calls=10)
        self.credentials: list[str] = []

    @asynccontextmanager
    async def session(self, credential: str) -> AsyncIterator[McpReadSession]:
        self.credentials.append(credential)
        async with InMemoryTransport(self._app.server()) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as client:
                await client.initialize()
                yield McpReadSession(client, budget=self._budget)


class RecordingChatModel:
    """Modelo falso: devolve o texto pedido e guarda o que recebeu."""

    def __init__(self, reply: str = "O prazo é de 45 dias [1].") -> None:
        self.reply = reply
        self.inputs: list[list[dict[str, Any]]] = []

    async def stream(
        self, *, input_messages: list[dict[str, Any]], cache_key: str
    ) -> AsyncIterator[ProviderEvent]:
        self.inputs.append(input_messages)
        for piece in (self.reply[:3], self.reply[3:]):
            if piece:
                yield ProviderEvent("delta", piece)
        yield ProviderEvent("usage", {"inputTokens": 10, "cachedInputTokens": 0, "outputTokens": 5})

    def context(self) -> list[dict[str, Any]]:
        for message in self.inputs[-1]:
            if str(message.get("content", "")).startswith("CONTEXT\n"):
                return json.loads(message["content"].removeprefix("CONTEXT\n"))
        return []


def page_version(
    page_id: str, version: int, title: str, text: str, *, deleted: bool = False
) -> IndexPageVersion:
    return IndexPageVersion(
        pageId=page_id, version=version, title=title, text=text, deleted=deleted
    )


def build_stack(
    *, model: RecordingChatModel | None = None, app: FakeDocsApp | None = None
) -> tuple[InMemoryVectorIndex, IndexingService, ChatService, InMemoryMcpClient | None]:
    index = InMemoryVectorIndex()
    embedder = HashingEmbeddingProvider(dimensions=512)
    indexing = IndexingService(
        index=index,
        embedder=embedder,
        model=embedder.model,
        chunk_max_chars=1_600,
        max_embedding_inputs=100,
        max_text_bytes=2_000_000,
    )
    mcp_client = InMemoryMcpClient(app) if app else None
    chat = ChatService(
        index=index,
        embedder=embedder,
        provider=model or RecordingChatModel(),
        idempotency=InMemoryIdempotencyStore(),
        token_counter=FixedTokenCounter(),
        mcp_client=mcp_client,
        tuning=ChatTuning(min_score=0.6, similar_min_score=0.52),
    )
    return index, indexing, chat, mcp_client


def parse_sse(raw: bytes) -> list[tuple[str, Any]]:
    events: list[tuple[str, Any]] = []
    for block in raw.decode().strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events
