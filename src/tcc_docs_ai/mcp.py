from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

import httpx2
from mcp import Implementation
from mcp.client.client import Client
from mcp.client.streamable_http import streamable_http_client
from mcp.types import TextContent

from tcc_docs_ai.errors import (
    McpBudgetExceededError,
    McpToolForbiddenError,
    McpUnavailableError,
)

SEARCH_PAGES = "docs.search_pages"
GET_PAGE_CONTEXT = "docs.get_page_context"
#: "mcp-read" do TCC: lê o conteúdo ATUAL de uma página no app. É o caminho
#: usado quando o índice vetorial está atrás da versão da aplicação.
READ_PAGE = "docs.read_page"

#: Allowlist fixa. É o gate do cliente, não uma preferência: o servidor pode
#: anunciar qualquer tool, mas só estas três chegam ao modelo, e nenhuma delas
#: escreve. Ampliar exige nova decisão documentada em docs/tcc.
ALLOWED_TOOLS: tuple[str, ...] = (SEARCH_PAGES, READ_PAGE, GET_PAGE_CONTEXT)


@dataclass(frozen=True)
class McpBudget:
    """Tetos de uma geração. Nenhum é ajustável por conteúdo recuperado."""

    max_tool_calls: int = 6
    call_timeout_seconds: float = 10.0
    total_timeout_seconds: float = 45.0
    max_result_bytes: int = 200_000


@dataclass(frozen=True)
class McpToolSpec:
    """Descrição de tool entregue ao modelo, já reduzida ao que ele pode escolher."""

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True)
class McpToolResult:
    tool: str
    arguments: dict[str, Any]
    is_error: bool
    text: str


@dataclass
class McpUsage:
    tool_calls: int = 0
    result_bytes: int = 0
    tools: dict[str, int] = field(default_factory=dict)


def _fingerprint(tool: str, arguments: dict[str, Any]) -> str:
    return tool + "\0" + json.dumps(arguments, sort_keys=True, separators=(",", ":"))


class McpToolCaller(Protocol):
    """O mínimo que a sessão usa: listar e chamar tools.

    Deliberadamente estreito — mantém o laço testável contra um servidor em
    processo sem abrir espaço para usar recursos MCP fora deste rollout.
    """

    def list_tools(self, **kwargs: Any) -> Any: ...

    def call_tool(self, name: str, arguments: dict[str, Any], **kwargs: Any) -> Any: ...


class McpReadSession:
    """Sessão MCP de uma geração, com allowlist, orçamento e anti-loop.

    Cada instância vive por uma geração e morre com ela. O estado de orçamento é
    local à instância de propósito: reaproveitar sessão entre gerações deixaria
    um pedido herdar o saldo de outro.
    """

    def __init__(
        self,
        session: McpToolCaller,
        *,
        budget: McpBudget,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._session = session
        self._budget = budget
        self._clock = clock
        self._deadline = clock() + budget.total_timeout_seconds
        self._seen: set[str] = set()
        self.usage = McpUsage()

    async def available_tools(self) -> list[McpToolSpec]:
        """Tools que o modelo pode ver: interseção entre o anunciado e a allowlist.

        Filtrar aqui, e não confiar no servidor, mantém a allowlist válida mesmo
        que o app passe a expor uma tool nova antes de o serviço de IA ser atualizado.
        """
        try:
            listed = await self._session.list_tools()
        except Exception as exc:  # noqa: BLE001 - qualquer falha do transporte é indisponibilidade
            raise McpUnavailableError("MCP server did not list tools") from exc
        return [
            McpToolSpec(
                name=tool.name,
                description=tool.description or "",
                input_schema=dict(tool.input_schema or {}),
            )
            for tool in listed.tools
            if tool.name in ALLOWED_TOOLS
        ]

    async def call(self, tool: str, arguments: dict[str, Any]) -> McpToolResult:
        if tool not in ALLOWED_TOOLS:
            raise McpToolForbiddenError(
                "Model requested a tool outside the read-only allowlist",
                details={"tool": tool[:100]},
            )
        if self.usage.tool_calls >= self._budget.max_tool_calls:
            raise McpBudgetExceededError(
                "MCP tool call budget exhausted",
                details={"maxToolCalls": self._budget.max_tool_calls},
            )
        remaining = self._deadline - self._clock()
        if remaining <= 0:
            raise McpBudgetExceededError(
                "MCP retrieval time budget exhausted",
                details={"totalTimeoutSeconds": self._budget.total_timeout_seconds},
            )

        fingerprint = _fingerprint(tool, arguments)
        if fingerprint in self._seen:
            # Repetir a mesma chamada sem mudança de estado só queima orçamento;
            # é o formato mais comum de laço improdutivo do modelo.
            raise McpBudgetExceededError(
                "Model repeated an identical MCP call without new state",
                details={"tool": tool},
            )
        self._seen.add(fingerprint)

        self.usage.tool_calls += 1
        self.usage.tools[tool] = self.usage.tools.get(tool, 0) + 1
        try:
            result = await self._session.call_tool(
                tool,
                arguments,
                read_timeout_seconds=min(self._budget.call_timeout_seconds, remaining),
            )
        except Exception as exc:  # noqa: BLE001
            # Falha de MCP nunca vira contexto mais amplo nem conexão privilegiada.
            raise McpUnavailableError("MCP tool call failed", details={"tool": tool}) from exc

        text = "".join(
            block.text for block in getattr(result, "content", []) if isinstance(block, TextContent)
        )
        self.usage.result_bytes += len(text.encode("utf-8"))
        if self.usage.result_bytes > self._budget.max_result_bytes:
            raise McpBudgetExceededError(
                "MCP retrieval byte budget exhausted",
                details={"maxResultBytes": self._budget.max_result_bytes},
            )
        return McpToolResult(
            tool=tool,
            arguments=arguments,
            is_error=bool(getattr(result, "is_error", False)),
            text=text,
        )


class McpClient(Protocol):
    def session(self, credential: str) -> AbstractAsyncContextManager[McpReadSession]: ...


class StreamableHttpMcpClient:
    """Cliente MCP do serviço de IA sobre Streamable HTTP, com negociação fixada no deploy.

    A credencial delegada é injetada como cabeçalho da conexão e nunca entra em
    prompt, evento SSE, log ou cache: o modelo escolhe argumentos de tool, jamais
    em nome de quem a chamada acontece.

    `connect_mode` fixa a era do protocolo em vez de deixar o SDK negociar a cada
    conexão. Hoje o servidor MCP do app fala a era de handshake (`legacy`); descobrir
    isso por tentativa deixaria a versão efetiva variando com o que o outro lado
    responde, que é exatamente o que a decisão de fixar a versão proíbe.
    """

    def __init__(
        self,
        *,
        url: str,
        connect_mode: str,
        budget: McpBudget,
        connect_timeout_seconds: float = 10.0,
    ) -> None:
        self._url = url
        self._connect_mode = connect_mode
        self._budget = budget
        self._connect_timeout_seconds = connect_timeout_seconds

    @asynccontextmanager
    async def session(self, credential: str) -> AsyncIterator[McpReadSession]:
        headers = {"Authorization": f"Bearer {credential}"}
        timeout = httpx2.Timeout(
            self._connect_timeout_seconds,
            read=self._budget.total_timeout_seconds,
        )
        try:
            async with httpx2.AsyncClient(headers=headers, timeout=timeout) as http_client:
                transport = streamable_http_client(self._url, http_client=http_client)
                async with Client(
                    transport,
                    mode=cast(Any, self._connect_mode),
                    client_info=Implementation(name="tcc-docs-ai", version="0.2.0"),
                    # Sem cache de respostas: uma leitura autorizada agora não pode
                    # ser reservida depois que a ACL da fonte mudou.
                    cache=None,
                ) as client:
                    yield McpReadSession(cast(McpToolCaller, client), budget=self._budget)
        except (McpBudgetExceededError, McpToolForbiddenError, McpUnavailableError):
            raise
        except Exception as exc:  # noqa: BLE001
            raise McpUnavailableError("MCP session could not be established") from exc
