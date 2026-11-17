from __future__ import annotations

import hashlib
import json
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from tcc_docs_ai.errors import (
    BudgetExceededError,
    InvalidProposalError,
    McpUnavailableError,
    ServiceError,
    UpstreamUnavailableError,
)
from tcc_docs_ai.idempotency import IdempotencyRecord, IdempotencyStore
from tcc_docs_ai.mcp import McpClient
from tcc_docs_ai.prompt import prompt_runnable
from tcc_docs_ai.provider import EmbeddingBatch, EmbeddingProvider, ModelProvider
from tcc_docs_ai.retrieval import McpRetrievalOrchestrator
from tcc_docs_ai.schemas import (
    EmbeddingRequest,
    GenerationMode,
    GenerationRequest,
    ProposalOperation,
)
from tcc_docs_ai.tokens import TokenCounter

#: Token de controle do template de chat vazado pelo modelo (ex.: `</|assistant|>`,
#: `</|im_end|>`). Visto com o gpt-6-luna depois das operações válidas, repetido até
#: estourar `max_output_tokens`. Nenhuma operação começa com "<", então a linha marca
#: o fim da proposta.
CONTROL_TOKEN_LINE = re.compile(r"^<[/|]*[A-Za-z_]+\|?>$")

#: Sentinela do prompt para "nenhuma alteração cabe neste pedido".
NO_CHANGE_ACTION = "none"
NO_CHANGE_REASON_LIMIT = 300


def _no_change_reason(line: str) -> str | None:
    """Motivo da sentinela `{"action":"none","reason":...}`, ou None se não for ela."""
    try:
        value = json.loads(line)
    except ValueError:
        return None
    if not isinstance(value, dict) or value.get("action") != NO_CHANGE_ACTION:
        return None
    reason = value.get("reason")
    return str(reason).strip()[:NO_CHANGE_REASON_LIMIT] if reason else ""


async def _close(stream: AsyncIterator[Any]) -> None:
    """Encerra o stream do provedor (e a conexão HTTP) sem propagar erro."""
    close = getattr(stream, "aclose", None)
    if close is None:
        return
    try:
        await close()
    except Exception:  # noqa: BLE001 - fechar nunca deve mascarar o resultado
        pass


def sse(event: str, data: object) -> bytes:
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event}\ndata: {payload}\n\n".encode()


@dataclass(frozen=True)
class RetrievalTelemetry:
    """Números do laço MCP. Sem conteúdo recuperado e sem credencial."""

    tool_calls: int
    result_bytes: int
    truncated: bool
    tools: dict[str, int]


@dataclass(frozen=True)
class PreparedGeneration:
    consumer: str
    idempotency_scope: str
    idempotency_key: str
    cache_key: str
    input_messages: list[dict[str, Any]]
    input_tokens: int
    mode: GenerationMode = GenerationMode.INLINE
    replay: IdempotencyRecord | None = None
    idempotency_lease: str | None = None
    retrieval: RetrievalTelemetry | None = None


class GenerationService:
    def __init__(
        self,
        *,
        provider: ModelProvider,
        token_counter: TokenCounter,
        idempotency: IdempotencyStore,
        max_input_tokens: int,
        max_operations: int,
        mcp_client: McpClient | None = None,
        retrieval: McpRetrievalOrchestrator | None = None,
    ) -> None:
        self._provider = provider
        self._tokens = token_counter
        self._idempotency = idempotency
        self._max_input_tokens = max_input_tokens
        self._max_operations = max_operations
        # Sem cliente configurado o serviço não degrada para leitura mais ampla:
        # o modo mcp_read simplesmente não existe, e o inline segue intacto.
        self._mcp_client = mcp_client
        self._retrieval = retrieval

    async def prepare(
        self,
        request: GenerationRequest,
        *,
        consumer: str,
        idempotency_key: str,
        raw_body: bytes,
    ) -> PreparedGeneration:
        request_hash = hashlib.sha256(raw_body).hexdigest()
        idempotency_scope = f"{consumer}:{request.tenant_id}"
        cache_source = f"{consumer}:{request.tenant_id}:{request.session_id}"
        cache_key = hashlib.sha256(cache_source.encode()).hexdigest()

        # O lease é tomado antes da recuperação: um retry concorrente não pode
        # gastar o orçamento de leitura duas vezes para o mesmo pedido.
        idempotency = await self._idempotency.start(
            idempotency_scope, idempotency_key, request_hash
        )
        if idempotency.replay:
            return PreparedGeneration(
                consumer=consumer,
                idempotency_scope=idempotency_scope,
                idempotency_key=idempotency_key,
                cache_key=cache_key,
                input_messages=[],
                input_tokens=0,
                mode=request.mode,
                replay=idempotency.replay,
                idempotency_lease=None,
            )

        lease = idempotency.lease_token
        try:
            retrieved, telemetry = await self._retrieve(request, cache_key=cache_key)
            # This runnable is deliberately CPU-only and deterministic. Using invoke avoids
            # dispatching a trivial serialization step through LangChain's thread executor.
            input_messages = prompt_runnable.invoke((request, retrieved))
            serialized = json.dumps(input_messages, ensure_ascii=False, separators=(",", ":"))
            input_tokens = self._tokens.count(serialized)
            if input_tokens > self._max_input_tokens:
                raise BudgetExceededError(
                    "Authoritative input token budget exceeded",
                    details={
                        "inputTokens": input_tokens,
                        "maxInputTokens": self._max_input_tokens,
                    },
                )
        except Exception:
            # Falhar aqui sem devolver o lease deixaria a chave travada até o TTL,
            # e o consumidor não conseguiria repetir o mesmo pedido corrigido.
            if lease:
                await self._idempotency.abandon(idempotency_scope, idempotency_key, lease)
            raise

        return PreparedGeneration(
            consumer=consumer,
            idempotency_scope=idempotency_scope,
            idempotency_key=idempotency_key,
            cache_key=cache_key,
            input_messages=input_messages,
            input_tokens=input_tokens,
            mode=request.mode,
            replay=None,
            idempotency_lease=lease,
            retrieval=telemetry,
        )

    async def _retrieve(
        self, request: GenerationRequest, *, cache_key: str
    ) -> tuple[list[dict[str, Any]], RetrievalTelemetry | None]:
        if request.mode is not GenerationMode.MCP_READ:
            return [], None
        if not self._mcp_client or not self._retrieval:
            raise McpUnavailableError("mcp_read mode is not enabled on this deployment")
        credential = request.mcp_credential
        if not credential:  # pragma: no cover - o schema já garante
            raise McpUnavailableError("mcp_read mode requires a delegated credential")

        async with self._mcp_client.session(credential.get_secret_value()) as session:
            outcome = await self._retrieval.retrieve(
                session=session,
                request=request.request,
                cache_key=cache_key,
            )
        return outcome.messages, RetrievalTelemetry(
            tool_calls=outcome.tool_calls,
            result_bytes=outcome.result_bytes,
            truncated=outcome.truncated,
            tools=outcome.tools_used,
        )

    async def stream(self, prepared: PreparedGeneration) -> AsyncIterator[bytes]:
        if prepared.replay:
            for event in prepared.replay.events:
                yield event
            return

        scope, key = prepared.idempotency_scope, prepared.idempotency_key
        lease = prepared.idempotency_lease
        if not lease:
            raise UpstreamUnavailableError("Generation idempotency lease is missing")
        meta = sse("meta", self._meta_payload(prepared))
        await self._idempotency.append(scope, key, lease, meta)
        yield meta
        buffer = ""
        operation_count = 0
        seen: set[str] = set()
        no_change: str | None = None
        finished = False

        def accept(line: str) -> bytes | None:
            """Uma linha do modelo vira operação, fim da proposta ou erro tipado."""
            nonlocal operation_count, no_change, finished
            text = line.strip()
            if not text:
                return None
            if CONTROL_TOKEN_LINE.match(text):
                finished = True
                return None
            reason = _no_change_reason(text)
            if reason is not None:
                # "Nada a alterar" só vale sozinho; misturado com operações, as
                # operações já emitidas prevalecem e a sentinela é ignorada.
                if operation_count == 0:
                    no_change = reason
                finished = True
                return None
            encoded = self._parse_operation(text, operation_count + 1)
            if encoded in seen:
                # O modelo às vezes repete a mesma operação; a segunda viraria um
                # conflito de hash no cliente.
                return None
            seen.add(encoded)
            operation_count += 1
            return encoded

        upstream = self._provider.stream(
            input_messages=prepared.input_messages, cache_key=prepared.cache_key
        )
        try:
            async for event in upstream:
                if event.kind == "usage":
                    encoded = sse("usage", event.value)
                    await self._idempotency.append(scope, key, lease, encoded)
                    yield encoded
                    continue
                buffer += str(event.value)
                while "\n" in buffer and not finished:
                    line, buffer = buffer.split("\n", 1)
                    encoded = accept(line)
                    if encoded:
                        await self._idempotency.append(scope, key, lease, encoded)
                        yield encoded
                if finished:
                    # Fim da proposta: fecha o stream da OpenAI em vez de ler o resto
                    # (tokens de controle repetidos até o limite de saída virariam
                    # `response.incomplete`). Nesse caso não há evento `usage`.
                    break
            if not finished and buffer.strip():
                encoded = accept(buffer)
                if encoded:
                    await self._idempotency.append(scope, key, lease, encoded)
                    yield encoded
            done_payload: dict[str, Any] = {"operationCount": operation_count}
            if no_change is not None and operation_count == 0:
                done_payload["noChangeReason"] = no_change
            done = sse("done", done_payload)
            await self._idempotency.append(scope, key, lease, done)
            await self._idempotency.complete(scope, key, lease)
            yield done
        except ServiceError as exc:
            encoded = sse("error", exc.payload()["error"])
            await self._idempotency.append(scope, key, lease, encoded)
            await self._idempotency.complete(scope, key, lease)
            yield encoded
        except Exception:
            await self._idempotency.abandon(scope, key, lease)
            raise
        finally:
            await _close(upstream)

    def _meta_payload(self, prepared: PreparedGeneration) -> dict[str, Any]:
        """Contagem autoritativa mais o custo da recuperação, sem conteúdo lido."""
        payload: dict[str, Any] = {
            "inputTokens": prepared.input_tokens,
            "mode": prepared.mode.value,
        }
        if prepared.retrieval:
            payload["retrieval"] = {
                "toolCalls": prepared.retrieval.tool_calls,
                "resultBytes": prepared.retrieval.result_bytes,
                "truncated": prepared.retrieval.truncated,
                "tools": prepared.retrieval.tools,
            }
        return payload

    def _parse_operation(self, line: str, count: int) -> bytes:
        if count > self._max_operations:
            raise InvalidProposalError(
                "Model exceeded proposal operation limit",
                details={"maxOperations": self._max_operations},
            )
        try:
            operation = ProposalOperation.model_validate_json(line)
        except (ValidationError, ValueError) as exc:
            raise InvalidProposalError(
                "Model returned an invalid block operation", details={"line": line[:500]}
            ) from exc
        return sse("operation", operation.model_dump(by_alias=True, exclude_none=True, mode="json"))


class EmbeddingService:
    def __init__(self, *, provider: EmbeddingProvider, max_inputs: int) -> None:
        self._provider = provider
        self._max_inputs = max_inputs

    async def embed(self, request: EmbeddingRequest) -> EmbeddingBatch:
        if len(request.texts) > self._max_inputs:
            raise BudgetExceededError(
                "Embedding input count exceeded",
                details={
                    "inputs": len(request.texts),
                    "maxInputs": self._max_inputs,
                },
            )
        return await self._provider.embed(request.texts)
