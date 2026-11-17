from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import unicodedata
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast

from openai import APIConnectionError, APITimeoutError, AsyncOpenAI

from tcc_docs_ai.errors import InvalidProposalError, UpstreamUnavailableError


@dataclass(frozen=True)
class ProviderEvent:
    kind: Literal["delta", "usage"]
    value: str | dict[str, int]


@dataclass(frozen=True)
class ToolCall:
    """Pedido de tool emitido pelo modelo, antes de qualquer validação."""

    call_id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class PlanStep:
    """Uma rodada de planejamento: ou o modelo pede tools, ou encerra a leitura."""

    tool_calls: tuple[ToolCall, ...]
    usage: dict[str, int] | None = None


class ModelProvider(Protocol):
    def stream(
        self, *, input_messages: list[dict[str, Any]], cache_key: str
    ) -> AsyncIterator[ProviderEvent]: ...


class PlanningProvider(Protocol):
    async def plan(
        self,
        *,
        input_messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        cache_key: str,
    ) -> PlanStep: ...


@dataclass(frozen=True)
class EmbeddingBatch:
    model: str
    dimensions: int
    embeddings: list[list[float]]
    input_tokens: int


class EmbeddingProvider(Protocol):
    async def embed(self, texts: list[str]) -> EmbeddingBatch: ...


def sampling_parameters(temperature: float, reasoning_effort: str | None) -> dict[str, Any]:
    """Parâmetros de amostragem enviados em TODA chamada ao modelo.

    Temperatura zero é requisito do TCC: a mesma pergunta sobre os mesmos
    documentos deve produzir a mesma resposta, sem criatividade. Modelos de
    raciocínio só aceitam `temperature` com `reasoning.effort=none`.
    """
    params: dict[str, Any] = {"temperature": temperature}
    if reasoning_effort:
        params["reasoning"] = {"effort": reasoning_effort}
    return params


class OpenAIResponsesProvider:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        max_output_tokens: int,
        connection_retries: int,
        temperature: float = 0.0,
        reasoning_effort: str | None = "none",
    ) -> None:
        self._client = AsyncOpenAI(api_key=api_key, max_retries=0)
        self._model = model
        self._max_output_tokens = max_output_tokens
        self._connection_retries = connection_retries
        self._sampling = sampling_parameters(temperature, reasoning_effort)

    async def stream(
        self, *, input_messages: list[dict[str, Any]], cache_key: str
    ) -> AsyncIterator[ProviderEvent]:
        for attempt in range(self._connection_retries + 1):
            emitted = False
            try:
                async with self._client.responses.stream(
                    model=self._model,
                    input=cast(Any, input_messages),
                    **self._sampling,
                    max_output_tokens=self._max_output_tokens,
                    prompt_cache_key=cache_key,
                    store=False,
                ) as stream:
                    async for event in stream:
                        if event.type == "response.output_text.delta":
                            emitted = True
                            yield ProviderEvent("delta", event.delta)
                        elif event.type == "response.completed":
                            usage = event.response.usage
                            details = getattr(usage, "input_tokens_details", None)
                            yield ProviderEvent(
                                "usage",
                                {
                                    "inputTokens": usage.input_tokens,
                                    "cachedInputTokens": getattr(details, "cached_tokens", 0) or 0,
                                    "outputTokens": usage.output_tokens,
                                },
                            )
                        elif event.type == "response.incomplete":
                            details = event.response.incomplete_details
                            raise InvalidProposalError(
                                "OpenAI response ended before the proposal was complete",
                                details={"reason": getattr(details, "reason", "unknown")},
                            )
                        elif event.type == "response.failed":
                            error = event.response.error
                            raise UpstreamUnavailableError(
                                "OpenAI failed to generate the proposal",
                                details={"upstreamCode": getattr(error, "code", "unknown")},
                            )
                return
            except (APIConnectionError, APITimeoutError) as exc:
                if emitted or attempt >= self._connection_retries:
                    raise UpstreamUnavailableError(
                        "OpenAI connection failed",
                        details={"partialResponse": emitted, "retryAttempt": attempt},
                    ) from exc
                await asyncio.sleep(0.25 * (2**attempt))

    async def plan(
        self,
        *,
        input_messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        cache_key: str,
    ) -> PlanStep:
        """Uma rodada de recuperação: pede tools ou sinaliza que já tem contexto.

        Sem streaming de propósito. O laço de leitura não produz proposta; só a
        geração final faz isso, e ela continua sendo o único caminho de escrita.
        """
        for attempt in range(self._connection_retries + 1):
            try:
                response = await self._client.responses.create(
                    model=self._model,
                    input=cast(Any, input_messages),
                    tools=cast(Any, tools),
                    tool_choice="auto",
                    **self._sampling,
                    max_output_tokens=self._max_output_tokens,
                    prompt_cache_key=cache_key,
                    store=False,
                )
            except (APIConnectionError, APITimeoutError) as exc:
                if attempt >= self._connection_retries:
                    raise UpstreamUnavailableError("OpenAI planning call failed") from exc
                await asyncio.sleep(0.25 * (2**attempt))
                continue

            calls = tuple(
                ToolCall(
                    call_id=str(getattr(item, "call_id", "") or getattr(item, "id", "")),
                    name=str(getattr(item, "name", "")),
                    arguments=str(getattr(item, "arguments", "") or "{}"),
                )
                for item in (response.output or [])
                if getattr(item, "type", "") == "function_call"
            )
            usage = response.usage
            details = getattr(usage, "input_tokens_details", None)
            return PlanStep(
                tool_calls=calls,
                usage={
                    "inputTokens": usage.input_tokens,
                    "cachedInputTokens": getattr(details, "cached_tokens", 0) or 0,
                    "outputTokens": usage.output_tokens,
                }
                if usage
                else None,
            )

        raise AssertionError("unreachable")


class OpenAIEmbeddingProvider:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        dimensions: int,
        connection_retries: int,
    ) -> None:
        self._client = AsyncOpenAI(api_key=api_key, max_retries=0)
        self._model = model
        self.model = model
        self._dimensions = dimensions
        self._connection_retries = connection_retries

    async def embed(self, texts: list[str]) -> EmbeddingBatch:
        for attempt in range(self._connection_retries + 1):
            try:
                response = await self._client.embeddings.create(
                    model=self._model,
                    input=texts,
                    dimensions=self._dimensions,
                    encoding_format="float",
                )
                ordered = sorted(response.data, key=lambda item: item.index)
                vectors = [list(item.embedding) for item in ordered]
                if len(vectors) != len(texts) or any(
                    len(vector) != self._dimensions for vector in vectors
                ):
                    raise UpstreamUnavailableError(
                        "OpenAI returned an invalid embedding batch",
                        details={"expectedInputs": len(texts)},
                    )
                return EmbeddingBatch(
                    model=self._model,
                    dimensions=self._dimensions,
                    embeddings=vectors,
                    input_tokens=response.usage.prompt_tokens,
                )
            except (APIConnectionError, APITimeoutError) as exc:
                if attempt >= self._connection_retries:
                    raise UpstreamUnavailableError("OpenAI embedding request failed") from exc
                await asyncio.sleep(0.25 * (2**attempt))

        raise AssertionError("unreachable")


_WORD = re.compile(r"[a-z0-9]+")


def normalize_words(text: str) -> list[str]:
    """Palavras minúsculas e sem acento. Usado no hashing e na busca por título."""
    folded = unicodedata.normalize("NFKD", text.lower())
    folded = "".join(char for char in folded if not unicodedata.combining(char))
    return _WORD.findall(folded)


class HashingEmbeddingProvider:
    """Embedding determinístico por feature hashing, sem rede e sem chave.

    Só para desenvolvimento e testes: aproxima sobreposição lexical, não
    semântica. Unigramas e bigramas caem em `dimensions` posições com sinal
    derivado do hash; o vetor sai normalizado (L2), então o cosseno do Redis
    funciona igual ao dos embeddings da OpenAI.
    """

    def __init__(self, *, dimensions: int = 256) -> None:
        self._dimensions = dimensions
        self.model = f"hashing-v1-{dimensions}"

    def _vector(self, text: str) -> list[float]:
        words = normalize_words(text)
        features = words + [f"{a} {b}" for a, b in zip(words, words[1:], strict=False)]
        vector = [0.0] * self._dimensions
        for feature in features:
            digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            vector[value % self._dimensions] += 1.0 if (value >> 63) & 1 else -1.0
        norm = math.sqrt(sum(item * item for item in vector))
        if norm == 0:
            # Texto sem palavras ainda precisa de um vetor válido (norma > 0).
            vector[0] = 1.0
            return vector
        return [item / norm for item in vector]

    async def embed(self, texts: list[str]) -> EmbeddingBatch:
        return EmbeddingBatch(
            model=self.model,
            dimensions=self._dimensions,
            embeddings=[self._vector(text) for text in texts],
            input_tokens=sum(len(normalize_words(text)) for text in texts),
        )


class ExtractiveAnswerProvider:
    """Responde sem LLM, devolvendo literalmente os trechos de evidência.

    Existe para o desenvolvimento sem chave da OpenAI: o fluxo inteiro (índice,
    verificação de versão, leitura MCP, não-encontrado) é exercitado, e a
    resposta continua incapaz de inventar — ela só copia o CONTEXT.
    """

    model = "extractive-v1"

    async def stream(
        self, *, input_messages: list[dict[str, Any]], cache_key: str
    ) -> AsyncIterator[ProviderEvent]:
        context: list[dict[str, Any]] = []
        for message in input_messages:
            content = message.get("content")
            if isinstance(content, str) and content.startswith("CONTEXT\n"):
                context = json.loads(content.removeprefix("CONTEXT\n"))
        if not context:
            yield ProviderEvent("delta", "NAO_ENCONTRADO")
            return
        lines = ["Trechos dos documentos relacionados à pergunta:"]
        for item in context[:3]:
            excerpt = " ".join(str(item.get("text", "")).split())[:400]
            lines.append(f"\n[{item['n']}] {item['title']}: {excerpt}")
        yield ProviderEvent("delta", "\n".join(lines))
        yield ProviderEvent("usage", {"inputTokens": 0, "cachedInputTokens": 0, "outputTokens": 0})
