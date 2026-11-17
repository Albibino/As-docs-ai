"""Saídas reais do gpt-6-luna na escrita assistida (ver docs/tcc/03-decisoes.md, D12).

Cada caso reproduz um texto que o modelo devolveu de fato nas sondas
(`scripts/writing_variance_probe.py`) e fixa o que o serviço emite por SSE.
"""

from __future__ import annotations

import json

import pytest

from tcc_docs_ai.idempotency import InMemoryIdempotencyStore
from tcc_docs_ai.provider import ProviderEvent
from tcc_docs_ai.schemas import GenerationRequest
from tcc_docs_ai.service import GenerationService
from tests.conftest import FixedTokenCounter

REPLACE = json.dumps(
    {
        "action": "replace",
        "blockId": "b1",
        "baseHash": "h1",
        "content": [{"type": "text", "text": "No primeiro dia, o colaborador recebe o notebook."}],
    },
    separators=(",", ":"),
    ensure_ascii=False,
)


class ScriptedProvider:
    """Devolve o texto em pedaços e registra se o stream foi fechado cedo."""

    def __init__(self, text: str, *, chunk: int = 7, usage: bool = True) -> None:
        self.text = text
        self.chunk = chunk
        self.usage = usage
        self.closed_early = False
        self.consumed = 0

    async def stream(self, *, input_messages, cache_key):
        try:
            for start in range(0, len(self.text), self.chunk):
                self.consumed = start + self.chunk
                yield ProviderEvent("delta", self.text[start : start + self.chunk])
            if self.usage:
                yield ProviderEvent(
                    "usage", {"inputTokens": 10, "cachedInputTokens": 0, "outputTokens": 5}
                )
        except GeneratorExit:
            self.closed_early = True
            raise


def request_model() -> GenerationRequest:
    return GenerationRequest.model_validate(
        {
            "tenantId": "t1",
            "sessionId": "s1",
            "request": "edite",
            "blocks": [{"blockId": "b1", "type": "paragraph", "hash": "h1", "content": "x"}],
        }
    )


async def run(provider: ScriptedProvider) -> list[tuple[str, object]]:
    service = GenerationService(
        provider=provider,  # type: ignore[arg-type]
        token_counter=FixedTokenCounter(),
        idempotency=InMemoryIdempotencyStore(),
        max_input_tokens=8_000,
        max_operations=40,
    )
    prepared = await service.prepare(
        request_model(), consumer="docs-app", idempotency_key="k", raw_body=b"{}"
    )
    body = b"".join([event async for event in service.stream(prepared)]).decode()
    events = []
    for block in body.strip().split("\n\n"):
        name = block.split("\n")[0].removeprefix("event: ")
        data = json.loads(block.split("\n")[1].removeprefix("data: "))
        events.append((name, data))
    return events


def names(events: list[tuple[str, object]]) -> list[str]:
    return [name for name, _ in events]


@pytest.mark.asyncio
async def test_leaked_control_tokens_end_the_proposal_and_close_the_stream() -> None:
    # Visto em 7 de 10 chamadas com uma regra extra no prompt: a operação certa,
    # depois "</|assistant|>" repetido até estourar max_output_tokens.
    provider = ScriptedProvider(REPLACE + " \n" + "</|assistant|> \n" * 600)
    events = await run(provider)
    assert names(events) == ["meta", "operation", "done"]
    assert events[-1][1] == {"operationCount": 1}
    assert provider.closed_early
    assert provider.consumed < len(provider.text) // 10


@pytest.mark.parametrize("token", ["</assistant>", "</|im_end|>", "<|end|>"])
@pytest.mark.asyncio
async def test_other_control_token_spellings_also_end_the_proposal(token: str) -> None:
    events = await run(ScriptedProvider(f"{REPLACE}\n{token}\n{token}\n"))
    assert names(events) == ["meta", "operation", "done"]


@pytest.mark.asyncio
async def test_no_change_sentinel_becomes_done_with_reason() -> None:
    line = '{"action":"none","reason":"A solicitação é uma pergunta e não pede alteração."}'
    events = await run(ScriptedProvider(line + "\n"))
    # A sentinela encerra a proposta e fecha o stream (sem esperar `usage`).
    assert names(events) == ["meta", "done"]
    assert events[-1][1] == {
        "operationCount": 0,
        "noChangeReason": "A solicitação é uma pergunta e não pede alteração.",
    }


@pytest.mark.asyncio
async def test_no_change_sentinel_without_trailing_newline() -> None:
    events = await run(ScriptedProvider('{"action":"none","reason":"Nada a fazer."}'))
    assert events[-1][1] == {"operationCount": 0, "noChangeReason": "Nada a fazer."}


@pytest.mark.asyncio
async def test_sentinel_after_operations_is_ignored() -> None:
    events = await run(ScriptedProvider(f'{REPLACE}\n{{"action":"none","reason":"x"}}\n'))
    assert names(events) == ["meta", "operation", "done"]
    assert events[-1][1] == {"operationCount": 1}


@pytest.mark.asyncio
async def test_identical_operations_are_emitted_once() -> None:
    # Visto em 2 de 10 correções: a mesma substituição duas vezes. A segunda
    # chegaria ao cliente com um baseHash já superado (conflito).
    events = await run(ScriptedProvider(f"{REPLACE}\n{REPLACE}\n"))
    assert names(events) == ["meta", "operation", "usage", "done"]
    assert events[-1][1] == {"operationCount": 1}


@pytest.mark.asyncio
async def test_prose_answer_is_still_rejected() -> None:
    # Com o prompt original, 8 de 10 perguntas após uma correção voltavam assim.
    events = await run(ScriptedProvider("Não há informações no documento.\n"))
    assert names(events) == ["meta", "error"]
    assert events[-1][1]["code"] == "invalid_model_proposal"  # type: ignore[index]
