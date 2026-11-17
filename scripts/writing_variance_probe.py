"""Sonda de variação da escrita assistida (não é teste automatizado).

Roda pedidos de edição e pedidos fora do escopo N vezes contra a API real,
pelo `GenerationService` (o mesmo caminho do endpoint), e classifica o que o
serviço emitiria por SSE:
- `ops`: operações (lista as ações);
- `nada`: `done` sem operações, com o motivo dado pelo modelo;
- `erro`: evento `error` (ex.: `invalid_model_proposal` quando o modelo responde
  em prosa).

Uso:
    set -a; . ./.env; set +a
    .venv/bin/python scripts/writing_variance_probe.py 8
"""

from __future__ import annotations

import asyncio
import json
import sys
import uuid
from collections import Counter

from tcc_docs_ai.config import Settings
from tcc_docs_ai.idempotency import InMemoryIdempotencyStore
from tcc_docs_ai.provider import OpenAIResponsesProvider
from tcc_docs_ai.schemas import GenerationRequest
from tcc_docs_ai.service import GenerationService
from tcc_docs_ai.tokens import TokenCounter

PARAGRAPH = {
    "type": "paragraph",
    "attrs": {"id": "p1"},
    "content": [
        {
            "type": "text",
            "text": "no primeiro dia o colaborador recebe o notebook e tem q fazer o "
            "cadastro no sistema de ponto",
        }
    ],
}

CORRECTION = "Corrija a ortografia e a pontuação deste parágrafo."
# Histórico como o editor envia depois de uma correção aceita (caso visto na
# tela: a pergunta seguinte voltava como prosa e virava "Algo deu errado").
AFTER_CORRECTION = [
    {"role": "user", "content": CORRECTION},
    {
        "role": "assistant",
        "content": '[{"action":"replace","blockId":"p1","baseHash":"h1","content":'
        '[{"type":"text","text":"No primeiro dia, o colaborador recebe o notebook e tem '
        'que fazer o cadastro no sistema de ponto."}]}]',
    },
]

REQUESTS: dict[str, tuple[str, list[dict[str, str]]]] = {
    "corrigir ortografia": (CORRECTION, []),
    "pergunta de conhecimento geral": ("Qual é a capital da França?", []),
    "pergunta após uma correção (histórico)": ("Qual é a capital da França?", AFTER_CORRECTION),
    "fato ausente da página": ("Acrescente o horário de início do expediente.", []),
    "lista com o conteúdo da página": ("Transforme o parágrafo numa lista com os passos.", []),
}


def classify(body: str) -> str:
    events = []
    for block in body.strip().split("\n\n"):
        lines = block.split("\n")
        events.append((lines[0].removeprefix("event: "), json.loads(lines[1][6:])))
    error = next((data for name, data in events if name == "error"), None)
    if error:
        return f"erro: {error['code']}: {str(error.get('details', ''))[:90]}"
    ops = [data for name, data in events if name == "operation"]
    done = next(data for name, data in events if name == "done")
    if not ops:
        return f"nada: {done.get('noChangeReason', '')[:90]}"
    return "ops: " + "; ".join(
        f"{op['action']}({str(op.get('content', op.get('title', '')))[:80]})" for op in ops
    )


async def run(
    service: GenerationService, request: str, history: list[dict[str, str]]
) -> str:
    generation = GenerationRequest.model_validate(
        {
            "tenantId": "probe",
            "sessionId": str(uuid.uuid4()),
            "request": request,
            "title": "Onboarding",
            "history": history,
            "blocks": [
                {"blockId": "p1", "type": "paragraph", "hash": "h1", "content": PARAGRAPH}
            ],
        }
    )
    prepared = await service.prepare(
        generation, consumer="docs-app", idempotency_key=str(uuid.uuid4()), raw_body=b"{}"
    )
    body = b"".join([event async for event in service.stream(prepared)]).decode()
    return classify(body)


async def main(n: int) -> None:
    settings = Settings()
    provider = OpenAIResponsesProvider(
        api_key=settings.openai_api_key.get_secret_value(),  # type: ignore[union-attr]
        model=settings.openai_model,
        max_output_tokens=settings.ai_max_output_tokens,
        connection_retries=1,
        temperature=settings.openai_temperature,
        reasoning_effort=settings.openai_reasoning_effort,
    )
    service = GenerationService(
        provider=provider,
        token_counter=TokenCounter(settings.openai_model),
        idempotency=InMemoryIdempotencyStore(),
        max_input_tokens=settings.ai_max_input_tokens,
        max_operations=settings.ai_max_proposal_operations,
    )
    print(f"modelo={settings.openai_model} temperature={settings.openai_temperature}")
    for name, (request, history) in REQUESTS.items():
        outputs = await asyncio.gather(*(run(service, request, history) for _ in range(n)))
        print(f"\n== {name} ({len(set(outputs))} saídas distintas em {n})")
        for output, count in Counter(outputs).most_common():
            print(f"  {count}x  {output}")


if __name__ == "__main__":
    asyncio.run(main(int(sys.argv[1]) if len(sys.argv) > 1 else 8))
