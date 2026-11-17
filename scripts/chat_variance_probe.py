"""Sonda de variação do modelo de chat (não é teste automatizado).

Roda a mesma pergunta N vezes contra a API real, com o prompt e os parâmetros
do serviço, e mostra quantas respostas distintas saíram. Uso:

    set -a; . ./.env; set +a
    .venv/bin/python scripts/chat_variance_probe.py 8
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from collections import Counter

from tcc_docs_ai.chat_prompt import build_chat_input
from tcc_docs_ai.config import Settings
from tcc_docs_ai.provider import OpenAIResponsesProvider

QUESTION = "Com quantos dias de antecedência devo pedir férias?"
RULE_45 = "O pedido de férias deve ser feito com 45 dias de antecedência."
RULE_30 = "O pedido de férias deve ser feito com 30 dias de antecedência."


def chunk(n: int, title: str, text: str, version: int, origin: str = "index") -> dict:
    return {
        "n": n,
        "pageId": f"page-{n}",
        "title": title,
        "version": version,
        "origin": origin,
        "text": f"## Prazo de solicitação\n\n{text}",
    }


SCENARIOS = {
    "uma página": [
        chunk(1, "Política de férias", RULE_45, 2),
    ],
    "duas páginas em conflito": [
        chunk(1, "Política de férias A", RULE_45, 2),
        chunk(2, "Política de férias B", RULE_30, 1),
    ],
    "sem resposta": [
        chunk(1, "Política de férias", RULE_45, 2),
    ],
}


async def run(provider: OpenAIResponsesProvider, context: list[dict], question: str) -> str:
    text = ""
    async for event in provider.stream(
        input_messages=build_chat_input({"context": context, "question": question, "history": []}),
        cache_key=f"probe-{uuid.uuid4()}",
    ):
        if event.kind == "delta":
            text += str(event.value)
    return text.strip()


async def main(n: int) -> None:
    settings = Settings()
    provider = OpenAIResponsesProvider(
        api_key=settings.openai_api_key.get_secret_value(),  # type: ignore[union-attr]
        model=settings.openai_model,
        max_output_tokens=300,
        connection_retries=1,
        temperature=settings.openai_temperature,
        reasoning_effort=settings.openai_reasoning_effort,
    )
    print(
        f"modelo={settings.openai_model} temperature={settings.openai_temperature}"
        f" effort={settings.openai_reasoning_effort}"
    )
    for name, context in SCENARIOS.items():
        question = "Qual é o valor do vale-refeição?" if name == "sem resposta" else QUESTION
        outputs = await asyncio.gather(*(run(provider, context, question) for _ in range(n)))
        counts = Counter(outputs)
        print(f"\n== {name}: {len(counts)} respostas distintas em {n}")
        for text, count in counts.most_common():
            print(f"  {count}x  {text!r}")


if __name__ == "__main__":
    asyncio.run(main(int(sys.argv[1]) if len(sys.argv) > 1 else 8))
