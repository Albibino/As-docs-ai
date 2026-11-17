from __future__ import annotations

import os
import uuid

import pytest

from tcc_docs_ai.config import Settings
from tcc_docs_ai.provider import OpenAIResponsesProvider

requires_key = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"), reason="requires a real OpenAI API key"
)


def _provider(settings: Settings, *, max_output_tokens: int) -> OpenAIResponsesProvider:
    # Mesma configuração do serviço: temperatura 0 e reasoning effort do
    # ambiente (padrão "none"), como em `build_openai_provider`.
    return OpenAIResponsesProvider(
        api_key=settings.openai_api_key.get_secret_value(),  # type: ignore[union-attr]
        model=settings.openai_model,
        max_output_tokens=max_output_tokens,
        connection_retries=0,
        temperature=settings.openai_temperature,
        reasoning_effort=settings.openai_reasoning_effort,
    )


@pytest.mark.asyncio
@requires_key
async def test_configured_model_accepts_temperature_zero() -> None:
    """O modelo configurado precisa aceitar temperatura 0 (requisito do TCC)."""
    settings = Settings()
    assert settings.openai_temperature == 0.0
    provider = _provider(settings, max_output_tokens=50)
    text = ""
    async for event in provider.stream(
        input_messages=[
            {"role": "developer", "content": "Responda exatamente com a palavra OK."},
            {"role": "user", "content": "Responda."},
        ],
        cache_key=f"integration-{uuid.uuid4()}",
    ):
        if event.kind == "delta":
            text += str(event.value)
    assert "OK" in text.upper()


@pytest.mark.asyncio
@requires_key
async def test_second_turn_reports_cached_input_tokens() -> None:
    """Regression guard for stable-prefix caching."""
    settings = Settings()
    provider = _provider(settings, max_output_tokens=100)
    stable_page = "Contexto estável. " * 1_500
    messages = [
        {"role": "developer", "content": "Return one short sentence."},
        {"role": "user", "content": stable_page},
        {"role": "user", "content": "Summarize."},
    ]
    cache_key = f"integration-{uuid.uuid4()}"

    async def usage() -> dict[str, int]:
        result: dict[str, int] = {}
        async for event in provider.stream(input_messages=messages, cache_key=cache_key):
            if event.kind == "usage":
                result = event.value  # type: ignore[assignment]
        return result

    await usage()
    second = await usage()
    assert second["cachedInputTokens"] > 0
