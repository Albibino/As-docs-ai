from __future__ import annotations

import pytest

from tcc_docs_ai.chat import ChatService
from tcc_docs_ai.chat_prompt import CHAT_SYSTEM_PROMPT, NOT_FOUND_SENTINEL
from tcc_docs_ai.provider import ExtractiveAnswerProvider, sampling_parameters
from tcc_docs_ai.schemas import ChatRequest
from tests.rag_fixtures import (
    CREDENTIAL,
    DEPLOY,
    POLICY_V1,
    POLICY_V2,
    TENANT,
    AppPage,
    FakeDocsApp,
    RecordingChatModel,
    build_stack,
    page_version,
    parse_sse,
)

QUESTION = "Com quantos dias de antecedência devo fazer o pedido de férias?"


def chat_request(question: str = QUESTION, **overrides: object) -> ChatRequest:
    payload: dict[str, object] = {
        "tenantId": TENANT,
        "sessionId": "s1",
        "question": question,
        "mcpCredential": CREDENTIAL,
    }
    payload.update(overrides)
    return ChatRequest.model_validate(payload)


async def run(chat: ChatService, request: ChatRequest, key: str = "k1") -> list[tuple[str, object]]:
    prepared = await chat.prepare(
        request, consumer="docs-app", idempotency_key=key, raw_body=key.encode()
    )
    return parse_sse(b"".join([event async for event in chat.stream(prepared)]))


def first(events: list[tuple[str, object]], name: str) -> dict:
    return next(data for event, data in events if event == name)  # type: ignore[return-value]


async def test_stale_index_is_detected_and_page_is_read_through_mcp() -> None:
    """Caso de teste pedido: desatualizado no índice, atualizado na aplicação."""
    app = FakeDocsApp()
    app.put(AppPage("p-ferias", "Política de férias", version=2, markdown=POLICY_V2))
    model = RecordingChatModel()
    _, indexing, chat, mcp_client = build_stack(model=model, app=app)
    # O índice só conhece a versão 1 (30 dias); a aplicação já está na 2 (45 dias).
    await indexing.index(TENANT, [page_version("p-ferias", 1, "Política de férias", POLICY_V1)])

    events = await run(chat, chat_request())

    meta = first(events, "meta")
    assert meta["stalePages"] == [
        {
            "pageId": "p-ferias",
            "title": "Política de férias",
            "indexedVersion": 1,
            "currentVersion": 2,
        }
    ]
    # A leitura MCP aconteceu exatamente na página desatualizada.
    assert ("docs.read_page", {"pageId": "p-ferias"}) in app.calls
    assert mcp_client and mcp_client.credentials == [CREDENTIAL]
    context = model.context()
    assert context and all(item["origin"] == "mcp_read" for item in context)
    assert all(item["version"] == 2 for item in context)
    joined = " ".join(item["text"] for item in context)
    assert "45 dias" in joined
    assert "30 dias" not in joined  # o conteúdo velho do índice não chega ao modelo
    sources = first(events, "sources")
    assert sources[0]["origin"] == "mcp_read"
    assert first(events, "done") == {"found": True}


async def test_up_to_date_index_answers_without_mcp_read() -> None:
    app = FakeDocsApp()
    app.put(AppPage("p-ferias", "Política de férias", version=2, markdown=POLICY_V2))
    model = RecordingChatModel()
    _, indexing, chat, _ = build_stack(model=model, app=app)
    await indexing.index(TENANT, [page_version("p-ferias", 2, "Política de férias", POLICY_V2)])

    events = await run(chat, chat_request())

    assert first(events, "meta")["stalePages"] == []
    assert "docs.read_page" not in app.tool_names()
    assert all(item["origin"] == "index" for item in model.context())


async def test_only_the_highest_indexed_version_is_evidence() -> None:
    model = RecordingChatModel()
    _, indexing, chat, _ = build_stack(model=model)
    # Duas versões indexadas em momentos diferentes: as duas ficam no índice.
    await indexing.index(TENANT, [page_version("p-ferias", 1, "Política de férias", POLICY_V1)])
    await indexing.index(TENANT, [page_version("p-ferias", 2, "Política de férias", POLICY_V2)])

    events = await run(chat, chat_request(mcpCredential=None))

    meta = first(events, "meta")
    assert meta["ignoredOutdatedChunks"] > 0
    assert meta["freshnessChecked"] is False
    assert {item["version"] for item in model.context()} == {2}


async def test_not_found_does_not_call_the_model_and_lists_similar_pages() -> None:
    model = RecordingChatModel()
    _, indexing, chat, _ = build_stack(model=model)
    await indexing.index(
        TENANT,
        [
            page_version("p-ferias", 1, "Política de férias", POLICY_V1),
            page_version("p-deploy", 1, "Guia de deploy", DEPLOY),
        ],
    )
    chat._tuning = chat._tuning.__class__(min_score=0.99, similar_min_score=0.5)  # noqa: SLF001

    events = await run(
        chat, chat_request("Qual é o prazo do pedido de férias?", mcpCredential=None)
    )

    assert model.inputs == []  # sem evidência o modelo nem é chamado
    not_found = first(events, "not_found")
    assert not_found["similar"][0]["pageId"] == "p-ferias"
    assert first(events, "done") == {"found": False}
    assert not any(event == "delta" for event, _ in events)


async def test_model_sentinel_becomes_not_found_with_similar() -> None:
    model = RecordingChatModel(reply=NOT_FOUND_SENTINEL)
    _, indexing, chat, _ = build_stack(model=model)
    await indexing.index(TENANT, [page_version("p-ferias", 1, "Política de férias", POLICY_V1)])

    events = await run(chat, chat_request("Quem aprova o pedido de férias?", mcpCredential=None))

    assert model.inputs, "havia evidência, então o modelo foi consultado"
    assert not any(event == "delta" for event, _ in events)
    assert first(events, "not_found")["similar"][0]["pageId"] == "p-ferias"


async def test_empty_index_and_no_mcp_is_not_found_without_similar() -> None:
    model = RecordingChatModel()
    _, _, chat, _ = build_stack(model=model)
    events = await run(chat, chat_request(mcpCredential=None))
    assert first(events, "not_found") == {"similar": []}
    assert model.inputs == []


async def test_page_missing_from_index_is_found_through_app_search() -> None:
    app = FakeDocsApp()
    app.put(AppPage("p-novo", "Política de férias", version=1, markdown=POLICY_V2))
    model = RecordingChatModel()
    _, _, chat, _ = build_stack(model=model, app=app)

    events = await run(chat, chat_request())

    stale = first(events, "meta")["stalePages"]
    assert stale[0]["pageId"] == "p-novo" and stale[0]["indexedVersion"] is None
    assert "45 dias" in " ".join(item["text"] for item in model.context())


async def test_page_deleted_in_app_is_never_evidence() -> None:
    app = FakeDocsApp()
    app.put(AppPage("p-ferias", "Política de férias", 1, POLICY_V1, deleted=True))
    model = RecordingChatModel()
    _, indexing, chat, _ = build_stack(model=model, app=app)
    await indexing.index(TENANT, [page_version("p-ferias", 1, "Política de férias", POLICY_V1)])

    events = await run(chat, chat_request())

    assert model.inputs == []
    assert first(events, "done") == {"found": False}


async def test_context_page_is_checked_even_when_not_ranked() -> None:
    app = FakeDocsApp()
    app.put(AppPage("p-ferias", "Política de férias", version=2, markdown=POLICY_V2))
    model = RecordingChatModel()
    _, indexing, chat, _ = build_stack(model=model, app=app)
    await indexing.index(TENANT, [page_version("p-ferias", 1, "Política de férias", POLICY_V1)])

    await run(chat, chat_request("O que mudou neste documento?", contextPageId="p-ferias"))

    get_context = next(args for name, args in app.calls if name == "docs.get_page_context")
    assert "p-ferias" in get_context["pageIds"]
    assert ("docs.read_page", {"pageId": "p-ferias"}) in app.calls


async def test_replays_completed_chat_without_new_model_call() -> None:
    model = RecordingChatModel()
    _, indexing, chat, _ = build_stack(model=model)
    await indexing.index(TENANT, [page_version("p-ferias", 2, "Política de férias", POLICY_V2)])
    request = chat_request(mcpCredential=None)

    first_run = await run(chat, request, key="same")
    second_run = await run(chat, request, key="same")

    assert first_run == second_run
    assert len(model.inputs) == 1


async def test_extractive_provider_only_copies_context() -> None:
    _, indexing, chat, _ = build_stack(model=ExtractiveAnswerProvider())  # type: ignore[arg-type]
    await indexing.index(TENANT, [page_version("p-ferias", 2, "Política de férias", POLICY_V2)])
    events = await run(chat, chat_request(mcpCredential=None))
    text = "".join(data["text"] for event, data in events if event == "delta")  # type: ignore[index]
    assert "45 dias" in text


def test_prompt_forbids_inventing_and_defines_sentinel() -> None:
    assert "SOMENTE" in CHAT_SYSTEM_PROMPT
    assert NOT_FOUND_SENTINEL in CHAT_SYSTEM_PROMPT


def test_prompt_rules_found_with_the_real_model() -> None:
    # Regras acrescentadas depois de testar contra o gpt-6-luna (ver
    # docs/tcc/03-decisoes.md, D11): sem elas o modelo às vezes respondia só
    # "[1]" e tratava páginas divergentes como "não encontrado".
    assert "Nunca responda só com citações" in CHAT_SYSTEM_PROMPT
    assert "páginas DIFERENTES" in CHAT_SYSTEM_PROMPT
    assert "MESMA página" in CHAT_SYSTEM_PROMPT


@pytest.mark.parametrize("effort", ["none", None])
def test_every_model_call_uses_temperature_zero(effort: str | None) -> None:
    params = sampling_parameters(0.0, effort)
    assert params["temperature"] == 0.0
    assert ("reasoning" in params) is (effort is not None)
