from __future__ import annotations

import pytest

from tcc_docs_ai.chunking import chunk_markdown
from tcc_docs_ai.errors import BudgetExceededError
from tcc_docs_ai.provider import HashingEmbeddingProvider
from tcc_docs_ai.schemas import IndexStatus
from tests.rag_fixtures import POLICY_V1, POLICY_V2, TENANT, build_stack, page_version


class CountingEmbedder(HashingEmbeddingProvider):
    def __init__(self) -> None:
        super().__init__(dimensions=64)
        self.texts: list[str] = []

    async def embed(self, texts: list[str]):
        self.texts.extend(texts)
        return await super().embed(texts)


def test_chunker_keeps_heading_path_and_is_deterministic() -> None:
    first = chunk_markdown(POLICY_V1, title="Férias", max_chars=200)
    second = chunk_markdown(POLICY_V1, title="Férias", max_chars=200)
    assert first == second
    assert any("Prazo de solicitação" in chunk.text for chunk in first)
    assert all(chunk.text.startswith("Férias") for chunk in first)


def test_chunker_respects_max_chars() -> None:
    text = "# T\n\n" + " ".join(f"Frase número {n}." for n in range(400))
    chunks = chunk_markdown(text, title="T", max_chars=300)
    assert len(chunks) > 1
    assert all(len(chunk.text) <= 300 for chunk in chunks)


async def test_new_version_is_appended_and_old_version_is_kept() -> None:
    index, indexing, _, _ = build_stack()
    await indexing.index(TENANT, [page_version("p1", 1, "Férias", POLICY_V1)])
    v1_elements = index.element_ids(TENANT)

    results = await indexing.index(TENANT, [page_version("p1", 2, "Férias", POLICY_V2)])

    assert results[0].status is IndexStatus.INDEXED
    elements = index.element_ids(TENANT)
    # Append-only: nada da versão 1 foi sobrescrito nem removido.
    assert v1_elements < elements
    assert {element.split(":")[1] for element in elements} == {"1", "2"}
    state = (await index.page_states(TENANT, ["p1"]))["p1"]
    assert state.latest_version == 2
    assert state.versions == (1, 2)


async def test_never_reindexes_an_already_indexed_version() -> None:
    embedder = CountingEmbedder()
    index, indexing, _, _ = build_stack()
    indexing._embedder = embedder  # noqa: SLF001 - instrumenta o embedder do teste
    page = page_version("p1", 1, "Férias", POLICY_V1)

    first = await indexing.index(TENANT, [page])
    embedded = len(embedder.texts)
    vadds = index.vadd_calls
    second = await indexing.index(TENANT, [page])

    assert first[0].status is IndexStatus.INDEXED
    assert second[0].status is IndexStatus.ALREADY_INDEXED
    assert len(embedder.texts) == embedded  # nenhum embedding novo
    assert index.vadd_calls == vadds  # nenhuma escrita nova


async def test_older_version_never_replaces_latest() -> None:
    index, indexing, _, _ = build_stack()
    await indexing.index(TENANT, [page_version("p1", 3, "Férias", POLICY_V2)])
    results = await indexing.index(TENANT, [page_version("p1", 2, "Férias", POLICY_V1)])

    assert results[0].status is IndexStatus.OLDER_THAN_LATEST
    assert (await index.page_states(TENANT, ["p1"]))["p1"].latest_version == 3
    assert all(element.split(":")[1] == "3" for element in index.element_ids(TENANT))


async def test_same_batch_keeps_the_highest_version_as_latest() -> None:
    index, indexing, _, _ = build_stack()
    results = await indexing.index(
        TENANT,
        [page_version("p1", 2, "Férias", POLICY_V2), page_version("p1", 1, "Férias", POLICY_V1)],
    )
    assert [result.status for result in results] == [
        IndexStatus.INDEXED,
        IndexStatus.OLDER_THAN_LATEST,
    ]
    assert (await index.page_states(TENANT, ["p1"]))["p1"].latest_version == 2


async def test_deletion_is_a_versioned_tombstone() -> None:
    index, indexing, _, _ = build_stack()
    await indexing.index(TENANT, [page_version("p1", 1, "Férias", POLICY_V1)])
    results = await indexing.index(TENANT, [page_version("p1", 2, "Férias", "", deleted=True)])
    state = (await index.page_states(TENANT, ["p1"]))["p1"]
    assert results[0].chunks == 0
    assert state.deleted is True and state.latest_version == 2
    # A versão 1 continua no índice (histórico), só deixa de ser a atual.
    assert any(element.startswith("p1:1:") for element in index.element_ids(TENANT))


async def test_tenants_are_isolated() -> None:
    index, indexing, _, _ = build_stack()
    await indexing.index("tenant-a", [page_version("p1", 1, "Férias", POLICY_V1)])
    assert await index.page_states("tenant-b", ["p1"]) == {}
    assert index.element_ids("tenant-b") == set()


async def test_rejects_text_above_budget() -> None:
    _, indexing, _, _ = build_stack()
    indexing._max_text_bytes = 10  # noqa: SLF001
    with pytest.raises(BudgetExceededError):
        await indexing.index(TENANT, [page_version("p1", 1, "Férias", POLICY_V1)])
