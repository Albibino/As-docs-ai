from __future__ import annotations

from collections.abc import Sequence

from tcc_docs_ai.chunking import chunk_markdown
from tcc_docs_ai.errors import BudgetExceededError
from tcc_docs_ai.provider import EmbeddingProvider
from tcc_docs_ai.schemas import IndexPageVersion, IndexResult, IndexStatus
from tcc_docs_ai.vectorstore import IndexedChunk, PageIndexState, VectorIndex


async def embed_in_batches(
    embedder: EmbeddingProvider, texts: Sequence[str], *, batch_size: int
) -> list[list[float]]:
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = await embedder.embed(list(texts[start : start + batch_size]))
        vectors.extend(batch.embeddings)
    return vectors


class IndexingService:
    """Indexa versões de página no índice vetorial, sem nunca reindexar.

    Regras (requisito do TCC):

    1. Cada versão é um registro novo e imutável (`pageId:version:chunk`).
    2. Versão já indexada é ignorada: nenhuma chamada de embedding, nenhuma
       escrita. Reenviar o mesmo lote é seguro e barato.
    3. Versão menor que a maior já indexada é ignorada: a maior é a atual.
    4. Exclusão é uma versão também (lápide), não um DELETE no índice.
    """

    def __init__(
        self,
        *,
        index: VectorIndex,
        embedder: EmbeddingProvider,
        model: str,
        chunk_max_chars: int,
        max_embedding_inputs: int,
        max_text_bytes: int,
    ) -> None:
        self._index = index
        self._embedder = embedder
        self.model = model
        self._chunk_max_chars = chunk_max_chars
        self._batch = max_embedding_inputs
        self._max_text_bytes = max_text_bytes

    async def states(self, tenant_id: str, page_ids: Sequence[str]) -> dict[str, PageIndexState]:
        return await self._index.page_states(tenant_id, page_ids)

    async def index(self, tenant_id: str, pages: Sequence[IndexPageVersion]) -> list[IndexResult]:
        for page in pages:
            if len(page.text.encode("utf-8")) > self._max_text_bytes:
                raise BudgetExceededError(
                    "Page text exceeds the indexing budget",
                    details={"pageId": page.page_id, "maxTextBytes": self._max_text_bytes},
                )
        states = await self._index.page_states(tenant_id, [page.page_id for page in pages])
        results: list[IndexResult] = []
        # Dentro do mesmo lote, a maior versão de cada página vence também: o lote
        # é percorrido da maior para a menor versão, e as menores nem são embedadas.
        planned: dict[str, int] = {}
        for page in sorted(pages, key=lambda item: (item.page_id, -item.version)):
            state = states.get(page.page_id)
            latest = max(state.latest_version if state else 0, planned.get(page.page_id, 0))
            if state and page.version in state.versions:
                results.append(self._result(page, IndexStatus.ALREADY_INDEXED))
                continue
            if page.version <= latest:
                status = (
                    IndexStatus.ALREADY_INDEXED
                    if page.version == latest
                    else IndexStatus.OLDER_THAN_LATEST
                )
                results.append(self._result(page, status))
                continue

            chunks = (
                []
                if page.deleted
                else chunk_markdown(page.text, title=page.title, max_chars=self._chunk_max_chars)
            )
            indexed = [
                IndexedChunk(
                    page_id=page.page_id,
                    version=page.version,
                    index=chunk.index,
                    title=page.title,
                    space_id=page.space_id,
                    text=chunk.text,
                )
                for chunk in chunks
            ]
            vectors = await embed_in_batches(
                self._embedder, [chunk.text for chunk in indexed], batch_size=self._batch
            )
            await self._index.add_version(
                tenant_id,
                page_id=page.page_id,
                version=page.version,
                title=page.title,
                deleted=page.deleted,
                chunks=indexed,
                vectors=vectors,
            )
            planned[page.page_id] = page.version
            results.append(self._result(page, IndexStatus.INDEXED, len(indexed)))
        by_key = {(result.page_id, result.version): result for result in results}
        return [by_key[(page.page_id, page.version)] for page in pages]

    @staticmethod
    def _result(page: IndexPageVersion, status: IndexStatus, chunks: int = 0) -> IndexResult:
        return IndexResult(pageId=page.page_id, version=page.version, status=status, chunks=chunks)
