from __future__ import annotations

import hashlib
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from tcc_docs_ai.chat_prompt import NOT_FOUND_SENTINEL, chat_prompt_runnable
from tcc_docs_ai.chunking import chunk_markdown
from tcc_docs_ai.errors import BudgetExceededError, McpUnavailableError, ServiceError
from tcc_docs_ai.idempotency import IdempotencyRecord, IdempotencyStore
from tcc_docs_ai.mcp import GET_PAGE_CONTEXT, READ_PAGE, SEARCH_PAGES, McpClient, McpReadSession
from tcc_docs_ai.provider import EmbeddingProvider, ModelProvider
from tcc_docs_ai.schemas import ChatRequest
from tcc_docs_ai.service import sse
from tcc_docs_ai.tokens import TokenCounter
from tcc_docs_ai.vectorstore import PageIndexState, VectorIndex, similarity


@dataclass(frozen=True)
class ChatTuning:
    top_k: int = 8
    min_score: float = 0.66
    similar_min_score: float = 0.58
    max_similar: int = 5
    max_fresh_reads: int = 3
    max_fresh_chars: int = 12_000
    chunk_max_chars: int = 1_600
    max_input_tokens: int = 500_000


@dataclass(frozen=True)
class Evidence:
    page_id: str
    title: str
    version: int
    #: `index` (trecho do índice na versão atual) ou `mcp_read` (lido do app).
    origin: str
    text: str
    score: float


@dataclass(frozen=True)
class StalePage:
    page_id: str
    title: str
    indexed_version: int | None
    current_version: int


@dataclass
class Retrieval:
    evidence: list[Evidence] = field(default_factory=list)
    similar: list[dict[str, Any]] = field(default_factory=list)
    stale: list[StalePage] = field(default_factory=list)
    freshness_checked: bool = False
    mcp_tool_calls: int = 0
    ignored_outdated_chunks: int = 0


@dataclass(frozen=True)
class PreparedChat:
    idempotency_scope: str
    idempotency_key: str
    cache_key: str
    request: ChatRequest
    replay: IdempotencyRecord | None
    lease: str | None


def _parse_json(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


class ChatService:
    """Conversa sobre os documentos com RAG versionado e leitura MCP.

    Fluxo de uma pergunta:

    1. Busca vetorial no índice. Só valem trechos da MAIOR versão indexada de
       cada página; trechos de versões antigas são descartados.
    2. Verificação de frescor pelo MCP (`docs.get_page_context`): para cada
       página candidata compara a versão do índice com a versão atual no app.
       Também pergunta ao app (`docs.search_pages`) por páginas que o índice
       ainda não conhece.
    3. Página com versão no app maior que a do índice é DESATUALIZADA: seus
       trechos do índice saem da evidência e o conteúdo atual é lido com
       `docs.read_page` (o "mcp-read").
    4. Sem evidência acima do limiar, a resposta é "não encontrei" com as
       páginas semelhantes — sem chamar o modelo, então não há o que inventar.
    5. Com evidência, o modelo responde com temperatura 0, só a partir do
       CONTEXT, citando [n]. Se ele devolver a sentinela, vira "não encontrei".
    """

    def __init__(
        self,
        *,
        index: VectorIndex,
        embedder: EmbeddingProvider,
        provider: ModelProvider,
        idempotency: IdempotencyStore,
        token_counter: TokenCounter,
        mcp_client: McpClient | None,
        tuning: ChatTuning,
    ) -> None:
        self._index = index
        self._embedder = embedder
        self._provider = provider
        self._idempotency = idempotency
        self._tokens = token_counter
        self._mcp = mcp_client
        self._tuning = tuning

    async def prepare(
        self, request: ChatRequest, *, consumer: str, idempotency_key: str, raw_body: bytes
    ) -> PreparedChat:
        scope = f"{consumer}:{request.tenant_id}:chat"
        cache_key = hashlib.sha256(
            f"{consumer}:{request.tenant_id}:{request.session_id}:chat".encode()
        ).hexdigest()
        started = await self._idempotency.start(
            scope, idempotency_key, hashlib.sha256(raw_body).hexdigest()
        )
        return PreparedChat(
            idempotency_scope=scope,
            idempotency_key=idempotency_key,
            cache_key=cache_key,
            request=request,
            replay=started.replay,
            lease=started.lease_token,
        )

    async def stream(self, prepared: PreparedChat) -> AsyncIterator[bytes]:
        if prepared.replay:
            for event in prepared.replay.events:
                yield event
            return
        scope, key, lease = prepared.idempotency_scope, prepared.idempotency_key, prepared.lease
        assert lease is not None

        async def emit(event: str, data: object) -> bytes:
            encoded = sse(event, data)
            await self._idempotency.append(scope, key, lease, encoded)
            return encoded

        try:
            retrieval = await self.retrieve(prepared.request)
            yield await emit("meta", self._meta(retrieval))
            if retrieval.evidence:
                yield await emit("sources", self._sources(retrieval))
            if not retrieval.evidence:
                for event in await self._not_found(retrieval, emit):
                    yield event
            else:
                async for event in self._answer(prepared, retrieval, emit):
                    yield event
            await self._idempotency.complete(scope, key, lease)
        except ServiceError as exc:
            yield await emit("error", exc.payload()["error"])
            await self._idempotency.complete(scope, key, lease)
        except Exception:
            await self._idempotency.abandon(scope, key, lease)
            raise

    # --- recuperação ----------------------------------------------------------

    async def retrieve(self, request: ChatRequest) -> Retrieval:
        tuning = self._tuning
        query = request.question
        previous = [turn.content for turn in request.history if turn.role == "user"]
        if previous:
            # Pergunta de seguimento ("e o prazo?") herda o assunto anterior.
            query = f"{previous[-1]}\n{request.question}"
        query_vector = (await self._embedder.embed([query])).embeddings[0]

        hits = await self._index.search(request.tenant_id, query_vector, count=tuning.top_k * 4)
        candidate_ids = [hit.chunk.page_id for hit in hits]
        if request.context_page_id:
            candidate_ids.append(request.context_page_id)
        states = await self._index.page_states(request.tenant_id, candidate_ids)

        retrieval = Retrieval()
        current_hits = []
        for hit in hits:
            state = states.get(hit.chunk.page_id)
            if not state or state.deleted or hit.chunk.version != state.latest_version:
                # Maior versão = atual. Trecho de versão antiga nunca é evidência.
                retrieval.ignored_outdated_chunks += 1
                continue
            current_hits.append(hit)

        if self._mcp and request.mcp_credential:
            async with self._mcp.session(request.mcp_credential.get_secret_value()) as session:
                await self._check_freshness(
                    session, request, query, query_vector, states, current_hits, retrieval
                )
                retrieval.mcp_tool_calls = session.usage.tool_calls
            retrieval.freshness_checked = True
        else:
            self._collect_index_evidence(current_hits, set(), retrieval)

        retrieval.evidence.sort(key=lambda item: (-item.score, item.page_id))
        retrieval.evidence = retrieval.evidence[: tuning.top_k]
        retrieval.similar = self._similar(current_hits, retrieval)
        return retrieval

    def _collect_index_evidence(
        self, hits: list[Any], stale_ids: set[str], retrieval: Retrieval
    ) -> None:
        for hit in hits:
            if hit.chunk.page_id in stale_ids or hit.score < self._tuning.min_score:
                continue
            retrieval.evidence.append(
                Evidence(
                    page_id=hit.chunk.page_id,
                    title=hit.chunk.title,
                    version=hit.chunk.version,
                    origin="index",
                    text=hit.chunk.text,
                    score=hit.score,
                )
            )

    async def _check_freshness(
        self,
        session: McpReadSession,
        request: ChatRequest,
        query: str,
        query_vector: list[float],
        states: dict[str, PageIndexState],
        current_hits: list[Any],
        retrieval: Retrieval,
    ) -> None:
        tuning = self._tuning
        candidates: list[str] = []
        if request.context_page_id:
            candidates.append(request.context_page_id)
        for hit in current_hits:
            if hit.score >= tuning.similar_min_score:
                candidates.append(hit.chunk.page_id)
        # Páginas que o app encontra pelo texto ATUAL, inclusive as que o índice
        # ainda não tem ou tem só numa versão antiga.
        search = await session.call(SEARCH_PAGES, {"query": request.question, "limit": 5})
        app_hits = _parse_json(search.text).get("pages", []) if not search.is_error else []
        candidates.extend(str(page["pageId"]) for page in app_hits if page.get("pageId"))
        candidates = list(dict.fromkeys(candidates))[:20]
        if not candidates:
            return

        context = await session.call(GET_PAGE_CONTEXT, {"pageIds": candidates})
        if context.is_error:
            raise McpUnavailableError("docs.get_page_context failed")
        current = {str(page["pageId"]): page for page in _parse_json(context.text).get("pages", [])}
        # Estados de páginas que só vieram da busca do app.
        missing = [page_id for page_id in candidates if page_id not in states]
        if missing:
            states.update(await self._index.page_states(request.tenant_id, missing))

        stale_ids: set[str] = set()
        for page_id in candidates:
            app_page = current.get(page_id)
            if not app_page or app_page.get("deleted"):
                # Sem acesso, inexistente ou excluída no app: nunca vira evidência.
                stale_ids.add(page_id)
                continue
            state = states.get(page_id)
            indexed = state.latest_version if state else None
            if indexed is None or int(app_page["version"]) > indexed:
                retrieval.stale.append(
                    StalePage(
                        page_id=page_id,
                        title=str(app_page.get("title") or ""),
                        indexed_version=indexed,
                        current_version=int(app_page["version"]),
                    )
                )
                stale_ids.add(page_id)

        self._collect_index_evidence(current_hits, stale_ids, retrieval)

        for stale in retrieval.stale[: tuning.max_fresh_reads]:
            read = await session.call(READ_PAGE, {"pageId": stale.page_id})
            page = _parse_json(read.text) if not read.is_error else {}
            if not page.get("markdown") and not page.get("title"):
                continue
            await self._rank_fresh_content(
                page, stale, query_vector, request.context_page_id, retrieval
            )

    async def _rank_fresh_content(
        self,
        page: dict[str, Any],
        stale: StalePage,
        query_vector: list[float],
        context_page_id: str | None,
        retrieval: Retrieval,
    ) -> None:
        """Ranqueia em memória o conteúdo lido pelo MCP. Nada vai para o índice.

        A indexação da versão nova é trabalho do app (job versionado); a
        conversa só lê. Assim o índice continua append-only e com um único
        escritor.
        """
        tuning = self._tuning
        markdown = str(page.get("markdown") or "")[: tuning.max_fresh_chars]
        title = str(page.get("title") or stale.title)
        version = int(page.get("version") or stale.current_version)
        chunks = chunk_markdown(markdown, title=title, max_chars=tuning.chunk_max_chars)
        if not chunks:
            return
        vectors = (await self._embedder.embed([chunk.text for chunk in chunks])).embeddings
        scored = sorted(
            (
                (similarity(query_vector, vector), chunk)
                for chunk, vector in zip(chunks, vectors, strict=True)
            ),
            key=lambda item: (-item[0], item[1].index),
        )
        # A página aberta pelo usuário é o assunto da dúvida: aceita o limiar de
        # "semelhante". As demais precisam do limiar normal de evidência.
        threshold = (
            tuning.similar_min_score if stale.page_id == context_page_id else tuning.min_score
        )
        for score, chunk in scored[:3]:
            if score < threshold:
                if score >= tuning.similar_min_score:
                    retrieval.similar.append(
                        {"pageId": stale.page_id, "title": title, "score": score}
                    )
                continue
            retrieval.evidence.append(
                Evidence(
                    page_id=stale.page_id,
                    title=title,
                    version=version,
                    origin="mcp_read",
                    text=chunk.text,
                    score=score,
                )
            )

    def _similar(self, current_hits: list[Any], retrieval: Retrieval) -> list[dict[str, Any]]:
        evidence_pages = {item.page_id for item in retrieval.evidence}
        stale_pages = {item.page_id for item in retrieval.stale}
        best: dict[str, dict[str, Any]] = {}
        for item in [
            *(
                {"pageId": hit.chunk.page_id, "title": hit.chunk.title, "score": hit.score}
                for hit in current_hits
                if hit.score >= self._tuning.similar_min_score
                and hit.chunk.page_id not in stale_pages
            ),
            *retrieval.similar,
        ]:
            if item["pageId"] in evidence_pages:
                continue
            if item["pageId"] not in best or item["score"] > best[item["pageId"]]["score"]:
                best[item["pageId"]] = item
        ranked = sorted(best.values(), key=lambda item: (-item["score"], item["pageId"]))
        return [
            {"pageId": item["pageId"], "title": item["title"], "score": round(item["score"], 4)}
            for item in ranked[: self._tuning.max_similar]
        ]

    # --- resposta -------------------------------------------------------------

    async def _not_found(self, retrieval: Retrieval, emit: Any) -> list[bytes]:
        return [
            await emit("not_found", {"similar": retrieval.similar}),
            await emit("done", {"found": False}),
        ]

    async def _answer(
        self, prepared: PreparedChat, retrieval: Retrieval, emit: Any
    ) -> AsyncIterator[bytes]:
        request = prepared.request
        context = [
            {
                "n": position + 1,
                "pageId": item.page_id,
                "title": item.title,
                "version": item.version,
                "origin": item.origin,
                "text": item.text,
            }
            for position, item in enumerate(retrieval.evidence)
        ]
        messages = chat_prompt_runnable.invoke(
            {
                "context": context,
                "history": [turn.model_dump() for turn in request.history],
                "question": request.question,
            }
        )
        input_tokens = self._tokens.count(json.dumps(messages, ensure_ascii=False))
        if input_tokens > self._tuning.max_input_tokens:
            raise BudgetExceededError(
                "Authoritative input token budget exceeded",
                details={
                    "inputTokens": input_tokens,
                    "maxInputTokens": self._tuning.max_input_tokens,
                },
            )

        state = "pending"  # pending | answer | not_found
        buffer = ""
        async for event in self._provider.stream(
            input_messages=messages, cache_key=prepared.cache_key
        ):
            if event.kind == "usage":
                yield await emit("usage", event.value)
                continue
            if state == "answer":
                yield await emit("delta", {"text": str(event.value)})
                continue
            if state == "not_found":
                continue  # o texto do modelo é descartado; só o usage interessa
            buffer += str(event.value)
            stripped = buffer.lstrip()
            if stripped.startswith(NOT_FOUND_SENTINEL):
                state = "not_found"
            elif NOT_FOUND_SENTINEL.startswith(stripped):
                continue  # ainda pode virar a sentinela: segura o texto
            else:
                state = "answer"
                yield await emit("delta", {"text": buffer})

        if state == "answer":
            yield await emit("done", {"found": True})
            return
        # Sentinela, prefixo dela ou resposta vazia: o modelo não achou no CONTEXT.
        yield await emit("not_found", {"similar": self._merge_similar(retrieval)})
        yield await emit("done", {"found": False})

    def _merge_similar(self, retrieval: Retrieval) -> list[dict[str, Any]]:
        # O modelo julgou a evidência insuficiente: as páginas dela viram as
        # sugestões mais próximas, seguidas das semelhantes já calculadas.
        best: dict[str, dict[str, Any]] = {}
        for item in retrieval.evidence:
            entry = {"pageId": item.page_id, "title": item.title, "score": round(item.score, 4)}
            if item.page_id not in best or entry["score"] > best[item.page_id]["score"]:
                best[item.page_id] = entry
        for entry in retrieval.similar:
            best.setdefault(entry["pageId"], entry)
        ranked = sorted(best.values(), key=lambda item: (-item["score"], item["pageId"]))
        return ranked[: self._tuning.max_similar]

    @staticmethod
    def _meta(retrieval: Retrieval) -> dict[str, Any]:
        return {
            "freshnessChecked": retrieval.freshness_checked,
            "evidenceCount": len(retrieval.evidence),
            "ignoredOutdatedChunks": retrieval.ignored_outdated_chunks,
            "mcpToolCalls": retrieval.mcp_tool_calls,
            "stalePages": [
                {
                    "pageId": item.page_id,
                    "title": item.title,
                    "indexedVersion": item.indexed_version,
                    "currentVersion": item.current_version,
                }
                for item in retrieval.stale
            ],
        }

    @staticmethod
    def _sources(retrieval: Retrieval) -> list[dict[str, Any]]:
        return [
            {
                "n": position + 1,
                "pageId": item.page_id,
                "title": item.title,
                "version": item.version,
                "origin": item.origin,
                "score": round(item.score, 4),
            }
            for position, item in enumerate(retrieval.evidence)
        ]
