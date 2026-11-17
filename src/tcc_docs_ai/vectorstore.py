from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from redis.asyncio import Redis
from redis.exceptions import RedisError

from tcc_docs_ai.errors import UpstreamUnavailableError


@dataclass(frozen=True)
class IndexedChunk:
    """Um elemento do índice: trecho de uma versão específica de uma página."""

    page_id: str
    version: int
    index: int
    title: str
    space_id: str | None
    text: str

    @property
    def element_id(self) -> str:
        # Identidade imutável: a versão faz parte da chave, então indexar a
        # versão N+1 nunca toca os elementos da versão N.
        return f"{self.page_id}:{self.version}:{self.index}"

    def attributes(self) -> dict[str, Any]:
        return {
            "pageId": self.page_id,
            "version": self.version,
            "index": self.index,
            "title": self.title,
            "spaceId": self.space_id,
            "text": self.text,
        }

    @classmethod
    def from_attributes(cls, attributes: dict[str, Any]) -> IndexedChunk:
        return cls(
            page_id=str(attributes["pageId"]),
            version=int(attributes["version"]),
            index=int(attributes["index"]),
            title=str(attributes.get("title") or ""),
            space_id=attributes.get("spaceId"),
            text=str(attributes.get("text") or ""),
        )


@dataclass(frozen=True)
class VectorHit:
    chunk: IndexedChunk
    #: Similaridade do Redis VSIM em [0, 1]: (cosseno + 1) / 2.
    score: float


@dataclass(frozen=True)
class PageIndexState:
    page_id: str
    latest_version: int
    deleted: bool
    title: str
    versions: tuple[int, ...] = field(default_factory=tuple)


class VectorIndex(Protocol):
    async def page_states(
        self, tenant_id: str, page_ids: Sequence[str]
    ) -> dict[str, PageIndexState]: ...

    async def add_version(
        self,
        tenant_id: str,
        *,
        page_id: str,
        version: int,
        title: str,
        deleted: bool,
        chunks: Sequence[IndexedChunk],
        vectors: Sequence[Sequence[float]],
    ) -> None: ...

    async def search(
        self, tenant_id: str, vector: Sequence[float], *, count: int
    ) -> list[VectorHit]: ...

    async def ping(self) -> bool: ...


def similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Mesma escala do Redis VSIM: (cosseno + 1) / 2, em [0, 1]."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    if norm == 0:
        return 0.0
    return (dot / norm + 1.0) / 2.0


# Marca a versão como indexada e avança o ponteiro "latest" apenas para cima.
# Roda depois do VADD de todos os trechos: se o processo cair no meio, a versão
# continua não-marcada e a próxima tentativa refaz os mesmos elementos (mesma
# chave, mesmo conteúdo), sem duplicar nada.
_COMMIT_VERSION = """
redis.call('SADD', KEYS[2], ARGV[1])
local latest = tonumber(redis.call('HGET', KEYS[1], 'latest') or '0')
local version = tonumber(ARGV[1])
if version > latest then
  redis.call('HSET', KEYS[1], 'latest', ARGV[1], 'deleted', ARGV[2], 'title', ARGV[3])
end
return 1
"""


class RedisVectorIndex:
    """Índice vetorial versionado sobre Redis 8 Vector Sets.

    Layout (tudo por ambiente, modelo de embedding e tenant):

    - `{root}:vectors` — vector set; elemento `pageId:version:chunk` com o
      trecho e seus metadados em atributos JSON;
    - `{root}:page:{pageId}` — hash com `latest`, `deleted` e `title`;
    - `{root}:page:{pageId}:versions` — set das versões já indexadas.

    Nada é sobrescrito nem apagado: cada versão é um registro novo e a maior
    versão é a atual. O modelo entra na chave porque vetores de modelos
    diferentes não são comparáveis.
    """

    def __init__(self, client: Redis, *, key_prefix: str, model: str) -> None:
        self._client = client
        self._prefix = key_prefix.rstrip(":")
        self._model = model

    def _root(self, tenant_id: str) -> str:
        # Hash tag mantém as chaves de um tenant no mesmo slot em Redis Cluster.
        return f"{self._prefix}:{self._model}:{{{tenant_id}}}"

    def _page_keys(self, tenant_id: str, page_id: str) -> tuple[str, str]:
        base = f"{self._root(tenant_id)}:page:{page_id}"
        return base, f"{base}:versions"

    async def ping(self) -> bool:
        try:
            return bool(await self._client.ping())
        except RedisError:
            return False

    async def page_states(
        self, tenant_id: str, page_ids: Sequence[str]
    ) -> dict[str, PageIndexState]:
        unique = list(dict.fromkeys(page_ids))
        if not unique:
            return {}
        try:
            async with self._client.pipeline(transaction=False) as pipe:
                for page_id in unique:
                    meta_key, versions_key = self._page_keys(tenant_id, page_id)
                    pipe.hgetall(meta_key)
                    pipe.smembers(versions_key)
                results = await pipe.execute()
        except RedisError as exc:
            raise UpstreamUnavailableError("Vector index is unavailable") from exc

        states: dict[str, PageIndexState] = {}
        for position, page_id in enumerate(unique):
            meta = {_text(k): _text(v) for k, v in (results[position * 2] or {}).items()}
            versions = tuple(sorted(int(_text(v)) for v in (results[position * 2 + 1] or [])))
            if not meta.get("latest"):
                continue
            states[page_id] = PageIndexState(
                page_id=page_id,
                latest_version=int(meta["latest"]),
                deleted=meta.get("deleted") == "1",
                title=meta.get("title", ""),
                versions=versions,
            )
        return states

    async def add_version(
        self,
        tenant_id: str,
        *,
        page_id: str,
        version: int,
        title: str,
        deleted: bool,
        chunks: Sequence[IndexedChunk],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        vector_key = f"{self._root(tenant_id)}:vectors"
        meta_key, versions_key = self._page_keys(tenant_id, page_id)
        try:
            for chunk, vector in zip(chunks, vectors, strict=True):
                await self._client.execute_command(
                    "VADD",
                    vector_key,
                    "VALUES",
                    len(vector),
                    *[float(value) for value in vector],
                    chunk.element_id,
                    "SETATTR",
                    json.dumps(chunk.attributes(), ensure_ascii=False, separators=(",", ":")),
                )
            await self._client.eval(
                _COMMIT_VERSION,
                2,
                meta_key,
                versions_key,
                str(version),
                "1" if deleted else "0",
                title,
            )
        except RedisError as exc:
            raise UpstreamUnavailableError("Vector index is unavailable") from exc

    async def search(
        self, tenant_id: str, vector: Sequence[float], *, count: int
    ) -> list[VectorHit]:
        vector_key = f"{self._root(tenant_id)}:vectors"
        try:
            if not await self._client.exists(vector_key):
                return []
            raw = await self._client.execute_command(
                "VSIM",
                vector_key,
                "VALUES",
                len(vector),
                *[float(value) for value in vector],
                "WITHSCORES",
                "WITHATTRIBS",
                "COUNT",
                count,
            )
        except RedisError as exc:
            raise UpstreamUnavailableError("Vector index is unavailable") from exc
        return _parse_vsim(raw)


def _text(value: object) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _parse_vsim(raw: Any) -> list[VectorHit]:
    """Aceita a resposta RESP2 (lista plana) e RESP3 (mapa elemento → [score, attrs])."""
    triples: list[tuple[Any, Any]] = []
    if isinstance(raw, dict):
        for value in raw.values():
            score, attrs = value if isinstance(value, list | tuple) else (value, None)
            triples.append((score, attrs))
    else:
        items = list(raw or [])
        for position in range(0, len(items) - 2, 3):
            triples.append((items[position + 1], items[position + 2]))
    hits: list[VectorHit] = []
    for score, attrs in triples:
        if attrs is None:
            continue
        chunk = IndexedChunk.from_attributes(json.loads(_text(attrs)))
        hits.append(VectorHit(chunk=chunk, score=float(_text(score))))
    return hits


class InMemoryVectorIndex:
    """Mesmo contrato do Redis, em memória, para testes determinísticos."""

    def __init__(self) -> None:
        self._vectors: dict[str, dict[str, tuple[IndexedChunk, list[float]]]] = {}
        self._pages: dict[tuple[str, str], dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        self.vadd_calls = 0

    async def ping(self) -> bool:
        return True

    async def page_states(
        self, tenant_id: str, page_ids: Sequence[str]
    ) -> dict[str, PageIndexState]:
        states: dict[str, PageIndexState] = {}
        for page_id in page_ids:
            meta = self._pages.get((tenant_id, page_id))
            if not meta:
                continue
            states[page_id] = PageIndexState(
                page_id=page_id,
                latest_version=meta["latest"],
                deleted=meta["deleted"],
                title=meta["title"],
                versions=tuple(sorted(meta["versions"])),
            )
        return states

    async def add_version(
        self,
        tenant_id: str,
        *,
        page_id: str,
        version: int,
        title: str,
        deleted: bool,
        chunks: Sequence[IndexedChunk],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        async with self._lock:
            store = self._vectors.setdefault(tenant_id, {})
            for chunk, vector in zip(chunks, vectors, strict=True):
                self.vadd_calls += 1
                store[chunk.element_id] = (chunk, list(vector))
            meta = self._pages.setdefault(
                (tenant_id, page_id),
                {"latest": 0, "deleted": False, "title": title, "versions": set()},
            )
            meta["versions"].add(version)
            if version > meta["latest"]:
                meta.update(latest=version, deleted=deleted, title=title)

    async def search(
        self, tenant_id: str, vector: Sequence[float], *, count: int
    ) -> list[VectorHit]:
        scored = [
            VectorHit(chunk=chunk, score=similarity(vector, stored))
            for chunk, stored in self._vectors.get(tenant_id, {}).values()
        ]
        scored.sort(key=lambda hit: (-hit.score, hit.chunk.element_id))
        return scored[:count]

    def element_ids(self, tenant_id: str) -> set[str]:
        return set(self._vectors.get(tenant_id, {}))
