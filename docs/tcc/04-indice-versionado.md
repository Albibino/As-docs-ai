# Índice versionado (Redis 8 Vector Sets)

## Layout

Raiz: `tcc-docs-ai:{ENVIRONMENT}:vindex:{modelo}:{tenantId}`, com o tenant entre chaves
(hash tag, para manter tudo no mesmo slot em Redis Cluster).

| Chave | Tipo | Conteúdo |
|---|---|---|
| `{raiz}:vectors` | vector set | Elemento `pageId:version:chunk`. Atributos JSON: `pageId`, `version`, `index`, `title`, `spaceId`, `text`. |
| `{raiz}:page:{pageId}` | hash | `latest` (maior versão), `deleted` (`0`/`1`) e `title`. |
| `{raiz}:page:{pageId}:versions` | set | Versões já indexadas. |

## Regras de escrita (`IndexingService.index`)

1. Agrupa o lote por página e processa as versões **da maior para a menor**.
2. Se a versão já está em `:versions`, o resultado é `already_indexed`. Não há
   embedding nem escrita.
3. Se a versão é menor que `latest`, o resultado é `older_than_latest`. Também
   não há escrita.
4. Senão, corta o Markdown (`chunk_markdown`), gera os embeddings em lotes de
   `AI_MAX_EMBEDDING_INPUTS` e faz um `VADD` por trecho.
5. Por fim, um script Lua marca a versão e sobe `latest` **só se** ela for
   maior. Se o processo cair entre os `VADD` e o commit, a próxima tentativa
   regrava os mesmos elementos (mesma chave, mesmo conteúdo), sem duplicar.
6. Uma exclusão é uma versão também (`deleted: true`, sem trechos). Ela vira a
   `latest` e a página deixa de servir de evidência.

## Regras de leitura

- `VSIM … WITHSCORES WITHATTRIBS` devolve os trechos mais próximos. O score fica
  em `[0, 1]` (`(cosseno+1)/2`).
- `ChatService.retrieve` descarta todo trecho cuja `version` não seja a `latest`
  da página, ou cuja página esteja `deleted`. É isso que garante "maior versão =
  atual" mesmo com o histórico inteiro no índice.

## Configuração

| Variável | Padrão | Uso |
|---|---|---|
| `ENVIRONMENT` | `development` | Entra no prefixo `tcc-docs-ai:{ENVIRONMENT}:vindex`. |
| `AI_CHUNK_MAX_CHARS` | 1600 | Tamanho máximo do trecho. |
| `AI_MAX_INDEX_TEXT_BYTES` | 2.000.000 | Limite por página no `/v1/index/pages`. |
| `AI_EMBEDDING_PROVIDER` | `openai` (compose: `hashing`) | Provedor de embeddings. |
