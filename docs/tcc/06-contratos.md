# Contratos

Todas as rotas `/v1/*` exigem os cabeçalhos HMAC:

| Cabeçalho | Valor |
|---|---|
| `x-ai-consumer` | `docs-app` (nome do consumidor configurado). |
| `x-ai-timestamp` | Unix em segundos. Tolerância: `AI_HMAC_MAX_CLOCK_SKEW_SECONDS`. |
| `idempotency-key` | UUID. Uma repetição com a mesma chave e o mesmo corpo é servida do cache. |
| `x-ai-signature` | `v1=` + HMAC-SHA256 em hexadecimal de `timestamp\nPOST\npath\nidempotency-key\nsha256(body)`. |

O app implementa a mesma assinatura em
`apps/server/src/integrations/tcc-ai/ai-signing.ts`. O teste
`ai-signing.spec.ts` usa um vetor gerado por `tcc_docs_ai.auth.sign_request`.

## `POST /v1/index/pages`

```json
{
  "tenantId": "<workspaceId>",
  "pages": [
    {
      "pageId": "…",
      "version": 3,
      "title": "Política de férias",
      "spaceId": "…",
      "text": "## Prazo\nPedir com 45 dias…",
      "deleted": false
    }
  ]
}
```

Resposta:

```json
{
  "model": "hashing-v1-256",
  "results": [{ "pageId": "…", "version": 3, "status": "indexed", "chunks": 2 }]
}
```

`status` pode ser `indexed`, `already_indexed` ou `older_than_latest`. O app
avança `pages.ai_indexed_version` com o `version` de qualquer um dos três.

## `POST /v1/index/status`

`{ "tenantId": "…", "pageIds": ["…"] }` →
`{ "model": "…", "pages": [{ "pageId", "latestVersion", "deleted", "versions" }] }`

## `POST /v1/chat/stream`

```json
{
  "tenantId": "<workspaceId>",
  "sessionId": "<chatId>",
  "question": "Com quantos dias peço férias?",
  "history": [{ "role": "user", "content": "…" }, { "role": "assistant", "content": "…" }],
  "contextPageId": "<página aberta, opcional>",
  "mcpCredential": "<JWT curto, somente leitura, opcional>"
}
```

A resposta é SSE (`event:` + `data:`), na ordem abaixo:

| Evento | Dados |
|---|---|
| `meta` | `{ freshnessChecked, evidenceCount, ignoredOutdatedChunks, mcpToolCalls, stalePages: [{pageId, title, indexedVersion, currentVersion}] }` |
| `sources` | `[{ n, pageId, title, version, origin: "index" \| "mcp_read", score }]` (só quando há evidência). |
| `delta` | `{ text }` (zero ou mais). |
| `usage` | Tokens do provedor, quando houver. |
| `not_found` | `{ similar: [{ pageId, title, score }] }` |
| `done` | `{ found: true \| false }` |
| `error` | `{ code, message, … }` |

O app traduz esses eventos para o formato do cliente: `freshness`, `sources`,
`content`, `not_found` e `done`.

## `POST /v1/generations/stream` (escrita assistida)

Corpo: `tenantId`, `sessionId`, `request`, `blocks`
(com `hash` SHA-256 do JSON canônico de cada bloco) e, opcionalmente,
`selectedBlockIds`, `history`, `title`, `labels`, `availableLabels` e
`spacePages`. Resposta em SSE:

| Evento | Payload |
|---|---|
| `meta` | `{ inputTokens, mode }` |
| `usage` | `{ inputTokens, cachedInputTokens, outputTokens }` (pode faltar quando a proposta termina cedo) |
| `operation` | Uma operação: `replace`, `delete`, `insert_after`, `set_title`, `add_label` ou `remove_label`. |
| `done` | `{ operationCount }` ou, quando a IA decide não alterar nada, `{ operationCount: 0, noChangeReason }`. |
| `error` | `{ code, message, details }`; ex.: `invalid_model_proposal`. |

`noChangeReason` só aparece com `operationCount: 0` (ver D12).

