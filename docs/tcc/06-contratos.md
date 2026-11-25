# Contratos

## HMAC (padrão)

Com `AI_OAUTH_ENABLED=false` ou ausente, todas as rotas `/v1/*` exigem os
cabeçalhos HMAC. Configure `DOCS_AI_HMAC_SECRET` ou um consumidor não vazio em
`AI_CONSUMERS_JSON`; sem segredo, o serviço não aceita chamadas assinadas nem
fica pronto. As outras variáveis OAuth podem ficar vazias nesse modo.

| Cabeçalho | Valor |
|---|---|
| `x-ai-consumer` | `docs-app` (nome do consumidor configurado). |
| `x-ai-timestamp` | Unix em segundos. Tolerância: `AI_HMAC_MAX_CLOCK_SKEW_SECONDS`. |
| `idempotency-key` | Chave usada na assinatura. No chat e na escrita, a mesma chave e o mesmo corpo permitem replay. |
| `x-ai-signature` | `v1=` + HMAC-SHA256 em hexadecimal de `timestamp\nPOST\npath\nidempotency-key\nsha256(body)`. |

O app implementa a mesma assinatura em
`apps/server/src/integrations/tcc-ai/ai-signing.ts`. O teste
`ai-signing.spec.ts` usa um vetor gerado por `tcc_docs_ai.auth.sign_request`.

## OAuth 2.0 Client Credentials (local)

Ative o modo no serviço de IA quando o app e o Keycloak estiverem configurados
para o mesmo corte. `AI_OAUTH_ENABLED=true` substitui HMAC por
`Authorization: Bearer <access_token>` em todas as rotas `/v1/*`, sem fallback.
`DOCS_AI_HMAC_SECRET` pode ficar vazio nesse modo. Exemplo para o Compose local:

```dotenv
AI_OAUTH_ENABLED=true
AI_OIDC_ISSUER=http://localhost:9090/realms/as-docs
AI_OIDC_JWKS_URL=http://keycloak:8080/realms/as-docs/protocol/openid-connect/certs
AI_OAUTH_CLIENTS_JSON={"docs-app-indexer":{"subject":"<sub-indexador>","tenantId":"<workspaceId>"},"docs-app-assistant":{"subject":"<sub-assistente>","tenantId":"<workspaceId>"}}
```

O JSON deve conter exatamente os dois clientes do exemplo e, para cada um,
somente `subject` e `tenantId`, ambos preenchidos. `subject` é o valor exato de
`sub` da conta de serviço, não o nome do cliente nem seu segredo. `tenantId` é
o ID do workspace autorizado e deve corresponder ao da requisição. Em OAuth,
configuração ausente ou inválida impede a criação da aplicação.

A IA verifica localmente a assinatura `RS256` com a chave indicada por `kid`
no JWKS configurado. O `iss` deve bater exatamente com `AI_OIDC_ISSUER`, mesmo
quando `AI_OIDC_JWKS_URL` usa outro host para acesso interno. A audiência
`docs-ai` deve ser o valor de `aud` ou um item da lista. `exp` e `iat` são
obrigatórios; `nbf` é validado quando presente (RFC 7519, §4.1.5). Os tempos são
segundos Unix inteiros, sem tolerância adicional. `sub` e `azp` são obrigatórios
e devem corresponder ao cadastro e à política abaixo. O escopo indicado deve
constar como item de `scope`, separado por espaços.

| Rota | `azp` | Escopo |
|---|---|---|
| `POST /v1/index/pages` | `docs-app-indexer` | `ai:index:write` |
| `POST /v1/index/status` | `docs-app-indexer` | `ai:index:read` |
| `POST /v1/chat/stream` | `docs-app-assistant` | `ai:chat:invoke` |
| `POST /v1/generations/stream` | `docs-app-assistant` | `ai:writing:invoke` |
| `POST /v1/embeddings` | Nenhum | Negada em modo OAuth |

- `401` com `WWW-Authenticate: Bearer` se não houver credencial Bearer;
  `401` com `error="invalid_token"` na indicação se o token não puder ser validado.
- `403` para token validado sem autorização para cliente, conta de serviço,
  escopo, rota ou tenant.
- Chat e escrita ainda exigem um `idempotency-key` não vazio (`400` se ausente,
  vazio ou duplicado). A IA valida a autorização antes de abrir o SSE ou servir
  replay. A identidade de replay deriva de `iss`, `azp` e `sub`, não do token
  específico; os registros HMAC existentes não são aproveitados no corte.

O JWKS é carregado sob demanda, sem seguir redirecionamentos. Por processo,
o cache vale 5 minutos e há pelo menos 30 segundos entre o fim de uma tentativa
e a próxima, inclusive se chegarem vários `kid` desconhecidos. A validação
espera até 2 segundos pelo download, limitado a 64 KiB e 32 chaves. Se o DNS
ou a leitura ainda estiver bloqueado, não inicia outro download em paralelo.
Uma falha não renova as chaves antigas; chaves expiradas não são usadas. O
`/health/ready` valida a configuração do modo ativo e os serviços locais, sem
consultar o Keycloak. Use HTTPS fora do laboratório. Um issuer HTTPS exige
JWKS HTTPS. No laboratório, com issuer HTTP em `localhost`, o JWKS pode usar
HTTP em `keycloak:8080` ou em loopback.

`mcpCredential` permanece separada, com audiência `tcc-docs-mcp`. A IA não usa
o access token app → IA para chamar o MCP.

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

