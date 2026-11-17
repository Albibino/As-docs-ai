# Conversa, defasagem e leitura via MCP

## Fluxo de `POST /v1/chat/stream` (`ChatService`)

```
pergunta ──► embedding ──► VSIM no índice (top_k × 4)
                                 │
                 descarta trechos com version ≠ latest
                                 │
          ┌── há credencial MCP? ──┐
          │ sim                    │ não
          ▼                        ▼
  candidatos = página aberta     evidência = trechos do índice ≥ min_score
             + hits ≥ similar_min_score
             + docs.search_pages(pergunta)      (texto ATUAL do app)
          │
  docs.get_page_context(candidatos)   → versão atual de cada página
          │
  para cada candidato:
    sem acesso / excluída no app  → nunca é evidência
    versão atual > versão indexada → DESATUALIZADA
          │
  evidência do índice = só páginas em dia
  para até AI_RAG_MAX_FRESH_READS desatualizadas:
    docs.read_page(pageId)  ("mcp-read")
    → corta, gera embeddings em memória e ranqueia
    → trechos ≥ limiar viram evidência com origin = "mcp_read"
          │
  sem evidência ──► not_found {similar} ; done {found:false}   (modelo não é chamado)
  com evidência ──► modelo (temperatura 0) ──► delta… ; done {found:true}
                         └─ responde NAO_ENCONTRADO ──► not_found {similar}
```

### Limiar da página aberta

Quando o usuário pergunta com uma página aberta (`contextPageId`), essa página
é o assunto da dúvida. Ela entra como candidata mesmo sem ranquear no índice, e
o conteúdo lido pelo MCP aceita o limiar de "semelhante" (`0.58`) em vez do de
evidência (`0.66`).

### Páginas novas ainda não indexadas

`docs.search_pages` busca no texto atual do app. Por isso uma página criada
agora, ainda fora do índice, aparece como candidata com `indexedVersion: null`,
é marcada como desatualizada e lida pelo MCP.

## Ferramentas MCP (allowlist em `mcp.py`)

| Tool | Entrada | Saída |
|---|---|---|
| `docs.search_pages` | `{query, limit?}` | `{pages: [{pageId, title, version}]}` |
| `docs.get_page_context` | `{pageIds: [...]}` | `{pages: [{pageId, title, version, indexedVersion, deleted, updatedAt}]}` |
| `docs.read_page` | `{pageId}` | `{pageId, title, version, markdown, truncated}` |

Qualquer outro nome de tool é recusado antes da chamada
(`McpToolForbiddenError`). Há orçamento de chamadas
(`AI_MCP_MAX_TOOL_CALLS`), tempo por chamada e total, e tamanho máximo de
resposta.

A identidade vem da credencial (`Authorization: Bearer`) emitida pelo app a
cada conversa. Ela é curta, somente leitura e vinculada ao usuário e ao
workspace. A credencial nunca entra no prompt, nos eventos SSE ou nos logs.

## Prompt (`chat_prompt.py`)

- Responde só com base em `CONTEXT`, sem conhecimento externo.
- Cita as fontes com `[n]`.
- Se o contexto não responde à pergunta, a resposta é exatamente
  `NAO_ENCONTRADO`.
- Trechos com `origin = mcp_read` são a versão atual. Se houver conflito com
  trechos do índice, a versão atual prevalece.
