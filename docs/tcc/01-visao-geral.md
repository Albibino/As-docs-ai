# Visão geral

## Papel no sistema

```
 navegador ──► tcc-docs-app (NestJS) ──HMAC ou OAuth──► tcc-docs-ai (FastAPI)
                 │  ▲                               │      │
                 │  └──── MCP /mcp (Bearer) ◄───────┘      │
                 ▼                                         ▼
             Postgres                              Redis 8 (Vector Sets)
       (páginas + content_version)           (índice versionado + idempotência)
```

- **Indexação:** o app envia a versão atual de cada página alterada para
  `POST /v1/index/pages`. O serviço quebra o Markdown em trechos, gera os
  embeddings e grava cada trecho como um elemento novo do vector set.
- **Conversa / tirar dúvidas:** o app repassa a pergunta para
  `POST /v1/chat/stream`, junto com uma credencial MCP curta e somente leitura.
  Em seguida:
  1. o serviço busca no índice e descarta trechos de versões antigas;
  2. pelo MCP do app, compara a versão indexada com a versão atual de cada
     página candidata;
  3. se a página estiver desatualizada no índice, lê o conteúdo atual com
     `docs.read_page` e usa esse conteúdo como evidência;
  4. responde **somente** com base na evidência. Sem evidência suficiente,
     emite `not_found` com as páginas semelhantes.
- **Escrita assistida:** fluxo de propostas por bloco (`/v1/generations/stream`),
  agora com temperatura 0 e um prompt que proíbe inventar fatos.

## Garantias

| Requisito do TCC | Onde é garantido |
|---|---|
| Temperatura zero | `config.py` (`openai_temperature` com `ge=0, le=0`), `provider.sampling_parameters`; teste `test_every_model_call_uses_temperature_zero`. |
| Não reindexar; só versões novas | `indexing.py` + `vectorstore.py` (chave `pageId:version:chunk`, conjunto de versões); testes `test_never_reindexes_an_already_indexed_version`, `test_redis_index_is_append_only_and_versioned`. |
| Maior versão = atual | Ponteiro `latest` só sobe (script Lua `_COMMIT_VERSION`); `ChatService.retrieve` descarta trechos com `version != latest`. |
| Documento desatualizado no índice → `mcp-read` | `ChatService._check_freshness`; teste `test_stale_index_is_detected_and_page_is_read_through_mcp`. |
| Não inventar | Prompt `chat_prompt.py`, sentinela `NAO_ENCONTRADO`, provedor `extractive` sem chave. |
| "Não encontrei" + semelhantes | Evento `not_found {similar}`; testes `test_not_found_*` e `test_model_sentinel_becomes_not_found_with_similar`. |
