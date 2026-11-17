# Testes

## Comandos

```bash
uv sync --extra dev
uv run ruff check . && uv run ruff format --check .
uv run pytest                     # suíte inteira; integrações pulam sem as variáveis

# Integração com Redis 8 real (Vector Sets + idempotência):
docker run -d --name tcc-ai-redis -p 6391:6379 redis:8
AI_TEST_REDIS_URL=redis://localhost:6391/15 \
REDIS_URL_TEST=redis://localhost:6391/14 \
uv run pytest
```

`tests/integration/test_openai_cache.py` usa a API real e fica pulado sem
`OPENAI_API_KEY`. Ele confere que o modelo configurado aceita temperatura 0 e
que o cache de prefixo funciona. Para rodar com a chave do `.env`:

```bash
set -a; . ./.env; set +a
uv run pytest tests/integration
```

Os testes fora de `tests/integration` ignoram o `.env` e as variáveis
`OPENAI_*`/`AI_*` (fixture `isolated_settings_env`), então dão o mesmo
resultado com ou sem chave.

`scripts/chat_variance_probe.py` não é teste automatizado: roda a mesma
pergunta várias vezes contra a API e conta as respostas distintas (ver D11).

## Cobertura das regras do TCC

| Regra | Teste |
|---|---|
| Documento desatualizado no índice e atualizado no app → `docs.read_page` | `test_chat.py::test_stale_index_is_detected_and_page_is_read_through_mcp` |
| Índice em dia → sem leitura MCP | `test_chat.py::test_up_to_date_index_answers_without_mcp_read` |
| Só a maior versão é evidência | `test_chat.py::test_only_the_highest_indexed_version_is_evidence` |
| "Não encontrado" sem chamar o modelo, com semelhantes | `test_chat.py::test_not_found_does_not_call_the_model_and_lists_similar_pages` |
| Sentinela do modelo vira "não encontrado" | `test_chat.py::test_model_sentinel_becomes_not_found_with_similar` |
| Página fora do índice achada pela busca do app | `test_chat.py::test_page_missing_from_index_is_found_through_app_search` |
| Página excluída no app nunca é evidência | `test_chat.py::test_page_deleted_in_app_is_never_evidence` |
| Página aberta é checada mesmo sem ranquear | `test_chat.py::test_context_page_is_checked_even_when_not_ranked` |
| Provedor sem chave só copia o contexto | `test_chat.py::test_extractive_provider_only_copies_context` |
| Temperatura 0 em toda chamada | `test_chat.py::test_every_model_call_uses_temperature_zero`, `test_rag_api.py::test_temperature_other_than_zero_is_rejected` |
| Nunca reindexa uma versão | `test_indexing.py::test_never_reindexes_an_already_indexed_version` |
| Versão nova é anexada e a antiga mantida | `test_indexing.py::test_new_version_is_appended_and_old_version_is_kept` |
| Versão menor nunca substitui a maior | `test_indexing.py::test_older_version_never_replaces_latest`, `…same_batch_keeps_the_highest_version_as_latest` |
| Exclusão é lápide versionada | `test_indexing.py::test_deletion_is_a_versioned_tombstone` |
| Isolamento por tenant | `test_indexing.py::test_tenants_are_isolated`, `integration/test_redis_vector_index.py::test_redis_index_is_isolated_by_tenant` |
| Append-only no Redis real | `integration/test_redis_vector_index.py::test_redis_index_is_append_only_and_versioned` |
| Contrato HTTP (camelCase) e assinatura obrigatória | `test_rag_api.py::test_index_then_chat_over_http`, `…routes_require_signature` |
| Allowlist MCP (só leitura) | `test_mcp.py` |

## Teste ponta a ponta

O cenário completo, com o app de verdade (Postgres, trigger de versão, servidor
MCP, indexador e chat), está em `tcc-docs-app/scripts/tcc-e2e.py`. Ver
`tcc-docs-app/docs/tcc/05-testes.md`. Última execução: `E2E OK`, com estes
pontos verificados:
- defasagem "indexada v1, atual v2" detectada;
- fonte `mcp_read` usada;
- "não encontrado" sem nenhum texto;
- indexação incremental até a v2 sem nenhuma reindexação;
- no Redis, o conjunto de versões da página ficou `{1, 2}`, com os trechos das
  duas versões preservados.
