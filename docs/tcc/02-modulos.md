# Módulos do serviço

O código fica no pacote Python `tcc_docs_ai` (`src/tcc_docs_ai`), distribuído
como `tcc-docs-ai` 0.2.0.

## Conversa e índice versionado (RAG)

| Arquivo | Para quê |
|---|---|
| `src/tcc_docs_ai/chunking.py` | `chunk_markdown`: corta por heading, prefixa título e caminho dos headings em cada trecho e respeita `AI_CHUNK_MAX_CHARS`. É determinístico. |
| `src/tcc_docs_ai/vectorstore.py` | `RedisVectorIndex` (Redis 8 Vector Sets) e `InMemoryVectorIndex` (testes), com a identidade `pageId:version:chunk`. |
| `src/tcc_docs_ai/indexing.py` | `IndexingService`: decide o que indexar (só versões novas e maiores que a atual) e grava. |
| `src/tcc_docs_ai/chat.py` | `ChatService`: recuperação, detecção de defasagem, leitura MCP, resposta e "não encontrado". |
| `src/tcc_docs_ai/chat_prompt.py` | Prompt da conversa (contexto numerado, citações `[n]` e a sentinela `NAO_ENCONTRADO`). Regras calibradas no modelo real: texto completo, conflito entre páginas e `mcp_read` só na mesma página (D11). |
| `src/tcc_docs_ai/mcp.py` | Cliente MCP com allowlist das três leituras: `docs.search_pages`, `docs.get_page_context` e `docs.read_page`. |

## Escrita assistida

| Arquivo | Para quê |
|---|---|
| `src/tcc_docs_ai/prompt.py` | Proíbe fatos fora de `PAGE_SNAPSHOT`, `REQUEST` ou dos dados recuperados e responde `{"action":"none","reason":...}` quando nada deve mudar (D12). |
| `src/tcc_docs_ai/service.py` | Geração das propostas por bloco. A sentinela `none` vira `done {operationCount: 0, noChangeReason}`; token de controle vazado encerra a proposta e fecha o stream; operações duplicadas são descartadas (D12). |
| `src/tcc_docs_ai/retrieval.py` | Laço de leitura MCP entre o modelo e o servidor MCP do app. |

## Base comum

| Arquivo | Para quê |
|---|---|
| `src/tcc_docs_ai/app.py` | Rotas `/v1/chat/stream`, `/v1/index/pages`, `/v1/index/status`, `/v1/generations/stream`, `/v1/embeddings` e healthchecks. `/health/ready` não depende da chave da OpenAI. |
| `src/tcc_docs_ai/config.py` | Temperatura travada em 0; `OPENAI_REASONING_EFFORT` (padrão `none`); provedores `AI_EMBEDDING_PROVIDER` (`openai`/`hashing`) e `AI_CHAT_PROVIDER` (`openai`/`extractive`); parâmetros do RAG (`AI_RAG_*`, `AI_CHUNK_MAX_CHARS`). |
| `src/tcc_docs_ai/provider.py` | `sampling_parameters` (temperatura + `reasoning.effort`) aplicado em toda chamada; `HashingEmbeddingProvider` e `ExtractiveAnswerProvider` (sem chave). |
| `src/tcc_docs_ai/schemas.py` | Contratos HTTP: `IndexPageVersion`, `IndexRequest/Response`, `IndexStatus*`, `ChatRequest` e os da escrita assistida. |
| `src/tcc_docs_ai/auth.py` | Assinatura HMAC das rotas `/v1/*`, modo padrão. |
| `src/tcc_docs_ai/oauth.py` | Validação RS256 pelo JWKS configurado, cache de chaves e autorização por conta de serviço, rota, escopo e tenant. Ativo com `AI_OAUTH_ENABLED=true` (D14). |
| `src/tcc_docs_ai/oidc_config.py` | Validação de issuer, JWKS e vínculos de clientes compartilhada pela aplicação e pelo renderizador AWS; restringe a exceção HTTP do laboratório (D15). |
| `src/tcc_docs_ai/idempotency.py` | Idempotência das requisições no Redis. |
| `src/tcc_docs_ai/tokens.py` | Contagem de tokens; `gpt-6-luna` mapeado para `o200k_base` (D10). |
| `src/tcc_docs_ai/errors.py` | Erros tipados da API. |

## Testes, scripts e implantação

| Arquivo | Para quê |
|---|---|
| `tests/rag_fixtures.py`, `tests/test_indexing.py`, `tests/test_chat.py`, `tests/test_rag_api.py`, `tests/integration/test_redis_vector_index.py` | Testes do RAG e do índice versionado. |
| `tests/test_service_model_output.py` | Saídas reais do `gpt-6-luna` na escrita assistida. |
| `tests/conftest.py` | Fixture `isolated_settings_env`: testes unitários independem do `.env` e da chave. |
| `tests/integration/test_openai_cache.py` | Provedor com os parâmetros reais do serviço; teste de temperatura 0 contra a API. |
| `scripts/chat_variance_probe.py`, `scripts/writing_variance_probe.py` | Sondas de variação das respostas e da escrita assistida contra a API (manuais). |
| `scripts/try_generation.py` | Cliente de teste: assina a requisição e imprime o SSE cru. |
| `Dockerfile` | Usuário sem privilégio (65532), healthcheck, `uvicorn --factory` e vocabulário do `tiktoken` embutido (`TIKTOKEN_CACHE_DIR`), ver D9. |
| `docker-compose.yml` | Serviço `ai` + `redis:8` para rodar isolado. |
| `deploy/render-aws-oauth.sh`, `deploy/aws-oauth.yaml.template` | Manifests do EKS: Service privado, OAuth e sidecar TLS na 8443 sem capabilities. `deploy/test-aws-render.sh` valida o YAML sem cluster. |
| `deploy/aws-lab-build-inputs.json` | Lista versionada de entradas da imagem da IA, gerada pelo publicador do app e usada para calcular a tag imutável do Learner Lab. |
| `railway.toml` | Healthcheck em `/health/ready`. |
| `.env.example` | Sem segredos, com os provedores sem chave como padrão. |
