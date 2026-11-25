# As-docs-ai

Serviço de IA da versão AWS (rede privada, EKS, só ClusterIP). Deploy: [`deploy/README.md`](deploy/README.md).
Rodar a stack inteira localmente: `docker compose up --build` no repositório **As-docs-app**.

---

# tcc-docs-ai

Serviço de IA do TCC (FastAPI, Python 3.12). Atende o `tcc-docs-app` com
conversa, tirar dúvidas e escrita assistida. Em todas as chamadas ao modelo a
temperatura é **0**, e as respostas ficam restritas ao conteúdo dos documentos.

| Funcionalidade | Endpoint | Observação |
|---|---|---|
| Conversa / tirar dúvidas | `POST /v1/chat/stream` | RAG sobre o índice vetorial + leitura MCP das páginas desatualizadas (SSE). |
| Escrita assistida | `POST /v1/generations/stream` | Propostas de operações por bloco (SSE). Exige chave da OpenAI. |
| Indexação versionada | `POST /v1/index/pages` | Append-only: nunca reindexa uma versão já indexada. |
| Estado do índice | `POST /v1/index/status` | Maior versão indexada por página. |
| Embeddings | `POST /v1/embeddings` | API interna. |

Por padrão, as rotas `/v1/*` exigem assinatura HMAC. Com
`AI_OAUTH_ENABLED=true`, passam a exigir um token OAuth de Client Credentials,
sem fallback para HMAC. Veja a [configuração e as permissões por rota](docs/tcc/06-contratos.md#oauth-20-client-credentials-local).

O banco vetorial é o **Redis 8 (Vector Sets)**, o mesmo Redis que já guardava a
idempotência. Decisões, contratos e o fluxo completo estão em
[`docs/tcc/`](docs/tcc/README.md). Orientações para agentes estão em
[`AGENTS.md`](AGENTS.md).

## Rodando

A stack completa (app + IA + Postgres + Redis) fica no `tcc-docs-app`:

```bash
cd ../tcc-docs-app && cp .env.example .env && docker compose up --build
```

Só a IA, isolada:

```bash
cp .env.example .env          # defina DOCS_AI_HMAC_SECRET
docker compose up --build     # ai + redis:8
curl -s localhost:8000/health/ready
```

**Com chave** (configuração atual): `OPENAI_API_KEY` no `.env`,
`AI_EMBEDDING_PROVIDER=openai`, `AI_CHAT_PROVIDER=openai` e
`OPENAI_MODEL=gpt-6-luna`, com `OPENAI_TEMPERATURE=0` e
`OPENAI_REASONING_EFFORT=none` (o `gpt-6-luna` só aceita temperatura com
raciocínio desligado).

**Sem chave**, use `AI_EMBEDDING_PROVIDER=hashing` e
`AI_CHAT_PROVIDER=extractive`, que são o padrão do `docker-compose.yml` e do
`.env.example`. Os dois são determinísticos: a conversa responde copiando os
trechos mais relevantes, com citação, e nunca gera texto novo. A escrita
assistida exige chave.

## Testes

```bash
uv sync --extra dev
uv run ruff check . && uv run ruff format --check .
uv run pytest                                            # unitários (sem rede)
docker run -d --name tcc-ai-redis -p 6391:6379 redis:8
AI_TEST_REDIS_URL=redis://localhost:6391/15 REDIS_URL_TEST=redis://localhost:6391/14 uv run pytest
```
