# AGENTS.md — tcc-docs-ai

Guia para agentes que forem trabalhar neste repositório. Leia antes de mexer.

## O que é

Serviço de IA do TCC (FastAPI, Python 3.12). Responde a perguntas sobre os
documentos do `tcc-docs-app`, mantém o índice vetorial versionado (Redis 8
Vector Sets) e faz a escrita assistida. O pacote Python
se chama `tcc_docs_ai`.

Leia nesta ordem: `README.md` → `docs/tcc/README.md` → `docs/tcc/03-decisoes.md`.

## Regras do projeto (vindas do dono do TCC)

- Commit só quando o dono pedir. Nada de configuração de git ou publicação
  por conta própria. Construa, teste e documente.
- Não adicione ferramentas ou dependências novas. Use o que já está em
  `pyproject.toml` (FastAPI, pydantic, redis-py, langchain-core, mcp, openai…).
- Não use credenciais de outros projetos. O projeto tem chave própria da
  OpenAI, configurada pelo usuário em `.env` (chmod 600, fora de qualquer
  versionamento). **Nunca copie o valor** para documentação, `.env.example`,
  testes, logs ou para o chat. Modelo de chat: `gpt-6-luna`; embeddings:
  `text-embedding-3-small` (1536 dimensões).
- Os testes unitários ignoram o `.env` e as variáveis `OPENAI_*`/`AI_*`
  (fixture `isolated_settings_env` em `tests/conftest.py`). Só
  `tests/integration` usa o ambiente real. Sem chave, os provedores `hashing` e
  `extractive` continuam funcionando.
- Toda mudança ou decisão nova entra em `docs/tcc/` (atualize
  `02-modulos.md` e, se for decisão, `03-decisoes.md`).
- Comunicação e documentação em português do Brasil.

## Invariantes que não podem quebrar

1. **Temperatura 0** em toda chamada ao modelo (`config.py` recusa outro valor).
2. **Índice append-only:** a identidade de um trecho é `pageId:version:chunk`.
   Nunca regrave nem apague elementos de outra versão. Nunca gere embeddings de
   uma versão que já está em `…:versions`.
3. **Maior versão = atual:** o ponteiro `latest` só sobe. A busca descarta
   trechos com `version != latest`.
4. **A conversa só lê.** O conteúdo vindo do MCP é ranqueado em memória e nunca
   gravado no índice. O único escritor é `POST /v1/index/pages`, chamado pelo
   indexador do app.
5. **A defasagem é decidida por código** (`ChatService._check_freshness`),
   comparando a versão do app com a do índice. Não delegue essa decisão ao
   modelo.
6. **Não inventar.** Sem evidência suficiente → `not_found {similar}`, sem
   chamar o modelo. A sentinela `NAO_ENCONTRADO` do modelo vira o mesmo evento.
7. **Allowlist MCP** apenas com leituras: `docs.search_pages`,
   `docs.get_page_context` e `docs.read_page`. A credencial nunca vai para
   prompt, SSE ou log.
8. Os contratos com o app são camelCase (`pageId`, `contextPageId`…). Mudou
   aqui, mude também em `tcc-docs-app/apps/server/src/integrations/tcc-ai/`.

## Mapa do código

| Arquivo | Responsabilidade |
|---|---|
| `src/tcc_docs_ai/app.py` | Fábrica FastAPI (`create_app`), rotas, health e montagem dos serviços. |
| `src/tcc_docs_ai/config.py` | `Settings` (env). |
| `src/tcc_docs_ai/auth.py` | Assinatura HMAC (`sign_request`, `canonical_signature_input`). |
| `src/tcc_docs_ai/chat.py` / `chat_prompt.py` | Conversa RAG + MCP. |
| `src/tcc_docs_ai/indexing.py` / `vectorstore.py` / `chunking.py` | Índice versionado. |
| `src/tcc_docs_ai/mcp.py` | Cliente MCP (Streamable HTTP), allowlist e orçamento. |
| `src/tcc_docs_ai/provider.py` | OpenAI, `hashing`, `extractive` e `sampling_parameters`. |
| `src/tcc_docs_ai/service.py` / `prompt.py` / `retrieval.py` | Escrita assistida (herdada). |
| `src/tcc_docs_ai/idempotency.py` | Replay idempotente no Redis. |

## Como verificar uma mudança

```bash
uv sync --extra dev
uv run ruff check . && uv run ruff format --check .
docker run -d --name tcc-ai-redis -p 6391:6379 redis:8   # uma vez
AI_TEST_REDIS_URL=redis://localhost:6391/15 REDIS_URL_TEST=redis://localhost:6391/14 uv run pytest
```

Resultado esperado hoje: todos passam, exceto 1 pulado
(`test_openai_cache.py`, que exige chave; com `set -a; . ./.env; set +a` ele roda
contra a API real).

Para mudanças que afetem o fluxo app↔IA, rode também o E2E do app
(`tcc-docs-app/scripts/tcc-e2e.py`, ver `tcc-docs-app/docs/tcc/05-testes.md`).

## Armadilhas conhecidas

- O boot não pode depender de internet: o vocabulário do `tiktoken` é
  embutido na imagem (`TIKTOKEN_CACHE_DIR=/opt/tiktoken`). Ao trocar
  `OPENAI_MODEL` para outra família de tokenizer, pré-carregue o vocabulário
  novo no `Dockerfile`.
- O Redis precisa ser 8+. Em versões antigas, `VADD`/`VSIM` não existem e o
  `/health/ready` fica 503.
- Trocar o provedor ou o modelo de embedding muda a chave do índice: o índice
  novo começa vazio (ver `docs/tcc/08-docker-e-railway.md`).
- `pytest` travando na coleta costuma ser `__pycache__` antigo de outra versão
  do Python. Apague `__pycache__`/`.pytest_cache` e rode de novo.
- Ao mexer no `CHAT_SYSTEM_PROMPT`, rode `scripts/chat_variance_probe.py`
  contra a API real (ver D11): temperatura 0 não garante resposta idêntica.
- Com `hashing`, a similaridade depende de palavras em comum. Perguntas com
  sinônimos podem cair em `not_found`. É o comportamento correto do provedor sem
  chave, não um bug.
