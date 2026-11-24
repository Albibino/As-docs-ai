# Docker e Railway

## Imagem

`Dockerfile`:
- base `python:3.12-slim`;
- instala o pacote com `pip install .`, sem usar o `uv.lock`;
- roda como usuário sem privilégio (`65532`);
- `HEALTHCHECK` em `/health/live`;
- entrada com `uvicorn tcc_docs_ai.app:create_app --factory`, na porta `$PORT`;
- vocabulário do tokenizer (`o200k_base`) baixado no build em
  `/opt/tiktoken`. O container sobe em cerca de 2,5 s e não precisa de
  internet para isso.

## Local

| Modo | Comando |
|---|---|
| Só a IA + Redis 8 | `cp .env.example .env && docker compose up --build` (porta 8000) |
| Stack completa do TCC | `cd ../tcc-docs-app && docker compose up --build`. O compose do app constrói esta imagem a partir de `../tcc-docs-ai`. |

O compose usa `AI_EMBEDDING_PROVIDER=hashing` e `AI_CHAT_PROVIDER=extractive`
por padrão, ou seja, roda sem nenhuma chave. Com `OPENAI_API_KEY`,
`AI_EMBEDDING_PROVIDER=openai` e `AI_CHAT_PROVIDER=openai` no `.env` (o do
`tcc-docs-app`, quando a stack completa sobe por lá), o serviço usa o
`gpt-6-luna` e os embeddings `text-embedding-3-small`.

## Railway

`railway.toml`:
- build pelo `Dockerfile`;
- healthcheck em `/health/ready`: exige autenticação configurada, conversa,
  índice e Redis ativos. No deploy padrão, a autenticação é HMAC.

Serviços no projeto Railway, no mesmo ambiente:

1. `tcc-docs-ai`: este repositório.
2. `Redis`: precisa ser **Redis 8+**, por causa dos Vector Sets. Se o template
   gerenciado do Railway estiver numa versão anterior, use um serviço com a
   imagem `redis:8`, um volume em `/data` e o comando
   `redis-server --appendonly yes`.
3. `tcc-docs-app`, `Postgres` e o Redis do app: ver
   `tcc-docs-app/docs/tcc/06-docker-e-railway.md`.

Variáveis do serviço `tcc-docs-ai`:

```text
ENVIRONMENT=production
REDIS_URL=${{Redis.REDIS_URL}}
DOCS_AI_HMAC_SECRET=<mesmo valor do tcc-docs-app>
AI_EMBEDDING_PROVIDER=openai           # hashing para rodar sem chave
AI_CHAT_PROVIDER=openai                # extractive para rodar sem chave
OPENAI_API_KEY=<chave do projeto>      # variável secreta do Railway
OPENAI_MODEL=gpt-6-luna
OPENAI_TEMPERATURE=0
OPENAI_REASONING_EFFORT=none           # obrigatório para temperatura 0 no gpt-6-luna
AI_MCP_CLIENT_ENABLED=true
AI_MCP_SERVER_URL=http://${{tcc-docs-app.RAILWAY_PRIVATE_DOMAIN}}:${{tcc-docs-app.PORT}}/mcp
```

Não é preciso expor domínio público: o app fala com a IA pela rede privada
(`DOCS_AI_URL=http://${{tcc-docs-ai.RAILWAY_PRIVATE_DOMAIN}}:8000` no app).

> Ao trocar `AI_EMBEDDING_PROVIDER` de `hashing` para `openai`, o modelo muda e,
> com ele, a chave do índice. O índice novo começa vazio. Para popular de novo,
> zere `pages.ai_indexed_version` no app. O trigger impede que o marcador
> diminua, então desligue-o só nesse comando:
> `BEGIN; ALTER TABLE pages DISABLE TRIGGER pages_content_version_trigger;
> UPDATE pages SET ai_indexed_version = 0; ALTER TABLE pages ENABLE TRIGGER
> pages_content_version_trigger; COMMIT;`. O indexador reenvia a versão atual
> de cada página para o índice do modelo novo, sem tocar no antigo.
