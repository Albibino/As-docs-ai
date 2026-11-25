# Deploy do As-docs-ai no EKS (subnet privada)

## AWS Lab portátil — OAuth Client Credentials

Renderize na raiz do As-docs-ai com `bash deploy/render-aws-oauth.sh`.
O script usa Bash e Python 3.11+, escreve apenas os manifests em `stdout`
e não consulta o cluster, a AWS nem arquivos de credenciais.

| Variável de entrada | Valor |
| --- | --- |
| `AI_IMAGE`, `CADDY_IMAGE` | Imagens no formato `repositorio@sha256:<64 hex minúsculos>` (sem tag). A imagem da IA precisa incluir o suporte a `AI_OIDC_AWS_LAB_HTTP` e `AI_OIDC_CA_FILE`. |
| `OIDC_ISSUER` | Issuer exato anunciado pelo realm, incluindo seu caminho. HTTPS é o padrão de transporte público. |
| `OIDC_AWS_LAB_HTTP` | `false` por padrão. `true` exige o issuer `http://as-docs-public-<número>.us-east-1.elb.amazonaws.com/id/realms/as-docs`, sem porta ou barra adicional. A exceção é só para o `iss`; o JWKS continua privado e em HTTPS. |
| `AI_OAUTH_CLIENTS_JSON` | Objeto com `docs-app-indexer` e `docs-app-assistant`, cada um com os campos `subject` (o `sub` da **conta de serviço**) e `tenantId` (workspace). Os quatro valores têm formato UUID. Não inclua segredos. |
| `AI_PROVIDER_MODE` | `openai` por padrão. `diagnostic` usa embeddings `hashing` e chat `extractive` no namespace Redis `aws-lab-diagnostic`. Não precisa de chave, mantém o MCP ligado e não oferece escrita assistida por modelo. Ao mudar de modo, reindexe no namespace correspondente. |

Exemplo de formato dos vínculos (substitua os UUIDs pelos da instalação):

```bash
export AI_OAUTH_CLIENTS_JSON='{"docs-app-indexer":{"subject":"00000000-0000-4000-8000-000000000001","tenantId":"00000000-0000-4000-8000-000000000003"},"docs-app-assistant":{"subject":"00000000-0000-4000-8000-000000000002","tenantId":"00000000-0000-4000-8000-000000000003"}}'
# Depois de exportar as imagens e o issuer da instalação:
bash deploy/render-aws-oauth.sh > ai-oauth.yaml
```

O bootstrap precisa criar estes objetos no namespace `as-docs` antes do rollout:

- ConfigMap `as-docs-keycloak-ai-trust`, chave `ca.pem`: CA pública do Keycloak.
  A IA a acrescenta somente ao contexto TLS das consultas ao JWKS, sem substituir
  a confiança do sistema nem alterar o cliente OpenAI. O JWKS é fixo em
  `https://as-docs-keycloak.as-docs-identity.svc.cluster.local:8443/id/realms/as-docs/protocol/openid-connect/certs`.
- Secret TLS `as-docs-ai-oauth-tls`, com `tls.crt` e `tls.key`: certificado do
  Caddy com SAN `as-docs-ai.as-docs.svc.cluster.local`. Distribua a CA desse
  certificado para o backend. Não reutilize o Secret histórico `as-docs-ai-tls`.
- Secret `as-docs-ai-openai`, chave `OPENAI_API_KEY`, **somente em modo `openai`**.
  Os segredos OAuth continuam apenas no backend/Keycloak; a IA não recebe HMAC.

O backend usa `DOCS_AI_URL=https://as-docs-ai.as-docs.svc.cluster.local:8443`.
O Service `ClusterIP` serve apenas a porta `8443`; a NetworkPolicy permite
ingresso só dos pods `app: as-docs-backend` no mesmo namespace, desde que o
cluster aplique NetworkPolicy. Uvicorn escuta em `127.0.0.1:8000`. A sonda de
prontidão passa pelo Caddy em HTTPS e consulta `/health/ready` da IA. O
`default_sni` permite à sonda usar o IP do pod mesmo quando o certificado só
contém o nome DNS. A sonda não valida a cadeia do certificado nem consulta o
Keycloak. Verifique o TLS a partir
do backend com a CA configurada e confirme a indexação e o chat com tokens reais
após o rollout. Mantenha o Redis 8 e o servidor MCP do backend disponíveis.

Use o render como fonte dos recursos desse ambiente; não aplique os manifests
legados sobre os mesmos objetos. Alterações no YAML renderizado atualizam o
template dos pods automaticamente. Depois de rotacionar apenas a CA, o
certificado TLS ou a chave OpenAI, reinicie `deployment/as-docs-ai` para
recarregar o material. Num cluster que já roda o modo legado, coordene o corte
do backend e da IA: a porta muda, e não há fallback para HMAC.

Validação local sem cluster: `bash deploy/test-aws-render.sh` (usa
`kubectl kustomize`).

## Deploy legado — HMAC

Pré-requisito: namespace `as-docs` e Redis 8 do repositório **As-docs-app**
(`deploy/k8s/00-namespace.yaml` e `10-redis.yaml`).

```bash
docker build -t ghcr.io/albibino/as-docs-ai:latest .
kubectl -n as-docs create secret generic as-docs-ai-secret \
  --from-literal=OPENAI_API_KEY="<chave>" \
  --from-literal=DOCS_AI_HMAC_SECRET="<mesmo do backend>"
kubectl apply -f deploy/k8s/10-ai-config.yaml -f deploy/k8s/20-ai.yaml
```

- Endereço interno: `http://as-docs-ai.as-docs.svc.cluster.local:8000`
  (é o `DOCS_AI_URL` do backend).
- Saída para a OpenAI passa pelo NAT Gateway da subnet privada.
- Toda chamada exige assinatura HMAC (`DOCS_AI_HMAC_SECRET`, igual nos dois lados).
- Trocar modelo/Redis/MCP: edite o ConfigMap e `kubectl -n as-docs rollout restart deployment as-docs-ai`.
