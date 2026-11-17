# Deploy do As-docs-ai no EKS (subnet privada)

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
