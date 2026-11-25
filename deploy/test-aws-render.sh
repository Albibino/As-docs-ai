#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export KUBECONFIG=/dev/null
export AI_IMAGE=registry.example/as-docs-ai@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
export CADDY_IMAGE=registry.example/caddy@sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
export OIDC_ISSUER=https://docs.example.org/id/realms/as-docs OIDC_AWS_LAB_HTTP=false
export AI_OAUTH_CLIENTS_JSON='{
  "docs-app-indexer": {"subject":"00000000-0000-4000-8000-000000000001","tenantId":"00000000-0000-4000-8000-000000000003"},
  "docs-app-assistant": {"subject":"00000000-0000-4000-8000-000000000002","tenantId":"00000000-0000-4000-8000-000000000003"}
}'
unset AI_PROVIDER_MODE
scratch=$(mktemp -d)
trap 'rm -rf -- "$scratch"' EXIT
printf 'resources: [ai.yaml]\n' > "$scratch/kustomization.yaml"

bash render-aws-oauth.sh > "$scratch/ai.yaml"
kubectl kustomize "$scratch" > "$scratch/https.yaml"
export OIDC_ISSUER=http://as-docs-public-123456.us-east-1.elb.amazonaws.com/id/realms/as-docs
export OIDC_AWS_LAB_HTTP=true AI_PROVIDER_MODE=diagnostic
bash render-aws-oauth.sh > "$scratch/ai.yaml"
kubectl kustomize "$scratch" > "$scratch/http.yaml"
bash render-aws-oauth.sh | cmp -s - "$scratch/ai.yaml"
https_revision=$(grep -F 'as-docs.ai/render-sha256:' "$scratch/https.yaml")
http_revision=$(grep -F 'as-docs.ai/render-sha256:' "$scratch/http.yaml")
test "$https_revision" != "$http_revision"

grep -Fq "AI_OIDC_AWS_LAB_HTTP: \"false\"" "$scratch/https.yaml"
grep -Fq "AI_OIDC_AWS_LAB_HTTP: \"true\"" "$scratch/http.yaml"
grep -Fq "$OIDC_ISSUER" "$scratch/http.yaml"
grep -Fq 'AI_CHAT_PROVIDER: openai' "$scratch/https.yaml"
grep -Fq 'AI_EMBEDDING_PROVIDER: openai' "$scratch/https.yaml"
grep -Fq 'name: as-docs-ai-openai' "$scratch/https.yaml"
grep -Fq 'AI_CHAT_PROVIDER: extractive' "$scratch/http.yaml"
grep -Fq 'AI_EMBEDDING_PROVIDER: hashing' "$scratch/http.yaml"
grep -Fq 'ENVIRONMENT: aws-lab-diagnostic' "$scratch/http.yaml"
! grep -Eq 'as-docs-ai-openai|OPENAI_API_KEY|DOCS_AI_HMAC_SECRET|OIDC_CLIENT_SECRET' "$scratch/http.yaml" || exit 1
for manifest in "$scratch/https.yaml" "$scratch/http.yaml"; do
  grep -Fq 'type: ClusterIP' "$manifest"
  grep -Fq 'app: as-docs-backend' "$manifest"
  grep -Fq 'port: 8443' "$manifest"
  grep -Fq 'targetPort: https-ai' "$manifest"
  ! grep -Eq 'port: 8000|nodePort:|kind: (Ingress|Secret)' "$manifest" || exit 1
  grep -Fq 'path: /health/ready' "$manifest"
  grep -Fq 'scheme: HTTPS' "$manifest"
  grep -Fq 'default_sni as-docs-ai.as-docs.svc.cluster.local' "$manifest"
  grep -Fq 'reverse_proxy 127.0.0.1:8000' "$manifest"
  grep -Fq 'cp /usr/bin/caddy /config/caddy && exec /config/caddy run' "$manifest"
  grep -Fq -- '- 127.0.0.1' "$manifest"
  grep -Fq 'AI_OIDC_JWKS_URL: https://as-docs-keycloak.as-docs-identity.svc.cluster.local:8443/id/realms/as-docs/protocol/openid-connect/certs' "$manifest"
  grep -Fq 'name: as-docs-keycloak-ai-trust' "$manifest"
  grep -Fq 'key: ca.pem' "$manifest"
  grep -Fq 'secretName: as-docs-ai-oauth-tls' "$manifest"
  ! grep -Fq 'secretName: as-docs-ai-tls' "$manifest" || exit 1
  grep -Fq 'AI_MCP_CLIENT_ENABLED: "true"' "$manifest"
  grep -Fq "image: $AI_IMAGE" "$manifest"
  grep -Fq "image: $CADDY_IMAGE" "$manifest"
done

reject() {
  if env "$@" bash render-aws-oauth.sh > "$scratch/invalid.yaml" 2> "$scratch/error"; then
    printf 'Renderização inválida foi aceita.\n' >&2
    exit 1
  fi
  test ! -s "$scratch/invalid.yaml"
  ! grep -Fq 'nao-exiba-este-valor' "$scratch/error" || exit 1
}
reject AI_IMAGE=registry.example/as-docs-ai:latest
reject CADDY_IMAGE=registry.example/caddy:latest
reject OIDC_AWS_LAB_HTTP=yes
reject OIDC_AWS_LAB_HTTP=false
reject OIDC_ISSUER=https://docs.example.org/id/realms/as-docs
reject OIDC_ISSUER=http://localhost:9090/realms/as-docs
reject OIDC_ISSUER=http://as-docs-public-123456.us-east-1.elb.amazonaws.com:80/id/realms/as-docs
reject AI_PROVIDER_MODE=other
reject AI_OAUTH_CLIENTS_JSON=
reject AI_OAUTH_CLIENTS_JSON='{"clientSecret":"nao-exiba-este-valor"}'
reject "AI_OAUTH_CLIENTS_JSON=${AI_OAUTH_CLIENTS_JSON/00000000-0000-4000-8000-000000000001/nao-exiba-este-valor}"
reject "AI_OAUTH_CLIENTS_JSON=${AI_OAUTH_CLIENTS_JSON/\"subject\":/\"subject\":\"nao-exiba-este-valor\",\"subject\":}"
printf 'Renderização AWS OAuth validada com kubectl kustomize.\n'
