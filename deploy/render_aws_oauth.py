"""Renderiza o deploy AWS sem cluster, credenciais ou dependências Python externas."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from string import Template

from tcc_docs_ai.oidc_config import AWS_JWKS_URL, parse_oauth_clients, validate_oidc_urls

_IMAGE = re.compile(
    r"(?:[a-z0-9]+(?:[.-][a-z0-9]+)*(?::[0-9]+)?/)?"
    r"(?:[a-z0-9]+(?:(?:[._]|__|-+)[a-z0-9]+)*/)*"
    r"[a-z0-9]+(?:(?:[._]|__|-+)[a-z0-9]+)*@sha256:[0-9a-f]{64}"
)
_OPENAI_ENV = """
          env:
            - name: OPENAI_API_KEY
              valueFrom:
                secretKeyRef:
                  name: as-docs-ai-openai
                  key: OPENAI_API_KEY"""


def image(name: str) -> str:
    value = os.environ.get(name, "")
    if not _IMAGE.fullmatch(value):
        raise ValueError(f"{name} deve usar o formato repositorio@sha256:<digest> (sem tag)")
    return value


def render() -> str:
    ai_image, caddy_image = image("AI_IMAGE"), image("CADDY_IMAGE")
    mode = os.environ.get("AI_PROVIDER_MODE", "openai")
    if mode not in {"openai", "diagnostic"}:
        raise ValueError("AI_PROVIDER_MODE deve ser openai ou diagnostic")
    flag = os.environ.get("OIDC_AWS_LAB_HTTP", "false")
    if flag not in {"true", "false"}:
        raise ValueError("OIDC_AWS_LAB_HTTP deve ser true ou false")
    issuer = os.environ.get("OIDC_ISSUER", "")
    if flag == "false" and not issuer.startswith("https://"):
        raise ValueError("OIDC_ISSUER deve usar HTTPS quando OIDC_AWS_LAB_HTTP=false")
    validate_oidc_urls(issuer, AWS_JWKS_URL, aws_lab_http=flag == "true")
    clients = parse_oauth_clients(os.environ.get("AI_OAUTH_CLIENTS_JSON", ""), uuid_only=True)

    diagnostic = mode == "diagnostic"
    values = {
        "ai_image": json.dumps(ai_image),
        "caddy_image": json.dumps(caddy_image),
        "environment": "aws-lab-diagnostic" if diagnostic else "production",
        "embedding_provider": "hashing" if diagnostic else "openai",
        "chat_provider": "extractive" if diagnostic else "openai",
        "openai_env": "" if diagnostic else _OPENAI_ENV,
        "issuer": json.dumps(issuer),
        "lab_http": json.dumps(flag),
        "jwks": json.dumps(AWS_JWKS_URL),
        "clients": json.dumps(json.dumps(clients, separators=(",", ":"), sort_keys=True)),
    }
    template = Template(Path(__file__).with_name("aws-oauth.yaml.template").read_text())
    manifest = template.substitute(values, revision="")
    return template.substitute(values, revision=hashlib.sha256(manifest.encode()).hexdigest())


if __name__ == "__main__":
    try:
        print(render(), end="")
    except ValueError as error:
        raise SystemExit(str(error)) from None
