"""Validação compartilhada pela aplicação e pelo renderizador offline."""

from __future__ import annotations

import json
import re
from ipaddress import ip_address
from urllib.parse import urlsplit

INDEXER_CLIENT = "docs-app-indexer"
ASSISTANT_CLIENT = "docs-app-assistant"
AWS_JWKS_URL = (
    "https://as-docs-keycloak.as-docs-identity.svc.cluster.local:8443"
    "/id/realms/as-docs/protocol/openid-connect/certs"
)
_AWS_HTTP_ISSUER = re.compile(
    r"http://as-docs-public-[0-9]+\.us-east-1\.elb\.amazonaws\.com/id/realms/as-docs"
)
_UUID = re.compile(r"[a-fA-F0-9]{8}(?:-[a-fA-F0-9]{4}){3}-[a-fA-F0-9]{12}")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def _oidc_url(name: str, value: str) -> str:
    try:
        url = urlsplit(value)
        host = url.hostname
        port = url.port  # valida o intervalo e o formato da porta
        if (
            not host
            or port == 0
            or url.username is not None
            or any(c in value for c in "?#\\")
            or any(ord(c) <= 32 or ord(c) == 127 or c.isspace() for c in value)
        ):
            raise ValueError
        if url.scheme == "https":
            return value
        # HTTP só para o emissor/backchannel do laboratório local.
        if url.scheme == "http" and (
            host in {"localhost", "keycloak"} or ip_address(host).is_loopback
        ):
            return value
    except ValueError:
        pass
    raise ValueError(f"{name} must be an HTTPS URL or a local lab HTTP URL")


def validate_oidc_urls(issuer: str, jwks_url: str, *, aws_lab_http: bool) -> None:
    if aws_lab_http:
        if _AWS_HTTP_ISSUER.fullmatch(issuer) and jwks_url == AWS_JWKS_URL:
            return
        raise ValueError(
            "AI_OIDC_AWS_LAB_HTTP requires the exact ALB issuer and private HTTPS JWKS"
        )

    issuer_url = urlsplit(_oidc_url("AI_OIDC_ISSUER", issuer))
    keys_url = urlsplit(_oidc_url("AI_OIDC_JWKS_URL", jwks_url))
    if issuer_url.scheme == "http" and issuer_url.hostname != "localhost":
        raise ValueError("AI_OIDC_ISSUER must use HTTPS outside the local lab")
    if keys_url.scheme == "http" and (
        issuer_url.scheme != "http" or (keys_url.hostname == "keycloak" and keys_url.port != 8080)
    ):
        raise ValueError("AI_OIDC_JWKS_URL must use HTTPS outside the local lab")


def parse_oauth_clients(raw: str, *, uuid_only: bool = False) -> dict[str, dict[str, str]]:
    try:
        data = json.loads(raw, object_pairs_hook=_unique_object)
        if not isinstance(data, dict) or data.keys() != {INDEXER_CLIENT, ASSISTANT_CLIENT}:
            raise ValueError
        for binding in data.values():
            if not isinstance(binding, dict) or binding.keys() != {"subject", "tenantId"}:
                raise ValueError
            if not all(
                isinstance(v, str) and (_UUID.fullmatch(v) if uuid_only else v.strip())
                for v in binding.values()
            ):
                raise ValueError
    except (ValueError, RecursionError):
        field_type = "UUID" if uuid_only else "non-empty"
        raise ValueError(
            "AI_OAUTH_CLIENTS_JSON must map docs-app-indexer and docs-app-assistant "
            f"to {field_type} subject and tenantId fields"
        ) from None
    return data
