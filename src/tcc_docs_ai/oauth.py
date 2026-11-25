from __future__ import annotations

import asyncio
import hashlib
import json
import ssl
from dataclasses import dataclass
from http.client import HTTPException
from time import monotonic
from typing import Any
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, HTTPSHandler, ProxyHandler, build_opener
from urllib.request import Request as UrlRequest

import jwt
from fastapi import Request

from tcc_docs_ai.config import OAuthConfig
from tcc_docs_ai.errors import AuthorizationError, BearerTokenError
from tcc_docs_ai.oidc_config import ASSISTANT_CLIENT, INDEXER_CLIENT

_AUDIENCE = "docs-ai"
_ALGORITHM = "RS256"
_MAX_TOKEN_LENGTH = 16 * 1024
_MAX_KID_LENGTH = 256
_MAX_JWKS_BYTES = 64 * 1024
_MAX_JWKS_KEYS = 32
_JWKS_TIMEOUT_SECONDS = 2.0
_JWKS_TTL_SECONDS = 300
_JWKS_REFRESH_INTERVAL_SECONDS = 30

_POLICIES = {
    ("POST", "/v1/index/pages"): (INDEXER_CLIENT, "ai:index:write"),
    ("POST", "/v1/index/status"): (INDEXER_CLIENT, "ai:index:read"),
    ("POST", "/v1/chat/stream"): (ASSISTANT_CLIENT, "ai:chat:invoke"),
    ("POST", "/v1/generations/stream"): (ASSISTANT_CLIENT, "ai:writing:invoke"),
}


@dataclass(frozen=True)
class OAuthPrincipal:
    consumer: str
    tenant_id: str

    def for_tenant(self, tenant_id: str) -> str:
        if tenant_id != self.tenant_id:
            raise AuthorizationError()
        return self.consumer


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _download_jwks(
    url: str, deadline: float, tls_context: ssl.SSLContext | None = None
) -> bytes | None:
    remaining = deadline - monotonic()
    if remaining <= 0:
        return None
    opener = build_opener(ProxyHandler({}), _NoRedirect(), HTTPSHandler(context=tls_context))
    request = UrlRequest(url, headers={"Accept": "application/json", "Accept-Encoding": "identity"})
    try:
        with opener.open(request, timeout=remaining) as response:
            if response.status != 200:
                return None
            if response.headers.get("Content-Encoding", "identity").lower() != "identity":
                return None
            if int(response.headers.get("Content-Length", "0")) > _MAX_JWKS_BYTES:
                return None
            body = bytearray()
            while monotonic() < deadline:
                # read1 não espera que o servidor preencha o buffer inteiro.
                chunk = response.read1(min(8192, _MAX_JWKS_BYTES + 1 - len(body)))
                if not chunk:
                    return bytes(body) if monotonic() < deadline else None
                body.extend(chunk)
                if len(body) > _MAX_JWKS_BYTES:
                    return None
    except HTTPError as exc:
        exc.close()
    except (OSError, HTTPException, ValueError):
        pass
    return None


class JwksCache:
    def __init__(self, url: str, ca_file: str | None = None) -> None:
        self._url = url
        self._tls_context = None
        if ca_file is not None:
            # Acrescenta a CA só ao cliente JWKS, sem alterar as raízes padrão.
            self._tls_context = ssl.create_default_context()
            self._tls_context.load_verify_locations(cafile=ca_file)
        self._keys: dict[str, jwt.PyJWK] = {}
        self._expires_at = 0.0
        self._refresh_after = 0.0
        self._lock = asyncio.Lock()
        self._download_task: asyncio.Task[bytes | None] | None = None

    async def key(self, kid: str) -> jwt.PyJWK | None:
        now = monotonic()
        if now < self._expires_at and (key := self._keys.get(kid)) is not None:
            return key
        if now < self._refresh_after:
            return None
        async with self._lock:
            now = monotonic()
            if now < self._expires_at and (key := self._keys.get(kid)) is not None:
                return key
            if now < self._refresh_after:
                return None
            try:
                keys = await self._fetch()
            except (TypeError, ValueError, jwt.PyJWTError, RecursionError):
                # Uma falha não estende o TTL das chaves anteriores.
                return None
            finally:
                # Limite global por cache, inclusive para kids diferentes e falhas.
                self._refresh_after = monotonic() + _JWKS_REFRESH_INTERVAL_SECONDS
            self._keys = keys
            self._expires_at = monotonic() + _JWKS_TTL_SECONDS
            return keys.get(kid)

    async def _fetch(self) -> dict[str, jwt.PyJWK]:
        # O prazo da requisição não interrompe DNS/I/O bloqueante do urllib.
        # Enquanto a thread anterior não terminar, não começa outra.
        if self._download_task is not None and not self._download_task.done():
            raise ValueError("JWKS download still running")
        task = asyncio.create_task(
            asyncio.to_thread(
                _download_jwks, self._url, monotonic() + _JWKS_TIMEOUT_SECONDS, self._tls_context
            )
        )
        self._download_task = task
        done, _ = await asyncio.wait({task}, timeout=_JWKS_TIMEOUT_SECONDS)
        body = task.result() if done else None
        if body is None:
            raise ValueError("JWKS unavailable")
        doc = json.loads(body)
        keys = doc.get("keys") if isinstance(doc, dict) else None
        if not isinstance(keys, list) or len(keys) > _MAX_JWKS_KEYS:
            raise ValueError("Invalid JWKS key list")
        result: dict[str, jwt.PyJWK] = {}
        for item in keys:
            if not isinstance(item, dict):
                raise ValueError("Invalid JWKS key")
            if (
                item.get("kty") != "RSA"
                or item.get("use", "sig") != "sig"
                or item.get("alg", _ALGORITHM) != _ALGORITHM
            ):
                continue
            ops = item.get("key_ops", ["verify"])
            kid = item.get("kid")
            if (
                not isinstance(ops, list)
                or "verify" not in ops
                or not isinstance(kid, str)
                or not 0 < len(kid) <= _MAX_KID_LENGTH
            ):
                continue
            if kid in result or "d" in item:
                raise ValueError("Invalid RSA signing key in JWKS")
            key = jwt.PyJWK.from_dict(item, algorithm=_ALGORITHM)
            if not 2048 <= key.key.key_size <= 8192:
                raise ValueError("Invalid RSA signing key size in JWKS")
            result[kid] = key
        if not result:
            raise ValueError("No RS256 signing keys in JWKS")
        return result


class OAuthResourceServer:
    def __init__(self, config: OAuthConfig) -> None:
        self._config = config
        self._jwks = JwksCache(config.jwks_url, config.ca_file)

    async def authorize(self, request: Request) -> OAuthPrincipal:
        claims = await self._verified_claims(request)
        azp, subject = claims["azp"], claims["sub"]
        client = self._config.clients.get(azp)
        route = getattr(request.scope.get("route"), "path", "")
        policy = _POLICIES.get((request.method, route))
        scope = claims.get("scope")
        if (
            client is None
            or client.subject != subject
            or policy is None
            or policy[0] != azp
            or not isinstance(scope, str)
            or policy[1] not in scope.split(" ")
        ):
            raise AuthorizationError()
        identity = json.dumps([self._config.issuer, azp, subject], separators=(",", ":"))
        consumer = "oauth:" + hashlib.sha256(identity.encode()).hexdigest()
        return OAuthPrincipal(consumer, client.tenant_id)

    async def _verified_claims(self, request: Request) -> dict[str, Any]:
        headers = request.headers.getlist("authorization")
        if not headers:
            raise BearerTokenError(missing=True)
        scheme, _, token = headers[0].partition(" ")
        if len(headers) == 1 and scheme.lower() != "bearer":
            raise BearerTokenError(missing=True)
        token = token.lstrip(" ")
        if len(headers) != 1 or not 0 < len(token) <= _MAX_TOKEN_LENGTH:
            raise BearerTokenError()
        try:
            header = jwt.get_unverified_header(token)
            kid = header.get("kid")
            if (
                header.get("alg") != _ALGORITHM
                or not isinstance(kid, str)
                or not 0 < len(kid) <= _MAX_KID_LENGTH
            ):
                raise BearerTokenError()
            key = await self._jwks.key(kid)
            if key is None:
                raise BearerTokenError()
            claims = jwt.decode(
                token,
                key,
                algorithms=[_ALGORITHM],
                issuer=self._config.issuer,
                audience=_AUDIENCE,
                options={"require": ["iss", "aud", "exp", "iat", "sub", "azp"]},
            )
            if any(
                type(claims[name]) is not int for name in ("exp", "iat", "nbf") if name in claims
            ):
                raise BearerTokenError()
            if not all(isinstance(claims[c], str) and claims[c].strip() for c in ("sub", "azp")):
                raise BearerTokenError()
        except (jwt.PyJWTError, TypeError, ValueError, OverflowError, RecursionError):
            raise BearerTokenError() from None
        return claims
