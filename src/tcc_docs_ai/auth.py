from __future__ import annotations

import hashlib
import hmac
import time
from collections.abc import Mapping

from fastapi import Request

from tcc_docs_ai.errors import AuthenticationError


def canonical_signature_input(
    *, timestamp: str, method: str, path: str, idempotency_key: str, body: bytes
) -> bytes:
    body_hash = hashlib.sha256(body).hexdigest()
    return f"{timestamp}\n{method.upper()}\n{path}\n{idempotency_key}\n{body_hash}".encode()


def sign_request(
    secret: str, *, timestamp: str, method: str, path: str, idempotency_key: str, body: bytes
) -> str:
    digest = hmac.new(
        secret.encode(),
        canonical_signature_input(
            timestamp=timestamp,
            method=method,
            path=path,
            idempotency_key=idempotency_key,
            body=body,
        ),
        hashlib.sha256,
    ).hexdigest()
    return f"v1={digest}"


async def authenticate_request(
    request: Request,
    *,
    consumer_secrets: Mapping[str, str],
    max_clock_skew_seconds: int,
) -> str:
    consumer = request.headers.get("x-ai-consumer", "")
    timestamp = request.headers.get("x-ai-timestamp", "")
    idempotency_key = request.headers.get("idempotency-key", "")
    supplied = request.headers.get("x-ai-signature", "")
    if not consumer or not timestamp or not idempotency_key or not supplied:
        raise AuthenticationError("Missing signed request headers")
    secret = consumer_secrets.get(consumer)
    if not secret:
        raise AuthenticationError("Unknown consumer")
    try:
        request_time = int(timestamp)
    except ValueError as exc:
        raise AuthenticationError("Invalid timestamp") from exc
    if abs(int(time.time()) - request_time) > max_clock_skew_seconds:
        raise AuthenticationError("Expired request timestamp")
    body = await request.body()
    expected = sign_request(
        secret,
        timestamp=timestamp,
        method=request.method,
        path=request.url.path,
        idempotency_key=idempotency_key,
        body=body,
    )
    if not hmac.compare_digest(expected, supplied):
        raise AuthenticationError("Signature verification failed")
    return consumer
