from __future__ import annotations

import time

from httpx import AsyncClient

from tests.conftest import post_signed, signed_headers


async def test_rejects_missing_signature(client: AsyncClient, payload: dict[str, object]) -> None:
    response = await client.post("/v1/generations/stream", json=payload)
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "invalid_signature"


async def test_rejects_tampered_body(client: AsyncClient, payload: dict[str, object]) -> None:
    headers = signed_headers(payload)
    payload["request"] = "pedido adulterado"
    response = await client.post("/v1/generations/stream", json=payload, headers=headers)
    assert response.status_code == 401


async def test_rejects_expired_timestamp(client: AsyncClient, payload: dict[str, object]) -> None:
    response = await post_signed(client, payload, timestamp=str(int(time.time()) - 301))
    assert response.status_code == 401
