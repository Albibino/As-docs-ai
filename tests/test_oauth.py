from __future__ import annotations

import asyncio
import json
import threading
import time
from contextlib import suppress
from types import SimpleNamespace

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from httpx import ASGITransport, AsyncClient
from jwt.algorithms import RSAAlgorithm

from tcc_docs_ai import oauth
from tcc_docs_ai.app import create_app
from tcc_docs_ai.config import Settings
from tests.conftest import post_signed, signed_headers
from tests.rag_fixtures import CREDENTIAL, FakeDocsApp, build_stack

ISSUER = "http://localhost:9090/realms/as-docs"
JWKS_URL = "http://keycloak:8080/realms/as-docs/protocol/openid-connect/certs"
INDEXER = "docs-app-indexer"
ASSISTANT = "docs-app-assistant"
CLIENTS = {
    INDEXER: {"subject": "indexer-service-account"},
    ASSISTANT: {"subject": "assistant-service-account"},
}
WRITING = "/v1/generations/stream"
CHAT = "/v1/chat/stream"
INDEX_WRITE = "/v1/index/pages"
INDEX_READ = "/v1/index/status"
EMBEDDINGS = "/v1/embeddings"


@pytest.fixture
def oauth_settings(monkeypatch):
    monkeypatch.setenv("AI_OAUTH_ENABLED", "true")
    monkeypatch.setenv("AI_OIDC_ISSUER", ISSUER)
    monkeypatch.setenv("AI_OIDC_JWKS_URL", JWKS_URL)
    monkeypatch.setenv("AI_OAUTH_CLIENTS_JSON", json.dumps(CLIENTS))
    monkeypatch.setenv("DOCS_AI_HMAC_SECRET", "")
    return Settings(openai_model="gpt-5.6-luna")


@pytest.fixture(scope="module")
def signing_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


def public_jwk(key, kid):
    return {
        **RSAAlgorithm.to_jwk(key.public_key(), as_dict=True),
        "kid": kid,
        "alg": "RS256",
        "use": "sig",
    }


@pytest.fixture
async def jwks(oauth_settings, signing_key):
    upstream = SimpleNamespace(
        keys=[public_jwk(signing_key, "current")], requests=[], response=None, slow=False
    )

    async def reply(reader, writer):
        try:
            request = await reader.readuntil(b"\r\n\r\n")
            path = request.split(b" ")[1].decode("ascii")
            upstream.requests.append(path)
            assert b"\nauthorization:" not in request.lower()
            status, headers, body = upstream.response or (
                200,
                {},
                json.dumps({"keys": upstream.keys}).encode(),
            )
            headers = {"Connection": "close", "Content-Length": str(len(body)), **headers}
            writer.write(f"HTTP/1.1 {status} Test\r\n".encode())
            for name, value in headers.items():
                writer.write(f"{name}: {value}\r\n".encode())
            writer.write(b"\r\n")
            if upstream.slow:
                await asyncio.sleep(0.5)
                for value in body:
                    writer.write(bytes([value]))
                    await writer.drain()
                    await asyncio.sleep(0.005)
            else:
                writer.write(body)
                await writer.drain()
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            writer.close()
            with suppress(ConnectionError):
                await writer.wait_closed()

    async with await asyncio.start_server(reply, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        oauth_settings.ai_oidc_jwks_url = f"http://127.0.0.1:{port}/keys"
        upstream.url = oauth_settings.ai_oidc_jwks_url
        yield upstream


@pytest.fixture
def token(signing_key):
    def issue(
        *, client=ASSISTANT, scope="ai:writing:invoke", key=None, kid="current", jose=None, **claims
    ):
        now = int(time.time())
        return jwt.encode(
            {
                "iss": ISSUER,
                "aud": ["docs-ai", "account"],
                "iat": now - 1,
                "exp": now + 60,
                "sub": CLIENTS.get(client, CLIENTS[ASSISTANT])["subject"],
                "azp": client,
                "scope": scope,
                **claims,
            },
            key or signing_key,
            algorithm="RS256",
            headers={"kid": kid, **(jose or {})},
        )

    return issue


def bearer(token, key="idem-oauth"):
    return {"authorization": f"Bearer {token}", "idempotency-key": key}


@pytest.fixture
async def oauth_client(oauth_settings, jwks, service):
    app = create_app(oauth_settings, service=service)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client


@pytest.fixture
def clock(monkeypatch):
    now = SimpleNamespace(value=1000.0)
    monkeypatch.setattr(oauth, "monotonic", lambda: now.value)
    return now


async def test_writing_bearer_works_without_hmac_secret(oauth_client, token, payload, jwks):
    response = await oauth_client.post(WRITING, json=payload, headers=bearer(token()))
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: done" in response.text
    assert len(jwks.requests) == 1


@pytest.mark.parametrize(
    "allowlist",
    [
        "",
        "not-json",
        json.dumps({INDEXER: CLIENTS[INDEXER]}),
        json.dumps({**CLIENTS, "other": CLIENTS[ASSISTANT]}),
        json.dumps({**CLIENTS, ASSISTANT: {"subject": ""}}),
        json.dumps({**CLIENTS, ASSISTANT: {"subject": "sub", "tenantId": "other-workspace"}}),
        json.dumps({**CLIENTS, ASSISTANT: {**CLIENTS[ASSISTANT], "scope": "ai:chat:invoke"}}),
        "{" + json.dumps(INDEXER) + ":{}," + json.dumps(CLIENTS)[1:],
    ],
)
def test_oauth_rejects_invalid_client_configuration(oauth_settings, monkeypatch, allowlist):
    monkeypatch.setenv("AI_OAUTH_CLIENTS_JSON", allowlist)
    # HMAC não pode mascarar configuração OAuth inválida.
    monkeypatch.setenv("DOCS_AI_HMAC_SECRET", "legacy-test-secret")
    with pytest.raises(ValueError, match="AI_OAUTH_CLIENTS_JSON"):
        create_app(Settings(openai_model="gpt-5.6-luna"))


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("AI_OIDC_ISSUER", ""),
        ("AI_OIDC_ISSUER", "https://user@issuer.example/realm"),
        ("AI_OIDC_ISSUER", "http://issuer.example/realm"),
        ("AI_OIDC_JWKS_URL", ""),
        ("AI_OIDC_JWKS_URL", "file:///keys.json"),
        ("AI_OIDC_JWKS_URL", "https://issuer.example/keys#fragment"),
    ],
)
def test_oauth_requires_explicit_issuer_and_jwks(oauth_settings, monkeypatch, variable, value):
    monkeypatch.setenv(variable, value)
    with pytest.raises(ValueError, match=variable):
        create_app(Settings(openai_model="gpt-5.6-luna"))


def test_https_issuer_cannot_use_http_jwks(oauth_settings, monkeypatch):
    monkeypatch.setenv("AI_OIDC_ISSUER", "https://issuer.example/realm")
    with pytest.raises(ValueError, match="AI_OIDC_JWKS_URL"):
        create_app(Settings(openai_model="gpt-5.6-luna"))


async def test_default_legacy_accepts_blank_oauth_env_but_requires_hmac(
    monkeypatch, settings, service, payload, token
):
    monkeypatch.setenv("AI_OAUTH_CLIENTS_JSON", "")
    monkeypatch.setenv("AI_OIDC_ISSUER", "")
    monkeypatch.setenv("AI_OIDC_JWKS_URL", "")
    monkeypatch.setenv("DOCS_AI_HMAC_SECRET", "")
    _, indexing, chat, _ = build_stack()
    for configured in (settings, Settings(openai_model="gpt-5.6-luna")):
        app = create_app(configured, service=service, indexing_service=indexing, chat_service=chat)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            assert (
                await client.post(WRITING, json=payload, headers=bearer(token()))
            ).status_code == 401
            has_hmac = configured is settings
            assert (await client.get("/health/ready")).status_code == (200 if has_hmac else 503)
            assert (await post_signed(client, payload)).status_code == (200 if has_hmac else 401)


async def test_oauth_readiness_does_not_require_hmac(oauth_settings, service, jwks):
    _, indexing, chat, _ = build_stack()
    app = create_app(oauth_settings, service=service, indexing_service=indexing, chat_service=chat)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        assert (await client.get("/health/ready")).status_code == 200
    assert not jwks.requests


async def test_oauth_never_falls_back_to_a_valid_hmac(
    oauth_settings, monkeypatch, service, payload
):
    monkeypatch.setenv("DOCS_AI_HMAC_SECRET", "test-secret-with-at-least-32-bytes")
    app = create_app(Settings(openai_model="gpt-5.6-luna"), service=service)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await post_signed(client, payload)
        assert response.status_code == 401
        assert response.headers["www-authenticate"] == "Bearer"
        headers = {**signed_headers(payload), "authorization": "Bearer not-a-token"}
        response = await client.post(
            WRITING, content=json.dumps(payload, separators=(",", ":")), headers=headers
        )
        assert response.status_code == 401
        assert response.headers["www-authenticate"] == 'Bearer error="invalid_token"'


@pytest.mark.parametrize(
    "claims",
    [
        {"iss": "http://keycloak:8080/realms/as-docs"},
        {"aud": "tcc-docs-mcp"},
        {"aud": ["docs-ai-other", "account"]},
        {"iat": 4_102_444_800},
        {"exp": 1},
        {"nbf": 4_102_444_800},
        {"nbf": "1"},
        {"iat": True},
        {"exp": None},
        {"azp": None},
        {"sub": ""},
    ],
)
async def test_invalid_claims_are_unauthenticated(oauth_client, token, payload, claims):
    response = await oauth_client.post(WRITING, json=payload, headers=bearer(token(**claims)))
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == 'Bearer error="invalid_token"'


@pytest.mark.parametrize(
    "claims",
    [
        {"azp": "other-client"},
        {"sub": "unexpected-subject"},
        {"scope": "ai:chat:invoke"},
        {"scope": "ai:writing:invoke:other"},
        {"scope": ["ai:writing:invoke"]},
        {"scope": "openid\tai:writing:invoke"},
    ],
)
async def test_valid_token_without_authority_is_forbidden(oauth_client, token, payload, claims):
    response = await oauth_client.post(WRITING, json=payload, headers=bearer(token(**claims)))
    assert response.status_code == 403


async def test_writing_checks_current_authority_before_replay(
    oauth_client, provider, token, payload
):
    initial = await oauth_client.post(WRITING, json=payload, headers=bearer(token()))
    denied = await oauth_client.post(WRITING, json=payload, headers=bearer(token(scope="openid")))
    replay = await oauth_client.post(
        WRITING, json=payload, headers=bearer(token(aud="docs-ai", nbf=1, jti="another-token"))
    )
    assert (initial.status_code, denied.status_code, replay.status_code) == (200, 403, 200)
    assert denied.headers["content-type"].startswith("application/json")
    assert replay.content == initial.content
    assert provider.calls == 1


async def test_writing_requires_idempotency_header(oauth_client, token, payload, provider):
    response = await oauth_client.post(
        WRITING, json=payload, headers={"authorization": f"Bearer {token()}"}
    )
    assert response.status_code == 400
    assert provider.calls == 0


async def test_routes_enforce_client_scope_and_tenant(
    oauth_settings, jwks, service, token, payload
):
    _, indexing, chat, _ = build_stack()
    app = create_app(oauth_settings, service=service, indexing_service=indexing, chat_service=chat)
    cases = [
        (INDEX_WRITE, INDEXER, "ai:index:write", {"pages": [{"pageId": "p1", "version": 2}]}),
        (INDEX_READ, INDEXER, "ai:index:read", {"pageIds": ["p1"]}),
        (CHAT, ASSISTANT, "ai:chat:invoke", {"sessionId": "s1", "question": "Qual é o prazo?"}),
        (WRITING, ASSISTANT, "ai:writing:invoke", payload),
    ]
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        for route, client_id, scope, body in cases:
            original = {**body, "tenantId": "tenant-1"}
            other_client = ASSISTANT if client_id == INDEXER else INDEXER
            allowed = bearer(token(client=client_id, scope=f"openid {scope}"))
            denied_requests = [
                (original, bearer(token(client=other_client, scope=scope))),
                (original, bearer(token(client=client_id, scope="openid"))),
            ]
            for denied_body, headers in denied_requests:
                response = await client.post(route, json=denied_body, headers=headers)
                assert response.status_code == 403, (route, response.text)
            if client_id == INDEXER:
                del allowed["idempotency-key"]
            response = await client.post(route, json=original, headers=allowed)
            assert response.status_code == 200, (route, response.text)

            other_workspace = {**original, "tenantId": "other-workspace"}
            response = await client.post(route, json=other_workspace, headers=allowed)
            assert response.status_code == 200, (route, response.text)
            if route == INDEX_WRITE:
                assert response.json()["results"][0]["status"] == "indexed"
            if route == INDEX_READ:
                assert response.json()["pages"][0]["latestVersion"] == 2
    assert len(jwks.requests) == 1


async def test_embeddings_has_no_oauth_policy(oauth_client, token):
    body = {"tenantId": "tenant-1", "texts": ["teste"]}
    response = await oauth_client.post(
        EMBEDDINGS, json=body, headers=bearer(token(scope="ai:embeddings:invoke ai:writing:invoke"))
    )
    assert response.status_code == 403


async def test_chat_bearer_does_not_replace_mcp_credential(oauth_settings, jwks, token):
    _, indexing, chat, mcp = build_stack(app=FakeDocsApp())
    app = create_app(oauth_settings, indexing_service=indexing, chat_service=chat)
    body = {
        "tenantId": "tenant-1",
        "sessionId": "s1",
        "question": "Onde está o manual?",
        "mcpCredential": CREDENTIAL,
    }
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(CHAT, json=body, headers=bearer(token(scope="ai:chat:invoke")))
    assert response.status_code == 200
    assert mcp is not None and mcp.credentials == [CREDENTIAL]
    assert CREDENTIAL not in response.text


async def test_signature_must_match_pinned_key(oauth_client, token, payload):
    other_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    response = await oauth_client.post(WRITING, json=payload, headers=bearer(token(key=other_key)))
    assert response.status_code == 401


async def test_non_rs256_is_rejected_without_loading_jwks(oauth_client, payload, jwks):
    other_algorithm = jwt.encode({}, "x" * 32, algorithm="HS256", headers={"kid": "current"})
    response = await oauth_client.post(WRITING, json=payload, headers=bearer(other_algorithm))
    assert response.status_code == 401
    assert not jwks.requests


async def test_token_header_cannot_change_pinned_jwks(oauth_client, token, payload, jwks):
    access_token = token(jose={"jku": f"{jwks.url}/decoy"})
    response = await oauth_client.post(WRITING, json=payload, headers=bearer(access_token))
    assert response.status_code == 200
    assert jwks.requests == ["/keys"]


async def test_jwks_rotation_limits_concurrent_refreshes_for_different_kids(
    oauth_client, token, payload, jwks, clock
):
    async def post(kid, **kwargs):
        return await oauth_client.post(
            WRITING, json=payload, headers=bearer(token(kid=kid, **kwargs))
        )

    assert (await post("current")).status_code == 200
    assert (await post("not-yet-known")).status_code == 401
    assert len(jwks.requests) == 1

    next_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    jwks.keys = [public_jwk(next_key, "next")]
    clock.value += 31
    denied = await asyncio.gather(*(post(f"not-listed-{n}") for n in range(12)))
    assert all(response.status_code == 401 for response in denied)
    assert len(jwks.requests) == 2
    assert (await post("next", key=next_key)).status_code == 200
    assert (await post("current")).status_code == 401
    assert len(jwks.requests) == 2


async def test_expired_jwks_is_not_reused_during_outage(oauth_client, token, payload, jwks, clock):
    async def post(kid="current"):
        return await oauth_client.post(WRITING, json=payload, headers=bearer(token(kid=kid)))

    assert (await post()).status_code == 200
    jwks.response = (503, {}, b"")
    clock.value += 31
    assert (await post("unknown")).status_code == 401
    assert (await post()).status_code == 200  # cache válido, TTL não renovado
    assert len(jwks.requests) == 2
    clock.value += 300
    assert (await post()).status_code == 401
    assert (await post()).status_code == 401
    assert len(jwks.requests) == 3
    jwks.response = None
    clock.value += 31
    assert (await post()).status_code == 200
    assert len(jwks.requests) == 4


@pytest.mark.parametrize(
    "failure", ["redirect", "large-body", "large-set", "duplicate-kid", "broken"]
)
async def test_invalid_jwks_fails_closed(oauth_client, token, payload, jwks, failure):
    if failure == "redirect":
        jwks.response = (302, {"Location": f"{jwks.url}/decoy"}, b"")
    elif failure == "large-body":
        jwks.response = (200, {}, json.dumps({"keys": jwks.keys}).encode() + b" " * 65_537)
    elif failure == "large-set":
        jwks.keys *= 33
    elif failure == "duplicate-kid":
        jwks.keys *= 2
    else:
        jwks.keys[0]["n"] = None
    response = await oauth_client.post(WRITING, json=payload, headers=bearer(token()))
    assert response.status_code == 401
    assert jwks.requests == ["/keys"]


async def test_jwks_fetch_has_a_total_deadline(oauth_client, token, payload, jwks, monkeypatch):
    monkeypatch.setattr(oauth, "_JWKS_TIMEOUT_SECONDS", 0.15)
    jwks.slow = True
    async with asyncio.timeout(1):
        response = await oauth_client.post(WRITING, json=payload, headers=bearer(token()))
    assert response.status_code == 401
    assert len(jwks.requests) == 1


async def test_timed_out_jwks_worker_is_not_replaced_while_running(
    oauth_client, token, payload, monkeypatch, clock
):
    started, release = threading.Event(), threading.Event()
    attempts = []

    def blocked_download(*_args):
        attempts.append(1)
        started.set()
        release.wait(timeout=2)
        return None

    monkeypatch.setattr(oauth, "_download_jwks", blocked_download, raising=False)
    monkeypatch.setattr(oauth, "_JWKS_TIMEOUT_SECONDS", 0.15)

    async def post():
        return await oauth_client.post(WRITING, json=payload, headers=bearer(token()))

    try:
        first = await post()
        assert started.is_set() and first.status_code == 401
        clock.value += 31
        assert (await post()).status_code == 401
        assert attempts == [1]
    finally:
        release.set()


def test_oauth_consumer_identity_is_scoped_by_workspace():
    principal = oauth.OAuthPrincipal("oauth:trusted-backend")

    first = principal.for_tenant("workspace-a")
    second = principal.for_tenant("workspace-b")

    assert first != second
    assert first == principal.for_tenant("workspace-a")
    assert first.startswith("oauth:")

    with pytest.raises(oauth.AuthorizationError):
        principal.for_tenant("")
