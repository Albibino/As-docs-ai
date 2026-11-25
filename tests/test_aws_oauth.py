from __future__ import annotations

import json
import ssl
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from time import monotonic

import pytest

from tcc_docs_ai.config import Settings
from tcc_docs_ai.oauth import OAuthResourceServer, _download_jwks

ISSUER = "http://as-docs-public-123456.us-east-1.elb.amazonaws.com/id/realms/as-docs"
JWKS = (
    "https://as-docs-keycloak.as-docs-identity.svc.cluster.local:8443"
    "/id/realms/as-docs/protocol/openid-connect/certs"
)
CLIENTS = json.dumps(
    {
        "docs-app-indexer": {
            "subject": "00000000-0000-4000-8000-000000000001",
            "tenantId": "00000000-0000-4000-8000-000000000003",
        },
        "docs-app-assistant": {
            "subject": "00000000-0000-4000-8000-000000000002",
            "tenantId": "00000000-0000-4000-8000-000000000003",
        },
    }
)


def config(issuer: str, jwks: str, flag: bool = False, clients: str = CLIENTS):
    return Settings(
        openai_model="gpt-6-luna",
        ai_oauth_enabled=True,
        ai_oidc_issuer=issuer,
        ai_oidc_jwks_url=jwks,
        ai_oidc_aws_lab_http=flag,
        ai_oauth_clients_json=clients,
    ).oauth_config()


def test_aws_http_keeps_exact_public_issuer_and_private_jwks():
    result = config(ISSUER, JWKS, True)
    assert result is not None
    assert (result.issuer, result.jwks_url) == (ISSUER, JWKS)


@pytest.mark.parametrize(
    ("issuer", "jwks", "flag"),
    [
        (ISSUER, JWKS, False),
        (ISSUER.replace(".com/", ".com:80/"), JWKS, True),
        (ISSUER.replace("http://", "https://"), JWKS, True),
        (ISSUER.replace(".com/", ".com.evil.example/"), JWKS, True),
        (ISSUER.replace("us-east-1", "us-west-2"), JWKS, True),
        (ISSUER.replace("/id/", "/"), JWKS, True),
        (ISSUER + "/", JWKS, True),
        (ISSUER + "?x=1", JWKS, True),
        (ISSUER, JWKS.replace("https://", "http://"), True),
        (ISSUER, JWKS.replace(":8443", ""), True),
        (ISSUER, JWKS + "/", True),
        (ISSUER, "https://example.org/certs", True),
        ("http://localhost:9090/realms/as-docs", JWKS, True),
    ],
)
def test_aws_exception_rejects_all_other_pairs(issuer, jwks, flag):
    with pytest.raises(ValueError):
        config(issuer, jwks, flag)


def test_local_and_generic_https_do_not_need_aws_exception():
    assert config("http://localhost:9090/realms/as-docs", "http://keycloak:8080/certs")
    assert config("https://auth.example.org/realm", "https://auth.example.org/certs")


def test_aws_jwks_settings_require_uuid_bindings():
    invalid = CLIENTS.replace("00000000-0000-4000-8000-000000000001", "not-a-uuid")
    with pytest.raises(ValueError, match="AI_OAUTH_CLIENTS_JSON"):
        config("https://auth.example.org/realm", JWKS, clients=invalid)


def test_jwks_extra_ca_preserves_system_roots_and_verifies_hostname(tmp_path):
    ca, key = tmp_path / "ca.pem", tmp_path / "key.pem"
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-days",
            "1",
            "-subj",
            "/CN=localhost",
            "-addext",
            "subjectAltName=DNS:localhost",
            "-keyout",
            str(key),
            "-out",
            str(ca),
        ],
        check=True,
        capture_output=True,
    )

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"keys":[]}')

    server_tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_tls.load_cert_chain(ca, key)
    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        server.socket = server_tls.wrap_socket(server.socket, server_side=True)
        port = server.server_port
        worker = Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            address = f"https://localhost:{port}/certs"
            settings = Settings(
                openai_model="gpt-6-luna",
                ai_oauth_enabled=True,
                ai_oidc_issuer="https://example.org/realm",
                ai_oidc_jwks_url=address,
                ai_oidc_ca_file=str(ca),
                ai_oauth_clients_json=CLIENTS,
            )
            oauth = settings.oauth_config()
            assert oauth is not None
            context = OAuthResourceServer(oauth)._jwks._tls_context
            assert context is not None
            assert set(ssl.create_default_context().get_ca_certs(binary_form=True)) < set(
                context.get_ca_certs(binary_form=True)
            )
            assert _download_jwks(address, monotonic() + 2) is None
            assert _download_jwks(address, monotonic() + 2, context) == b'{"keys":[]}'
            wrong_host = f"https://127.0.0.1:{port}/certs"
            assert _download_jwks(wrong_host, monotonic() + 2, context) is None
        finally:
            server.shutdown()
            worker.join()
