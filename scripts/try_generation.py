#!/usr/bin/env python3
"""Cliente de teste do serviço de IA: assina a requisição e imprime o SSE cru.

Faz o que o app faria, sem precisar do app. Útil para exercitar a geração
contra a OpenAI real e ver operação por operação chegando.

Uso:
    ./scripts/try_generation.py "Deixe o texto mais formal"
    ./scripts/try_generation.py "Adicione uma tabela de riscos" --block-text "..."
    ./scripts/try_generation.py "Resuma" --repeat   # prova o replay idempotente

Lê OPENAI_API_KEY / DOCS_AI_HMAC_SECRET do .env do próprio projeto.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import sys
import time
import uuid
from pathlib import Path
from urllib import error, request

ROOT = Path(__file__).resolve().parents[1]
PATH = "/v1/generations/stream"


def load_env() -> dict[str, str]:
    env: dict[str, str] = {}
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip()
    env.update({k: v for k, v in os.environ.items() if k in {"DOCS_AI_HMAC_SECRET"}})
    return env


def sign(secret: str, *, timestamp: str, idempotency_key: str, body: bytes) -> str:
    body_hash = hashlib.sha256(body).hexdigest()
    message = f"{timestamp}\nPOST\n{PATH}\n{idempotency_key}\n{body_hash}".encode()
    digest = hmac.new(secret.encode(), message, hashlib.sha256).hexdigest()
    return f"v1={digest}"


def build_payload(args: argparse.Namespace) -> dict[str, object]:
    payload: dict[str, object] = {
        "tenantId": args.tenant,
        "sessionId": args.session,
        "request": args.request,
        "title": args.title,
        "labels": [],
        "availableLabels": ["Planejamento", "Arquitetura"],
        "blocks": [
            {
                "blockId": "block-1",
                "type": "paragraph",
                "hash": hashlib.sha256(args.block_text.encode()).hexdigest(),
                "content": {
                    "type": "paragraph",
                    "content": [{"type": "text", "text": args.block_text}],
                },
                "editable": True,
            },
            {
                "blockId": "block-2",
                "type": "table",
                "hash": hashlib.sha256(b"tabela-opaca").hexdigest(),
                "editable": False,
            },
        ],
        "selectedBlockIds": [],
    }
    if args.mode == "mcp_read":
        payload["mode"] = "mcp_read"
        payload["mcpCredential"] = args.mcp_credential or "sem-credencial"
    return payload


def call(args: argparse.Namespace, body: bytes, key: str) -> None:
    env = load_env()
    secret = env.get("DOCS_AI_HMAC_SECRET")
    if not secret:
        sys.exit("DOCS_AI_HMAC_SECRET não encontrado no .env")

    timestamp = str(int(time.time()))
    req = request.Request(
        args.url + PATH,
        data=body,
        method="POST",
        headers={
            "content-type": "application/json",
            "x-ai-consumer": "docs-app",
            "x-ai-timestamp": timestamp,
            "idempotency-key": key,
            "x-ai-signature": sign(secret, timestamp=timestamp, idempotency_key=key, body=body),
        },
    )

    try:
        with request.urlopen(req, timeout=args.timeout) as response:
            print(f"HTTP {response.status}  idempotency-key={key}\n", flush=True)
            for raw in response:
                line = raw.decode("utf-8").rstrip("\n")
                if line.startswith("data:"):
                    try:
                        parsed = json.loads(line.removeprefix("data:"))
                        print("  " + json.dumps(parsed, ensure_ascii=False, indent=2), flush=True)
                    except json.JSONDecodeError:
                        print("  " + line, flush=True)
                elif line:
                    print(line, flush=True)
                else:
                    print(flush=True)
    except error.HTTPError as exc:
        print(f"HTTP {exc.code}", flush=True)
        print(exc.read().decode("utf-8"), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", help="O que pedir ao modelo")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--tenant", default="workspace-de-teste")
    parser.add_argument("--session", default="sessao-1")
    parser.add_argument("--title", default="Plano de rollout")
    parser.add_argument(
        "--block-text",
        default="O rollout vai acontecer em etapas e cada etapa tem um responsavel.",
    )
    parser.add_argument("--mode", choices=["inline", "mcp_read"], default="inline")
    parser.add_argument("--mcp-credential", default=None)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--repeat",
        action="store_true",
        help="Envia duas vezes com a mesma chave: a segunda deve ser replay idêntico.",
    )
    args = parser.parse_args()

    body = json.dumps(build_payload(args), separators=(",", ":")).encode()
    key = str(uuid.uuid4())

    call(args, body, key)
    if args.repeat:
        print("\n=== REPLAY: mesma idempotency-key, nenhuma chamada nova ao modelo ===\n")
        call(args, body, key)


if __name__ == "__main__":
    main()
