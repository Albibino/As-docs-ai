from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from ipaddress import ip_address
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

INDEXER_CLIENT = "docs-app-indexer"
ASSISTANT_CLIENT = "docs-app-assistant"


@dataclass(frozen=True)
class OAuthClient:
    subject: str
    tenant_id: str


@dataclass(frozen=True)
class OAuthConfig:
    issuer: str
    jwks_url: str
    clients: dict[str, OAuthClient]


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


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    environment: str = "development"
    openai_api_key: SecretStr | None = None
    openai_model: str
    # TCC: temperatura zero é requisito, não preferência. O validador recusa
    # qualquer outro valor. Modelos de raciocínio da OpenAI só aceitam
    # `temperature` com `reasoning.effort=none`, por isso os dois andam juntos.
    openai_temperature: float = Field(default=0.0, ge=0.0, le=0.0)
    # Vazio omite o parâmetro `reasoning` (modelos sem raciocínio).
    openai_reasoning_effort: str | None = "none"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_embedding_dimensions: int = Field(default=1536, ge=1, le=3072)
    # `hashing` e `extractive` existem só para desenvolvimento e testes sem chave
    # da OpenAI: são determinísticos e nunca produzem texto fora dos documentos.
    ai_embedding_provider: Literal["openai", "hashing"] = "openai"
    ai_hashing_embedding_dimensions: int = Field(default=256, ge=16, le=4096)
    ai_chat_provider: Literal["openai", "extractive"] = "openai"
    docs_ai_hmac_secret: SecretStr | None = None
    ai_consumers_json: str | None = None
    ai_oauth_enabled: bool = False
    ai_oidc_issuer: str = ""
    ai_oidc_jwks_url: str = ""
    ai_oauth_clients_json: str = ""
    ai_max_input_tokens: int = Field(default=500_000, ge=1)
    ai_max_output_tokens: int = Field(default=4_000, ge=1)
    ai_max_proposal_operations: int = Field(default=40, ge=1)
    ai_hmac_max_clock_skew_seconds: int = Field(default=300, ge=1)
    ai_connection_retries: int = Field(default=2, ge=0, le=5)
    ai_max_embedding_inputs: int = Field(default=100, ge=1, le=1000)
    redis_url: SecretStr | None = None
    ai_idempotency_in_progress_ttl_seconds: int = Field(default=900, ge=30)
    ai_idempotency_completed_ttl_seconds: int = Field(default=86_400, ge=60)
    # Flag do cliente MCP, independente da flag do servidor no app. Desligá-la
    # é o rollback: o modo inline continua funcionando sem nenhuma outra mudança.
    ai_mcp_client_enabled: bool = False
    ai_mcp_server_url: str | None = None
    # Era do protocolo fixada no deploy, nunca negociada por tentativa. O servidor
    # MCP do app usa o SDK 1.x, cuja versão mais nova ainda é a de handshake.
    # Atualizar exige compatibilidade comprovada nos dois lados e novos testes.
    ai_mcp_connect_mode: str = "legacy"
    ai_mcp_max_tool_calls: int = Field(default=6, ge=1, le=50)
    ai_mcp_call_timeout_seconds: float = Field(default=10.0, gt=0, le=120)
    ai_mcp_total_timeout_seconds: float = Field(default=45.0, gt=0, le=600)
    ai_mcp_max_result_bytes: int = Field(default=200_000, ge=1_024)
    # --- RAG sobre o índice vetorial versionado ---------------------------------
    ai_chunk_max_chars: int = Field(default=1_600, ge=200, le=8_000)
    ai_rag_top_k: int = Field(default=8, ge=1, le=50)
    # Score do Redis VSIM em [0, 1]: (cosseno + 1) / 2. Abaixo do mínimo o trecho
    # não é evidência; entre `similar` e o mínimo vira sugestão de página parecida.
    ai_rag_min_score: float = Field(default=0.66, ge=0.0, le=1.0)
    ai_rag_similar_min_score: float = Field(default=0.58, ge=0.0, le=1.0)
    ai_rag_max_similar: int = Field(default=5, ge=0, le=20)
    # Teto de páginas desatualizadas lidas via MCP numa mesma pergunta.
    ai_rag_max_fresh_reads: int = Field(default=3, ge=0, le=20)
    ai_rag_max_fresh_chars: int = Field(default=12_000, ge=500, le=200_000)
    ai_max_index_text_bytes: int = Field(default=2_000_000, ge=1_024)

    @property
    def mcp_read_enabled(self) -> bool:
        return bool(self.ai_mcp_client_enabled and self.ai_mcp_server_url)

    @field_validator("openai_model")
    @classmethod
    def model_must_be_explicit(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("OPENAI_MODEL cannot be empty")
        return value.strip()

    @field_validator("openai_reasoning_effort")
    @classmethod
    def blank_effort_is_none(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.strip() or None

    def consumer_secrets(self) -> dict[str, str]:
        consumers: dict[str, str] = {}
        if self.ai_consumers_json:
            parsed = json.loads(self.ai_consumers_json)
            if not isinstance(parsed, dict) or not all(
                isinstance(key, str) and isinstance(value, str) for key, value in parsed.items()
            ):
                raise ValueError("AI_CONSUMERS_JSON must be a JSON object of string secrets")
            consumers.update(parsed)
        if self.docs_ai_hmac_secret:
            consumers.setdefault("docs-app", self.docs_ai_hmac_secret.get_secret_value())
        return consumers

    def oauth_config(self) -> OAuthConfig | None:
        if not self.ai_oauth_enabled:
            return None
        issuer = _oidc_url("AI_OIDC_ISSUER", self.ai_oidc_issuer)
        jwks_url = _oidc_url("AI_OIDC_JWKS_URL", self.ai_oidc_jwks_url)
        issuer_url = urlsplit(issuer)
        keys_url = urlsplit(jwks_url)
        if issuer_url.scheme == "http" and issuer_url.hostname != "localhost":
            raise ValueError("AI_OIDC_ISSUER must use HTTPS outside the local lab")
        if keys_url.scheme == "http" and (
            issuer_url.scheme != "http"
            or (keys_url.hostname == "keycloak" and keys_url.port != 8080)
        ):
            raise ValueError("AI_OIDC_JWKS_URL must use HTTPS outside the local lab")
        try:
            data = json.loads(self.ai_oauth_clients_json, object_pairs_hook=_unique_object)
            if not isinstance(data, dict) or data.keys() != {INDEXER_CLIENT, ASSISTANT_CLIENT}:
                raise ValueError
            clients = {}
            for client, binding in data.items():
                if not isinstance(binding, dict) or binding.keys() != {"subject", "tenantId"}:
                    raise ValueError
                subject, tenant = binding["subject"], binding["tenantId"]
                if not all(isinstance(v, str) and v.strip() for v in (subject, tenant)):
                    raise ValueError
                clients[client] = OAuthClient(subject, tenant)
        except (ValueError, RecursionError):
            raise ValueError(
                "AI_OAUTH_CLIENTS_JSON must map docs-app-indexer and docs-app-assistant "
                "to non-empty subject and tenantId fields"
            ) from None
        return OAuthConfig(issuer, jwks_url, clients)


@lru_cache
def get_settings() -> Settings:
    return Settings()
