from __future__ import annotations

from typing import Any


class ServiceError(Exception):
    code = "service_error"
    status_code = 500

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def payload(self) -> dict[str, Any]:
        return {"error": {"code": self.code, "message": self.message, "details": self.details}}


class AuthenticationError(ServiceError):
    code = "invalid_signature"
    status_code = 401


class BearerTokenError(ServiceError):
    code = "invalid_token"
    status_code = 401

    def __init__(self, *, missing: bool = False) -> None:
        super().__init__("Bearer token required" if missing else "Bearer token is invalid")
        self.challenge = "Bearer" if missing else 'Bearer error="invalid_token"'


class AuthorizationError(ServiceError):
    code = "forbidden"
    status_code = 403

    def __init__(self) -> None:
        super().__init__("Bearer token is not authorized for this request")


class MissingIdempotencyKeyError(ServiceError):
    code = "missing_idempotency_key"
    status_code = 400

    def __init__(self) -> None:
        super().__init__("A single non-empty idempotency-key header is required")


class BudgetExceededError(ServiceError):
    code = "input_budget_exceeded"
    status_code = 422


class InvalidProposalError(ServiceError):
    code = "invalid_model_proposal"
    status_code = 502


class IdempotencyConflictError(ServiceError):
    code = "idempotency_conflict"
    status_code = 409


class UpstreamUnavailableError(ServiceError):
    code = "upstream_unavailable"
    status_code = 503


class McpUnavailableError(ServiceError):
    """Falha do MCP. Nunca autoriza fallback para contexto mais amplo."""

    code = "mcp_unavailable"
    status_code = 503


class McpBudgetExceededError(ServiceError):
    """Teto de recuperação atingido. Nenhum limite é ampliado por conteúdo lido."""

    code = "mcp_budget_exceeded"
    status_code = 422


class McpToolForbiddenError(ServiceError):
    """O modelo pediu uma tool fora da allowlist somente leitura."""

    code = "mcp_tool_forbidden"
    status_code = 502
