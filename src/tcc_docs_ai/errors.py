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
