from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator


class Block(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    block_id: str = Field(alias="blockId", min_length=1)
    type: str = Field(min_length=1)
    hash: str = Field(min_length=1)
    content: Any | None = None
    editable: bool = True


class SpacePage(BaseModel):
    """A page the model may link to. Title only — never page content.

    Deliberately narrower than the caller's own index: the model needs a pageId
    to point at and a title to name it, and nothing else is worth the tokens.
    """

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    page_id: str = Field(alias="pageId", min_length=1)
    title: str = Field(min_length=1, max_length=200)


class ConversationTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class GenerationMode(StrEnum):
    """Autoridade de contexto da geração. Os modos são mutuamente exclusivos.

    `inline` é o fluxo atual: o consumidor entrega o contexto já autorizado.
    `mcp_read` delega a recuperação às tools somente leitura do app. Misturar
    os dois criaria duas autoridades para o mesmo manifesto de fontes.
    """

    INLINE = "inline"
    MCP_READ = "mcp_read"


class GenerationRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    tenant_id: str = Field(alias="tenantId", min_length=1)
    session_id: str = Field(alias="sessionId", min_length=1)
    request: str = Field(min_length=1)
    blocks: list[Block] = Field(min_length=1)
    selected_block_ids: list[str] = Field(default_factory=list, alias="selectedBlockIds")
    history: list[ConversationTurn] = Field(default_factory=list, max_length=50)
    partial_context: bool = Field(default=False, alias="partialContext")
    title: str | None = Field(default=None)
    labels: list[str] = Field(default_factory=list)
    available_labels: list[str] = Field(default_factory=list, alias="availableLabels")
    space_pages: list[SpacePage] = Field(default_factory=list, alias="spacePages", max_length=200)
    mode: GenerationMode = GenerationMode.INLINE
    mcp_credential: SecretStr | None = Field(default=None, alias="mcpCredential")

    @model_validator(mode="after")
    def validate_block_references(self) -> GenerationRequest:
        ids = [block.block_id for block in self.blocks]
        if len(ids) != len(set(ids)):
            raise ValueError("blockId values must be unique")
        unknown = set(self.selected_block_ids) - set(ids)
        if unknown:
            raise ValueError(f"selectedBlockIds contains unknown IDs: {sorted(unknown)}")
        return self

    @model_validator(mode="after")
    def validate_mode(self) -> GenerationRequest:
        if self.mode is GenerationMode.MCP_READ:
            if not self.mcp_credential:
                raise ValueError("mcp_read requires mcpCredential")
            # spacePages é contexto recuperado pelo consumidor. Aceitá-lo aqui
            # somaria uma segunda origem de fontes à recuperação do MCP.
            if self.space_pages:
                raise ValueError("mcp_read cannot be combined with inline spacePages")
        elif self.mcp_credential:
            raise ValueError("mcpCredential is only valid in mcp_read mode")
        return self


class EmbeddingRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    tenant_id: str = Field(alias="tenantId", min_length=1, max_length=200)
    texts: list[str] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def validate_texts(self) -> EmbeddingRequest:
        if any(not text.strip() or len(text.encode("utf-8")) > 32_768 for text in self.texts):
            raise ValueError("embedding texts must be non-empty and at most 32768 bytes")
        return self


class EmbeddingUsage(BaseModel):
    input_tokens: int = Field(alias="inputTokens", ge=0)


class EmbeddingResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    model: str
    dimensions: int = Field(ge=1)
    embeddings: list[list[float]]
    usage: EmbeddingUsage


# --- Índice vetorial versionado ------------------------------------------------


class IndexPageVersion(BaseModel):
    """Uma versão de página a indexar. A versão é monotônica e vem do app."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    page_id: str = Field(alias="pageId", min_length=1, max_length=200)
    version: int = Field(ge=1)
    title: str = Field(default="", max_length=500)
    space_id: str | None = Field(default=None, alias="spaceId", max_length=200)
    #: Conteúdo em Markdown. Vazio em `deleted` (lápide versionada).
    text: str = Field(default="")
    deleted: bool = False


class IndexRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    tenant_id: str = Field(alias="tenantId", min_length=1, max_length=200)
    pages: list[IndexPageVersion] = Field(min_length=1, max_length=50)


class IndexStatus(StrEnum):
    INDEXED = "indexed"
    #: A versão já existe no índice: nada é refeito (nunca reindexa).
    ALREADY_INDEXED = "already_indexed"
    #: Chegou uma versão menor que a atual: não vale gastar embedding nela.
    OLDER_THAN_LATEST = "older_than_latest"


class IndexResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    page_id: str = Field(alias="pageId")
    version: int
    status: IndexStatus
    chunks: int = 0


class IndexResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    model: str
    results: list[IndexResult]


class IndexStatusRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    tenant_id: str = Field(alias="tenantId", min_length=1, max_length=200)
    page_ids: list[str] = Field(alias="pageIds", min_length=1, max_length=500)


class PageIndexStatus(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    page_id: str = Field(alias="pageId")
    latest_version: int = Field(alias="latestVersion")
    deleted: bool
    versions: list[int]


class IndexStatusResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    model: str
    pages: list[PageIndexStatus]


# --- Conversa / tirar dúvidas --------------------------------------------------


class ChatRequest(BaseModel):
    """Pergunta sobre os documentos do tenant.

    `contextPageId` é a página aberta, quando houver: ela entra sempre como
    candidata, para que dúvidas sobre "este documento" funcionem mesmo quando o
    trecho não ranqueia no índice. `mcpCredential` é a credencial delegada,
    curta e somente leitura, usada para conferir a versão atual no app e ler a
    página pelo MCP quando o índice estiver desatualizado.
    """

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    tenant_id: str = Field(alias="tenantId", min_length=1, max_length=200)
    session_id: str = Field(alias="sessionId", min_length=1, max_length=200)
    question: str = Field(min_length=1, max_length=4_000)
    history: list[ConversationTurn] = Field(default_factory=list, max_length=20)
    context_page_id: str | None = Field(default=None, alias="contextPageId", max_length=200)
    mcp_credential: SecretStr | None = Field(default=None, alias="mcpCredential")


class OperationAction(StrEnum):
    REPLACE = "replace"
    DELETE = "delete"
    INSERT_AFTER = "insert_after"
    SET_TITLE = "set_title"
    ADD_LABEL = "add_label"
    REMOVE_LABEL = "remove_label"


class ProposalOperation(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    action: OperationAction
    block_id: str | None = Field(default=None, alias="blockId")
    base_hash: str | None = Field(default=None, alias="baseHash")
    after_block_id: str | None = Field(default=None, alias="afterBlockId")
    block_type: str | None = Field(default=None, alias="blockType")
    content: Any | None = None
    attrs: dict[str, Any] | None = None
    title: str | None = Field(default=None, min_length=1)
    label: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_action_fields(self) -> ProposalOperation:
        if self.action in {OperationAction.REPLACE, OperationAction.DELETE}:
            if not self.block_id or not self.base_hash:
                raise ValueError("replace/delete require blockId and baseHash")
        if self.action is OperationAction.INSERT_AFTER:
            if not self.after_block_id or not self.block_type or self.content is None:
                raise ValueError("insert_after requires afterBlockId, blockType and content")
        if self.action is OperationAction.REPLACE and self.content is None:
            raise ValueError("replace requires content")
        if self.content is not None and not isinstance(self.content, list):
            raise ValueError("content must be a JSON array of nodes")
        if self.action is OperationAction.SET_TITLE and not self.title:
            raise ValueError("set_title requires title")
        if (
            self.action in {OperationAction.ADD_LABEL, OperationAction.REMOVE_LABEL}
            and not self.label
        ):
            raise ValueError("add_label/remove_label require label")
        return self
