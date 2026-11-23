from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from redis.asyncio import Redis
from redis.exceptions import RedisError

from tcc_docs_ai.auth import authenticate_request
from tcc_docs_ai.chat import ChatService, ChatTuning
from tcc_docs_ai.config import Settings, get_settings
from tcc_docs_ai.errors import (
    BearerTokenError,
    MissingIdempotencyKeyError,
    ServiceError,
    UpstreamUnavailableError,
)
from tcc_docs_ai.idempotency import RedisIdempotencyStore
from tcc_docs_ai.indexing import IndexingService
from tcc_docs_ai.mcp import McpBudget, StreamableHttpMcpClient
from tcc_docs_ai.oauth import OAuthPrincipal, OAuthResourceServer
from tcc_docs_ai.provider import (
    EmbeddingProvider,
    ExtractiveAnswerProvider,
    HashingEmbeddingProvider,
    ModelProvider,
    OpenAIEmbeddingProvider,
    OpenAIResponsesProvider,
)
from tcc_docs_ai.retrieval import McpRetrievalOrchestrator
from tcc_docs_ai.schemas import (
    ChatRequest,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingUsage,
    GenerationRequest,
    IndexRequest,
    IndexResponse,
    IndexStatusRequest,
    IndexStatusResponse,
    PageIndexStatus,
)
from tcc_docs_ai.service import EmbeddingService, GenerationService
from tcc_docs_ai.tokens import TokenCounter
from tcc_docs_ai.vectorstore import RedisVectorIndex

SSE_HEADERS = {"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"}


def consumer_for_tenant(consumer: str | OAuthPrincipal, tenant_id: str) -> str:
    return consumer.for_tenant(tenant_id) if isinstance(consumer, OAuthPrincipal) else consumer


def idempotency_key(request: Request, consumer: str | OAuthPrincipal) -> str:
    if isinstance(consumer, OAuthPrincipal):
        values = request.headers.getlist("idempotency-key")
        if len(values) != 1 or not values[0].strip():
            raise MissingIdempotencyKeyError()
    return request.headers["idempotency-key"]


def build_embedder(settings: Settings) -> EmbeddingProvider | None:
    if settings.ai_embedding_provider == "hashing":
        return HashingEmbeddingProvider(dimensions=settings.ai_hashing_embedding_dimensions)
    if settings.openai_api_key:
        return OpenAIEmbeddingProvider(
            api_key=settings.openai_api_key.get_secret_value(),
            model=settings.openai_embedding_model,
            dimensions=settings.openai_embedding_dimensions,
            connection_retries=settings.ai_connection_retries,
        )
    return None


def build_openai_provider(settings: Settings) -> OpenAIResponsesProvider | None:
    if not settings.openai_api_key:
        return None
    return OpenAIResponsesProvider(
        api_key=settings.openai_api_key.get_secret_value(),
        model=settings.openai_model,
        max_output_tokens=settings.ai_max_output_tokens,
        connection_retries=settings.ai_connection_retries,
        temperature=settings.openai_temperature,
        reasoning_effort=settings.openai_reasoning_effort,
    )


def build_mcp_client(settings: Settings) -> StreamableHttpMcpClient | None:
    if not settings.mcp_read_enabled:
        return None
    return StreamableHttpMcpClient(
        url=str(settings.ai_mcp_server_url),
        connect_mode=settings.ai_mcp_connect_mode,
        budget=McpBudget(
            max_tool_calls=settings.ai_mcp_max_tool_calls,
            call_timeout_seconds=settings.ai_mcp_call_timeout_seconds,
            total_timeout_seconds=settings.ai_mcp_total_timeout_seconds,
            max_result_bytes=settings.ai_mcp_max_result_bytes,
        ),
    )


def chat_tuning(settings: Settings) -> ChatTuning:
    return ChatTuning(
        top_k=settings.ai_rag_top_k,
        min_score=settings.ai_rag_min_score,
        similar_min_score=settings.ai_rag_similar_min_score,
        max_similar=settings.ai_rag_max_similar,
        max_fresh_reads=settings.ai_rag_max_fresh_reads,
        max_fresh_chars=settings.ai_rag_max_fresh_chars,
        chunk_max_chars=settings.ai_chunk_max_chars,
        max_input_tokens=settings.ai_max_input_tokens,
    )


def create_app(
    settings: Settings | None = None,
    service: GenerationService | None = None,
    embedding_service: EmbeddingService | None = None,
    indexing_service: IndexingService | None = None,
    chat_service: ChatService | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    oauth_config = settings.oauth_config()
    oauth = OAuthResourceServer(oauth_config) if oauth_config is not None else None
    redis_client: Redis | None = None
    if settings.redis_url and (service is None or indexing_service is None or chat_service is None):
        redis_client = Redis.from_url(settings.redis_url.get_secret_value(), decode_responses=False)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        if redis_client is not None:
            await redis_client.aclose()

    app = FastAPI(title="TCC Docs AI", version="0.2.0", lifespan=lifespan)

    openai_provider = build_openai_provider(settings)
    embedder = build_embedder(settings)
    mcp_client = build_mcp_client(settings)
    token_counter = TokenCounter(settings.openai_model)

    def idempotency(feature: str) -> RedisIdempotencyStore:
        assert redis_client is not None
        return RedisIdempotencyStore(
            redis_client,
            key_prefix=f"tcc-docs-ai:{settings.environment}:{feature}-idempotency",
            in_progress_ttl_seconds=settings.ai_idempotency_in_progress_ttl_seconds,
            completed_ttl_seconds=settings.ai_idempotency_completed_ttl_seconds,
        )

    # Escrita assistida: precisa de LLM de verdade (propõe operações de bloco).
    if service is None and openai_provider and redis_client is not None:
        service = GenerationService(
            provider=openai_provider,
            token_counter=token_counter,
            idempotency=idempotency("generation"),
            max_input_tokens=settings.ai_max_input_tokens,
            max_operations=settings.ai_max_proposal_operations,
            mcp_client=mcp_client,
            retrieval=McpRetrievalOrchestrator(openai_provider) if mcp_client else None,
        )
    if embedding_service is None and embedder is not None:
        embedding_service = EmbeddingService(
            provider=embedder, max_inputs=settings.ai_max_embedding_inputs
        )

    vector_index = None
    if embedder is not None and redis_client is not None:
        vector_index = RedisVectorIndex(
            redis_client,
            key_prefix=f"tcc-docs-ai:{settings.environment}:vindex",
            model=str(getattr(embedder, "model", "embedding")),
        )
    if indexing_service is None and vector_index is not None and embedder is not None:
        indexing_service = IndexingService(
            index=vector_index,
            embedder=embedder,
            model=str(getattr(embedder, "model", "embedding")),
            chunk_max_chars=settings.ai_chunk_max_chars,
            max_embedding_inputs=settings.ai_max_embedding_inputs,
            max_text_bytes=settings.ai_max_index_text_bytes,
        )

    chat_provider: ModelProvider | None = (
        ExtractiveAnswerProvider() if settings.ai_chat_provider == "extractive" else openai_provider
    )
    if (
        chat_service is None
        and vector_index is not None
        and embedder is not None
        and chat_provider is not None
    ):
        chat_service = ChatService(
            index=vector_index,
            embedder=embedder,
            provider=chat_provider,
            idempotency=idempotency("chat"),
            token_counter=token_counter,
            mcp_client=mcp_client,
            tuning=chat_tuning(settings),
        )

    app.state.generation_service = service
    app.state.embedding_service = embedding_service
    app.state.indexing_service = indexing_service
    app.state.chat_service = chat_service
    app.state.settings = settings
    app.state.redis_client = redis_client

    @app.exception_handler(ServiceError)
    async def service_error_handler(_request: Request, exc: ServiceError) -> JSONResponse:
        headers = {"WWW-Authenticate": exc.challenge} if isinstance(exc, BearerTokenError) else None
        return JSONResponse(status_code=exc.status_code, content=exc.payload(), headers=headers)

    @app.get("/health/live")
    async def live() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health/ready")
    async def ready() -> JSONResponse:
        # JWKS é carregado sob demanda. O probe não consulta o provedor de identidade.
        auth_configured = oauth is not None or any(settings.consumer_secrets().values())
        configured = bool(auth_configured and app.state.chat_service and app.state.indexing_service)
        if configured and app.state.redis_client is not None:
            try:
                await app.state.redis_client.ping()
            except RedisError:
                configured = False
        return JSONResponse(
            status_code=200 if configured else 503,
            content={"status": "ready" if configured else "not_ready"},
        )

    async def request_consumer(request: Request) -> str | OAuthPrincipal:
        if oauth is not None:
            return await oauth.authorize(request)
        return await authenticate_request(
            request,
            consumer_secrets=settings.consumer_secrets(),
            max_clock_skew_seconds=settings.ai_hmac_max_clock_skew_seconds,
        )

    requires_consumer = Depends(request_consumer)

    @app.post("/v1/generations/stream")
    async def generate(
        payload: GenerationRequest,
        request: Request,
        consumer: str | OAuthPrincipal = requires_consumer,
    ) -> StreamingResponse:
        identity = consumer_for_tenant(consumer, payload.tenant_id)
        key = idempotency_key(request, consumer)
        generation_service = app.state.generation_service
        if generation_service is None:
            raise UpstreamUnavailableError(
                "Generation service requires OPENAI_API_KEY and REDIS_URL"
            )
        raw_body = await request.body()
        prepared = await generation_service.prepare(
            payload,
            consumer=identity,
            idempotency_key=key,
            raw_body=raw_body,
        )
        return StreamingResponse(
            generation_service.stream(prepared),
            media_type="text/event-stream",
            headers=SSE_HEADERS,
        )

    @app.post("/v1/chat/stream")
    async def chat(
        payload: ChatRequest,
        request: Request,
        consumer: str | OAuthPrincipal = requires_consumer,
    ) -> StreamingResponse:
        identity = consumer_for_tenant(consumer, payload.tenant_id)
        key = idempotency_key(request, consumer)
        active = app.state.chat_service
        if active is None:
            raise UpstreamUnavailableError(
                "Chat requires REDIS_URL, an embedding provider and a chat provider"
            )
        prepared = await active.prepare(
            payload,
            consumer=identity,
            idempotency_key=key,
            raw_body=await request.body(),
        )
        return StreamingResponse(
            active.stream(prepared), media_type="text/event-stream", headers=SSE_HEADERS
        )

    @app.post("/v1/index/pages", response_model=IndexResponse)
    async def index_pages(
        payload: IndexRequest,
        consumer: str | OAuthPrincipal = requires_consumer,
    ) -> IndexResponse:
        consumer_for_tenant(consumer, payload.tenant_id)
        active = app.state.indexing_service
        if active is None:
            raise UpstreamUnavailableError("Indexing requires REDIS_URL and an embedding provider")
        results = await active.index(payload.tenant_id, payload.pages)
        return IndexResponse(model=active.model, results=results)

    @app.post("/v1/index/status", response_model=IndexStatusResponse)
    async def index_status(
        payload: IndexStatusRequest,
        consumer: str | OAuthPrincipal = requires_consumer,
    ) -> IndexStatusResponse:
        consumer_for_tenant(consumer, payload.tenant_id)
        active = app.state.indexing_service
        if active is None:
            raise UpstreamUnavailableError("Indexing requires REDIS_URL and an embedding provider")
        states = await active.states(payload.tenant_id, payload.page_ids)
        return IndexStatusResponse(
            model=active.model,
            pages=[
                PageIndexStatus(
                    pageId=state.page_id,
                    latestVersion=state.latest_version,
                    deleted=state.deleted,
                    versions=list(state.versions),
                )
                for state in states.values()
            ],
        )

    @app.post("/v1/embeddings", response_model=EmbeddingResponse)
    async def embeddings(
        payload: EmbeddingRequest,
        _consumer: str | OAuthPrincipal = requires_consumer,
    ) -> EmbeddingResponse:
        active_service = app.state.embedding_service
        if active_service is None:
            raise UpstreamUnavailableError("No embedding provider is configured")
        result = await active_service.embed(payload)
        return EmbeddingResponse(
            model=result.model,
            dimensions=result.dimensions,
            embeddings=result.embeddings,
            usage=EmbeddingUsage(inputTokens=result.input_tokens),
        )

    return app
