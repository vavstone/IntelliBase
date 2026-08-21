import secrets
import time
import uuid
import httpx
import structlog
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openai import AsyncOpenAI
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from app.admin.routes import router as admin_router
from app.chat.routes import router as chat_router
from app.core.config import get_settings
from app.core.exceptions import (
    LLMAuthError,
    LLMContentFilterError,
    LLMError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUnsupportedCountryError
)
from app.routers import categories, chat, documents, health, models, rag

from app.observability.tracing import setup_tracing
from app.observability.logger import setup_logging

# Настраиваем structlog
setup_logging(level="INFO")

# Получаем логгер для использования во всём файле
logger = structlog.get_logger()

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_tracing(settings)

    # HTTP-клиент для локальной Ollama — БЕЗ прокси (внешний прокси
    # не достучится до localhost). Таймауты и limits — общие.
    _timeout = httpx.Timeout(settings.llm.request_timeout, connect=5.0)
    _limits = httpx.Limits(max_connections=50, max_keepalive_connections=10)

    http_ollama = httpx.AsyncClient(timeout=_timeout, limits=_limits)
    app.state.http_ollama = http_ollama

    # HTTP-клиент для внешних API — С прокси
    http_external = httpx.AsyncClient(
        proxy=settings.proxy_url,
        timeout=_timeout,
        limits=_limits,
    )
    app.state.http_external = http_external

    app.state.llm_ollama = AsyncOpenAI(
        base_url=settings.llm.ollama_base_url,
        api_key="ollama",
        http_client=http_ollama,
        timeout=settings.llm.request_timeout,
        max_retries=settings.llm.max_retries,
    )
    app.state.llm_openai = AsyncOpenAI(
        base_url=settings.llm.openai_base_url,
        api_key=settings.llm.openai_api_key.get_secret_value(),
        http_client=http_external,
        timeout=settings.llm.request_timeout,
        max_retries=settings.llm.max_retries,
    )
    app.state.llm_openrouter = AsyncOpenAI(
        base_url=settings.llm.openrouter_base_url,
        api_key=settings.llm.openrouter_api_key.get_secret_value(),
        http_client=http_external,
        timeout=settings.llm.request_timeout,
        max_retries=settings.llm.max_retries,
    )

    app.state.redis = None
    redis_client = None
    try:
        redis_client = Redis.from_url(settings.redis_url, decode_responses=True)
        await redis_client.ping()
        app.state.redis = redis_client
    except Exception as e:
        logger.warning("Redis недоступен (%s) — продолжаем без кеша", e)
        if redis_client is not None:
            try:
                await redis_client.close()
            except Exception:
                pass

    # Postgres: ленивый engine — не падаем, если БД недоступна на старте.
    app.state.async_engine = None
    app.state.session_factory = None
    try:
        engine = create_async_engine(settings.database_url, pool_pre_ping=True)
        app.state.async_engine = engine
        app.state.session_factory = async_sessionmaker(
            engine, expire_on_commit=False
        )
        # Схему БД управляем только через Alembic-миграции (entrypoint перед
        # стартом uvicorn: `alembic upgrade head`). Здесь таблиц не создаём и
        # не меняем — `create_all` не добавляет/не удаляет колонки и ведёт к
        # дрейфу схемы (колонка `sources` так и не появилась бы).
    except Exception as e:
        logger.warning("Postgres engine не создан (%s) — postgres-репозиторий недоступен", e)

    if settings.chat_repository == "json":
        logger.warning(
            "CHAT_REPOSITORY=json — admin-статистика, broadcast, алерты, "
            "rate-limit и A/B-промпты требуют postgres и будут недоступны"
        )

    # Генерация канарейки
    app.state.canary = secrets.token_hex(4)  # например, "a7f3b9e2"

    # Фоновые таски
    import asyncio as _asyncio
    from app.services.broadcaster import broadcast_worker as _broadcast_worker
    from app.services.alerter import threshold_monitor as _threshold_monitor

    _broadcast_task = _asyncio.create_task(
        _broadcast_worker(
            session_factory=app.state.session_factory,
            bot_url=settings.bot_url,
            internal_token=settings.internal_token.get_secret_value(),
        )
    )
    _monitor_task = _asyncio.create_task(
        _threshold_monitor(session_factory=app.state.session_factory)
    )

    # Qdrant VectorStore
    from app.services.vector_store import VectorStore
    qdrant_api_key = (
        settings.qdrant_api_key.get_secret_value()
        if settings.qdrant_api_key is not None
        else None
    )
    app.state.vector_store = VectorStore(
        url=settings.qdrant_url,
        api_key=qdrant_api_key,
        collection=settings.qdrant_collection,
        dim=settings.embedding_dim,
    )
    try:
        await app.state.vector_store.ensure_collection()
        logger.info(
            "Qdrant collection '%s' ready (dim=%d)",
            settings.qdrant_collection,
            settings.embedding_dim,
        )
    except Exception as e:
        logger.warning("Qdrant недоступен (%s) — продолжаем без векторного поиска", e)

    # Индексация + RAG (LlamaIndex) — опциональны: собираем один раз на старте.
    # Импорт ленивый: llama_index тяжёлый, не нужен тестам без lifespan.
    app.state.ingestion_service = None
    app.state.rag_service = None
    try:
        from app.services.ingestion import IngestionService
        from app.services.rag import RAGService

        ingestion = IngestionService(settings)
        app.state.ingestion_service = ingestion
        # Первичная индексация корпуса только если коллекция пуста — на рестартах
        # UPSERTS по сохранённому docstore всё равно пропустит неизменённое.
        if ingestion.is_collection_empty():
            await _asyncio.to_thread(ingestion.ingest_all)

        rag_service = RAGService(settings)
        await _asyncio.to_thread(rag_service.build)
        app.state.rag_service = rag_service
        logger.info("RAG-сервис готов (коллекция %s)", settings.rag_collection)
    except Exception as e:
        logger.warning(
            "RAG/индексация не инициализированы (%s) — /rag/query и /documents вернут 503",
            e,
        )

    yield

    # Останавливаем фоновые таски
    for _task in (_broadcast_task, _monitor_task):
        _task.cancel()
        try:
            await _task
        except _asyncio.CancelledError:
            pass

    try:
        await app.state.llm_ollama.close()
        await app.state.llm_openai.close()
        await app.state.llm_openrouter.close()
    except Exception:
        pass
    try:
        await app.state.http_ollama.aclose()
        await app.state.http_external.aclose()
    except Exception:
        pass
    if app.state.redis is not None:
        try:
            await app.state.redis.close()
        except Exception:
            pass
    if app.state.async_engine is not None:
        try:
            await app.state.async_engine.dispose()
        except Exception:
            pass
    if hasattr(app.state, "vector_store") and app.state.vector_store is not None:
        try:
            await app.state.vector_store.close()
        except Exception:
            pass
    if getattr(app.state, "rag_service", None) is not None:
        try:
            await app.state.rag_service.close()
        except Exception:
            pass
    if getattr(app.state, "ingestion_service", None) is not None:
        try:
            app.state.ingestion_service.close()
        except Exception:
            pass


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="FastAPI-сервис поиска корпоративной документации для LLM",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Request-ID"],
    expose_headers=["X-Request-ID", "X-LLM-Cost-USD"],
)


@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    # Получаем или генерируем request_id
    request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex[:12])
    request.state.request_id = request_id
    request.state.llm_cost = 0.0
    request.state.llm_tokens = 0
    owner_external_id = request.headers.get("X-Owner-External-Id")

    # Привязываем контекстные переменные для structlog
    structlog.contextvars.bind_contextvars(
        request_id=request_id,
        # TODO: заменить на реальное
        user_id = getattr(request.state, "user_id", None),
        path = request.url.path,
        method = request.method,
    )

    t0 = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("unhandled exception")
        raise

    duration_ms = (time.perf_counter() - t0) * 1000
    response.headers["X-Request-ID"] = request_id
    response.headers["X-LLM-Cost-USD"] = f"{request.state.llm_cost:.6f}"

    # Определяем detail_code для классификации ответа
    detail_code: str | None = None
    status = response.status_code
    if status == 403:
        detail_code = "moderation_blocked"
    elif status == 429:
        detail_code = "rate_limit"

    # Пишем метрику в request_metrics (fire-and-forget — не блокируем ответ)
    sf = getattr(request.app.state, "session_factory", None)
    if sf is not None:
        try:
            from sqlalchemy import text
            async with sf() as s:
                await s.execute(
                    text(
                        """
                        INSERT INTO request_metrics
                            (path, status_code, duration_ms, detail_code, owner_external_id, created_at)
                        VALUES (:p, :sc, :d, :dc, :o, NOW())
                        """
                    ),
                    {
                        "p": request.url.path,
                        "sc": status,
                        "d": duration_ms,
                        "dc": detail_code,
                        "o": owner_external_id,
                    },
                )
                await s.commit()
        except Exception:
            pass  # метрика не должна ронять запрос

    # Логируем завершение запроса (все привязанные поля автоматически добавятся)
    logger.info(
        "request completed",
        status=response.status_code,
        duration_ms=round(duration_ms, 2),
    )
    # Очищаем контекст ПОСЛЕ лога (фикс: раньше чистилось в finally ДО лога)
    structlog.contextvars.clear_contextvars()
    return response


_STATUS_MAP: list[tuple[type[LLMError], int, str]] = [
    (LLMContentFilterError, 400, "content_filter"),
    (LLMAuthError, 401, "llm_auth_error"),
    (LLMUnsupportedCountryError, 403, "llm_unsupported_country"),
    (LLMRateLimitError, 429, "llm_rate_limit"),
    (LLMTimeoutError, 504, "llm_timeout"),
    (LLMError, 502, "llm_error"),
]


@app.exception_handler(LLMError)
async def handle_llm_error(request: Request, exc: LLMError):
    for cls, status, code in _STATUS_MAP:
        if isinstance(exc, cls):
            return JSONResponse(
                status_code=status,
                content={"error": {"code": code, "message": str(exc)}},
                headers={"X-Request-ID": getattr(request.state, "request_id", "")},
            )
    return JSONResponse(
        status_code=502,
        content={"error": {"code": "llm_error", "message": str(exc)}},
    )


@app.exception_handler(RequestValidationError)
async def handle_validation(request: Request, exc: RequestValidationError):
    errors = [
        {"field": ".".join(str(p) for p in e["loc"][1:]), "message": e["msg"]}
        for e in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content={"error": {"code": "validation_error", "fields": errors}},
        headers={"X-Request-ID": getattr(request.state, "request_id", "")},
    )


app.include_router(categories.router)
app.include_router(chat.router)
app.include_router(chat_router)
app.include_router(admin_router)
app.include_router(models.router)
app.include_router(health.router)
app.include_router(rag.router)
app.include_router(documents.router)