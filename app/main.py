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
from app.routers import chat, health, models

from app.observability.tracing import setup_tracing
from app.observability.logger import setup_logging

# Настраиваем structlog
setup_logging(level="INFO")

# Получаем логгер для использования во всём файле
logger = structlog.get_logger()

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_tracing()

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
    try:
        redis_client = Redis.from_url(settings.redis_url, decode_responses=True)
        await redis_client.ping()
        app.state.redis = redis_client
    except Exception as e:
        logger.warning("Redis недоступен (%s) — продолжаем без кеша", e)

    # Postgres: ленивый engine — не падаем, если БД недоступна на старте.
    app.state.async_engine = None
    app.state.session_factory = None
    try:
        engine = create_async_engine(settings.database_url, pool_pre_ping=True)
        app.state.async_engine = engine
        app.state.session_factory = async_sessionmaker(
            engine, expire_on_commit=False
        )
        # Создаём отсутствующие таблицы, если их ещё нет
        # (для dev-окружений без alembic; в production — только миграции)
        from app.chat.repositories.pg_models import Base
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    except Exception as e:
        logger.warning("Postgres engine не создан (%s) — postgres-репозиторий недоступен", e)


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


app.include_router(chat.router)
app.include_router(chat_router)
app.include_router(admin_router)
app.include_router(models.router)
app.include_router(health.router)