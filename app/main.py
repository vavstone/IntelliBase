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

from app.deps.providers import get_settings
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

    # Создаём HTTP-клиент с прокси (если задан)
    http_client = httpx.AsyncClient(
        proxy=settings.proxy_url,  # может быть None
        timeout=httpx.Timeout(settings.llm.request_timeout, connect=5.0),
    )
    app.state.http_client = http_client

    app.state.llm = AsyncOpenAI(
        api_key=settings.llm.openai_api_key.get_secret_value(),
        http_client=http_client,
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

    # Генерация канарейки
    app.state.canary = secrets.token_hex(4)  # например, "a7f3b9e2"

    yield

    try:
        await app.state.llm.close()
        await app.state.http_client.aclose()
    except Exception:
        pass
    if app.state.redis is not None:
        try:
            await app.state.redis.close()
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
    finally:
        # Очищаем контекст после запроса
        structlog.contextvars.clear_contextvars()

    duration_ms = (time.perf_counter() - t0) * 1000
    response.headers["X-Request-ID"] = request_id
    response.headers["X-LLM-Cost-USD"] = f"{request.state.llm_cost:.6f}"

    # Логируем завершение запроса (все привязанные поля автоматически добавятся)
    logger.info(
        "request completed",
        status=response.status_code,
        duration_ms=round(duration_ms, 2),
    )
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
app.include_router(models.router)
app.include_router(health.router)