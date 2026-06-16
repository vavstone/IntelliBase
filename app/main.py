import logging
import time
import uuid
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

logger = logging.getLogger("llm-service")
logging.basicConfig(level=logging.INFO)

settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):

    app.state.llm = AsyncOpenAI(
        api_key=settings.llm.openai_api_key.get_secret_value(),
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

    yield

    try:
        await app.state.llm.close()
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
    description="FastAPI-сервис для LLM",
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
    request.state.request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex)
    request.state.llm_cost = 0.0
    request.state.llm_tokens = 0

    t0 = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("unhandled", extra={"request_id": request.state.request_id})
        raise

    duration_ms = (time.perf_counter() - t0) * 1000
    response.headers["X-Request-ID"] = request.state.request_id
    response.headers["X-LLM-Cost-USD"] = f"{request.state.llm_cost:.6f}"
    logger.info(
        "request method=%s path=%s status=%s duration_ms=%.2f request_id=%s",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
        request.state.request_id,
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
