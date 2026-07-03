import hashlib
import json
import logging
from typing import Literal

import structlog
from time import perf_counter
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.core.exceptions import (
    LLMAuthError,
    LLMContentFilterError,
    LLMError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUnsupportedCountryError
)
from app.schemas.chat import ChatDelta, ChatRequest, ChatResponse, Usage
from app.observability.pii import prompt_hash, redact_pii

try:
    from openai import (
        APIConnectionError,
        APITimeoutError,
        AuthenticationError,
        BadRequestError,
        RateLimitError,
        PermissionDeniedError
    )
except ImportError:
    APIConnectionError = APITimeoutError = AuthenticationError = BadRequestError = RateLimitError = PermissionDeniedError = ()  # type: ignore

logger = logging.getLogger("llm")

class LLMService:
    def __init__(self, llm_ollama, llm_openai, llm_openrouter, cache, ttl: int = 3600):
        self._stream_prompt = None
        self._stream_start_time = None
        self.llm_ollama = llm_ollama
        self.llm_openai = llm_openai
        self.llm_openrouter = llm_openrouter
        self.cache = cache
        self.ttl = ttl

    def get_llm(self, provider: str):
        if provider == "openai":
            return self.llm_openai
        elif provider == "openrouter":
            return  self.llm_openrouter
        return  self.llm_ollama

    def _key(self, req: ChatRequest) -> str:
        payload = req.model_dump(exclude={"user_id", "session_id", "stream"})
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return "chat:" + hashlib.sha256(blob.encode()).hexdigest()

    def _extract_prompt(self, req: ChatRequest) -> str:
        """Извлекает пользовательский промт из запроса."""
        # Берём последнее сообщение с ролью user
        user_messages = [m for m in req.messages if m.role == "user"]
        return user_messages[-1].content if user_messages else ""

    def _log_llm_completion(
        self,
        raw_prompt: str,
        response_content: str | None,
        model: str,
        provider: Literal["openai", "ollama", "openrouter"],
        usage: Usage | None,
        finish_reason: str | None,
        latency_ms: float
    ) -> None:
        """Логирование завершения вызова LLM с маскировкой PII в промпте и ответе."""
        logger = structlog.get_logger()
        log_data = {
            "model": model,
            "provider": provider,
            "input_tokens": usage.prompt_tokens if usage else 0,
            "output_tokens": usage.completion_tokens if usage else 0,
            "latency_ms": round(latency_ms, 2),
            "finish_reason": finish_reason or "stop",
            "prompt_hash": prompt_hash(raw_prompt) if raw_prompt else None,
            "prompt_preview": redact_pii(raw_prompt)[:120] if raw_prompt else None,
        }
        if response_content is not None:
            log_data["response_preview"] = redact_pii(response_content)[:120] if response_content else None
        logger.info("llm_request_completed", **log_data)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=10),
        retry=retry_if_exception_type((APIConnectionError, APITimeoutError, RateLimitError)),
        reraise=True
    )
    async def _call(self, req: ChatRequest) -> ChatResponse:
        try:
            llm = self.get_llm(req.provider)
            raw = await llm.chat.completions.create(
                model=req.model,
                messages=[m.model_dump() for m in req.messages],
                temperature=req.temperature,
                max_tokens=req.max_tokens,
            )
            response = ChatResponse.from_openai(raw)
            return response
        except RateLimitError as e:
            logger.warning("OpenAI rate limit: %s", e)
            raise LLMRateLimitError(str(e)) from e
        except AuthenticationError as e:
            logger.error("OpenAI authentication failed: %s", e)
            raise LLMAuthError(str(e)) from e
        except APITimeoutError as e:
            logger.error("OpenAI timeout: %s", e)
            raise LLMTimeoutError(str(e)) from e
        except APIConnectionError as e:
            logger.error("OpenAI connection error: %s", e)
            raise LLMError(f"connection error: {e}") from e
        except BadRequestError as e:
            msg = str(e).lower()
            if "content" in msg and ("filter" in msg or "policy" in msg):
                raise LLMContentFilterError(str(e)) from e
            raise LLMError(str(e)) from e
        except PermissionDeniedError as e:
            logger.error("permission denied error: %s", e)
            raise LLMUnsupportedCountryError(str(e)) from e

    async def complete(self, req: ChatRequest) -> ChatResponse:
        # Кешируем только детерминированные ответы и при наличии кеша.
        if req.temperature > 0 or self.cache is None:
            try:
                resp = await self._call_with_logging(req)
            except Exception as e:
                logger.error("LLM call failed after retries: %s", e)
                raise
            return resp

        key = self._key(req)
        blob = await self.cache.get(key)
        if blob:
            resp = ChatResponse.model_validate_json(blob)
            resp.cached = True
            return resp

        try:
            resp = await self._call_with_logging(req)
        except Exception as e:
            logger.error("LLM call failed after retries: %s", e)
            raise
        await self.cache.setex(key, self.ttl, resp.model_dump_json())
        return resp

    async def _call_with_logging(self, req: ChatRequest)->ChatResponse:
        raw_prompt = self._extract_prompt(req)
        start = perf_counter()
        resp = await self._call(req)
        latency_ms = (perf_counter() - start) * 1000
        resp.cached = False
        self._log_llm_completion(
            raw_prompt=raw_prompt,
            response_content=resp.content,
            model=req.model,
            provider=req.provider,
            usage=resp.usage,
            finish_reason=resp.finish_reason,
            latency_ms=latency_ms
        )
        return resp

    async def create_stream(self, req: ChatRequest):
        """Создаёт стрим и выбрасывает LLMError при ошибках."""
        self._stream_start_time = perf_counter()
        self._stream_prompt = self._extract_prompt(req)
        try:
            llm = self.get_llm(req.provider)
            return await llm.chat.completions.create(
                model=req.model or "unknown",
                messages=[m.model_dump() for m in req.messages],
                temperature=req.temperature,
                max_tokens=req.max_tokens,
                stream=True,
                stream_options={"include_usage": True},
            )
        except RateLimitError as e:
            raise LLMRateLimitError(str(e)) from e
        except AuthenticationError as e:
            raise LLMAuthError(str(e)) from e
        except APITimeoutError as e:
            raise LLMTimeoutError(str(e)) from e
        except BadRequestError as e:
            msg = str(e).lower()
            if "content" in msg and ("filter" in msg or "policy" in msg):
                raise LLMContentFilterError(str(e)) from e
            raise LLMError(str(e)) from e
        except PermissionDeniedError as e:
            raise LLMUnsupportedCountryError(str(e)) from e
        except APIConnectionError as e:
            raise LLMError(f"connection error: {e}") from e

    async def iterate_stream(self, stream, provider, model):
        """Итерирует по стриму и выдаёт SSE-совместимые строки."""
        #logger = structlog.get_logger()
        usage = None
        finish_reason = None
        #model = None
        async for chunk in stream:
            # Чанк с контентом
            #if getattr(chunk, "model", None):
            #    model = chunk.model
            if getattr(chunk, "choices", None):
                delta = chunk.choices[0].delta
                if getattr(delta, "content", None):
                    data = ChatDelta(content=delta.content).model_dump_json()
                    yield f"data: {data}\n\n"
                if chunk.choices[0].finish_reason:
                    finish_reason = chunk.choices[0].finish_reason
            # Чанк с usage (обычно последний)
            if getattr(chunk, "usage", None):
                usage = Usage.from_openai(chunk.usage)
                data = ChatDelta(usage=usage).model_dump_json()
                yield f"data: {data}\n\n"

        yield "data: [DONE]\n\n"

        if usage:
            latency_ms = (perf_counter() - self._stream_start_time) * 1000
            self._log_llm_completion(
                raw_prompt=self._stream_prompt,
                response_content=None,          # полный ответ не сохраняется для стрима
                model=model,
                provider=provider,
                usage=usage,
                finish_reason=finish_reason,
                latency_ms=latency_ms,
            )