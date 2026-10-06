"""Переключение на резервного LLM-провайдера при недоступности основного.

Зачем. Основной провайдер (DeepSeek) — облачный: нет сети, таймаут, 429 или
протухший ключ означают, что запрос не будет обслужен вообще. Резерв — локальная
Ollama, которая живёт на том же хосте и переживает падение внешнего API. Это и
ответ на вопрос «что происходит при недоступности API модели», и страховка
демонстрации.

Что НЕ подменяем. Ошибка фильтра контента — это ответ провайдера по существу
запроса (другой провайдер ответит так же, а попытка обойти фильтр сменой
модели — уже не отказоустойчивость). Ошибка формата запроса (400) — наша
ошибка: повтор на другой модели её не исправит, только замаскирует.

Три пути к модели в проекте (все три ходят через `with_fallback`):
чат бота (`app/chat/service.py`), синтез RAG (`app/services/rag.py`) и агент
(`app/main.py`, там же штатный `with_fallbacks` из LangChain).
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Имена классов исключений, означающих «провайдер недоступен». Сравниваем по
# имени, а не по классу: llama_index и langchain оборачивают исключения
# openai/httpx в свои, и isinstance до исходного класса не достаёт.
# Цепочку причин (`__cause__`/`__context__`) разбираем отдельно — обёртка
# обычно сохраняет исходное исключение внутри.
_AVAILABILITY_ERRORS = frozenset({
    # openai
    "APIConnectionError", "APITimeoutError", "RateLimitError",
    "AuthenticationError", "PermissionDeniedError", "InternalServerError",
    "APIStatusError", "APIError", "OpenAIError",
    # httpx (транспорт под openai и llama_index)
    "ConnectError", "ConnectTimeout", "ReadTimeout", "WriteTimeout",
    "PoolTimeout", "TimeoutException", "TransportError", "RemoteProtocolError",
    # доменные — см. app/core/exceptions.py
    "LLMError", "LLMTimeoutError", "LLMRateLimitError", "LLMAuthError",
    "LLMUnsupportedCountryError",
})

# Ошибки, при которых подменять провайдера нельзя: это не сбой доступности.
_NO_FALLBACK_ERRORS = frozenset({
    "LLMContentFilterError",      # провайдер ответил по существу запроса
    "BadRequestError",            # 400 — ошибка формата, наша
    "NotFoundError",              # нет такой модели/эндпоинта — повторится и там
    "UnprocessableEntityError",
})

# Для `Runnable.with_fallbacks` из LangChain (путь агента): он фильтрует по
# isinstance, поэтому именами не обойтись — нужны сами классы.
try:  # pragma: no cover — импорт есть всегда, страховка на случай урезанной сборки
    from openai import (
        APIConnectionError as _APIConnectionError,
        APIError as _APIError,
        APITimeoutError as _APITimeoutError,
        AuthenticationError as _AuthenticationError,
        InternalServerError as _InternalServerError,
        PermissionDeniedError as _PermissionDeniedError,
        RateLimitError as _RateLimitError,
    )

    OPENAI_UNAVAILABLE_ERRORS: tuple[type[BaseException], ...] = (
        _APIConnectionError,
        _APITimeoutError,
        _RateLimitError,
        _AuthenticationError,
        _PermissionDeniedError,
        _InternalServerError,
        _APIError,
    )
except ImportError:  # pragma: no cover
    OPENAI_UNAVAILABLE_ERRORS = ()


def _exception_chain(exc: BaseException) -> list[BaseException]:
    """Исключение и его причины (`__cause__`/`__context__`), без повторов."""
    chain: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and current not in chain:
        chain.append(current)
        current = current.__cause__ or current.__context__
    return chain


def is_availability_error(exc: BaseException) -> bool:
    """True, если ошибка означает недоступность провайдера, а не ошибку запроса.

    Неизвестное исключение считается нашей ошибкой и не подменяется: молча
    уводить на резервную модель баг в коде — значит прятать его от тестов.
    """
    names = {type(e).__name__ for e in _exception_chain(exc)}
    if names & _NO_FALLBACK_ERRORS:
        return False
    return bool(names & _AVAILABILITY_ERRORS)


async def with_fallback(
    primary: Callable[[], Awaitable[T]],
    fallback: Callable[[], Awaitable[T]] | None,
    *,
    context: str,
    fallback_name: str = "",
) -> T:
    """Зовёт `primary`; при недоступности провайдера — `fallback`.

    `fallback=None` — резерв не настроен или совпал с основным провайдером
    (подменять нечем): ошибка пробрасывается как есть, без лишнего вызова.

    Уместно для стримов: подменяем только создание стрима — оно происходит до
    первого чанка. Разрыв посреди уже начатого ответа не переигрывается, иначе
    пользователь увидел бы текст дважды.
    """
    try:
        return await primary()
    except Exception as exc:  # noqa: BLE001 — решение принимает is_availability_error
        if fallback is None or not is_availability_error(exc):
            raise
        logger.warning(
            "LLM-провайдер недоступен (%s): %s — переключаюсь на %s",
            context,
            type(exc).__name__,
            fallback_name or "резерв",
        )
        return await fallback()


def resolve_fallback(
    configured_provider: str, configured_model: str, primary_provider: str
) -> tuple[str, str] | None:
    """(provider, model) резерва или None, если подменять нечем.

    Резерв выключен пустой строкой (`LLM__FALLBACK_PROVIDER=`) либо совпадением
    с основным провайдером — в этом случае переключение было бы пустым.
    """
    provider = (configured_provider or "").strip()
    if not provider or provider == primary_provider:
        return None
    return provider, configured_model
