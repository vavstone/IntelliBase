"""Трейсинг в Phoenix через OpenInference (опциональный runtime-путь).

Включается флагом `PHOENIX_ENABLED=true`; LlamaIndexInstrumentor дополнительно
требует группы зависимостей `tracing` (`uv sync --extra tracing`). По умолчанию
выключено — сервис поднимается без трейсинга, спаны не пишутся.

Инструменторы подключаются один раз при старте:
- OpenAI — вызовы /chat, /chats, модерация (сырой openai-SDK);
- LlamaIndex — вызовы RAG (retriever с similarity scores, LLM с prompt/usage);
- LangChain/LangGraph — агент (`/agent/*`): узлы графа, tool-calls, LLM-вызовы;
- FastAPI — серверный спан на HTTP-запрос (см. `instrument_fastapi_app`).

Порядок вызовов на старте: `setup_tracing(settings)` → `instrument_fastapi_app(app, ...)`
до первого запроса (Starlette не разрешает добавлять middleware после старта).
"""

import logging
from importlib.util import find_spec
from typing import TYPE_CHECKING

from app.core.config import Settings

if TYPE_CHECKING:  # pragma: no cover - только для аннотаций
    from fastapi import FastAPI
    from opentelemetry.sdk.trace import TracerProvider

logger = logging.getLogger(__name__)


def _shim_llama_index_base_agent() -> None:
    """Шим для openinference-instrumentation-llama-index 3.3.x.

    Пакет всё ещё импортирует `llama_index.core.base.agent.types`
    (BaseAgent/BaseAgentWorker), но в llama-index 0.14 агенты переехали в
    `llama_index.core.agent`, а старый модуль удалён. В _handler.py эти классы
    используются только как аннотации типов, поэтому подставляем заглушки.
    """
    import sys
    import types

    if "llama_index.core.base.agent" in sys.modules:
        return
    import llama_index.core.base as _base

    agent_pkg = types.ModuleType("llama_index.core.base.agent")
    agent_types = types.ModuleType("llama_index.core.base.agent.types")

    class BaseAgent:  # noqa: D101
        pass

    class BaseAgentWorker:  # noqa: D101
        pass

    agent_types.BaseAgent = BaseAgent
    agent_types.BaseAgentWorker = BaseAgentWorker
    agent_pkg.types = agent_types
    sys.modules["llama_index.core.base.agent"] = agent_pkg
    sys.modules["llama_index.core.base.agent.types"] = agent_types
    _base.agent = agent_pkg


def setup_tracing(settings: Settings) -> "TracerProvider | None":
    """Регистрирует инструментеры OpenAI, LangChain и LlamaIndex → Phoenix.

    Возвращает tracer provider (его же нужно передать в `instrument_fastapi_app`),
    либо None, если трейсинг выключен или эндпоинт не задан.
    """
    if not settings.phoenix_enabled:
        return None
    endpoint = settings.phoenix_collector_endpoint
    if not endpoint:
        logger.warning("phoenix_enabled=true, но PHOENIX_COLLECTOR_ENDPOINT пуст")
        return None

    from openinference.instrumentation.openai import OpenAIInstrumentor
    from phoenix.otel import register

    tracer_provider = register(project_name="diploma-fastapi", endpoint=endpoint)
    OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)

    instrumented = ["OpenAI"]
    if find_spec("openinference.instrumentation.langchain") is not None:
        from openinference.instrumentation.langchain import LangChainInstrumentor

        LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
        instrumented.append("LangChain")
    else:
        logger.warning(
            "LangChainInstrumentor не установлен — трейсов агента не будет "
            "(uv sync --extra tracing)"
        )

    if find_spec("openinference.instrumentation.llama_index") is not None:
        _shim_llama_index_base_agent()
        from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

        LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)
        instrumented.append("LlamaIndex")
    else:
        logger.warning(
            "LlamaIndexInstrumentor не установлен — uv sync --extra tracing"
        )

    logger.info("Phoenix-трейсинг включён (%s): %s", ", ".join(instrumented), endpoint)
    return tracer_provider


def instrument_fastapi_app(
    app: "FastAPI", tracer_provider: "TracerProvider | None", excluded_urls: str = ""
) -> bool:
    """Серверный спан на каждый HTTP-запрос → все спаны запроса в одном трейсе.

    Без него спаны разных библиотек расходятся по отдельным трейсам: инструментеры
    LangChain и LlamaIndex не привязывают спаны к OTel-контексту (у LangChain это
    сделано намеренно — см. комментарий в `_tracer.py`), поэтому в UI один вопрос
    выглядит как 4–6 независимых трейсов вместо одного дерева «запрос → RAG → LLM».

    Вызывать до первого запроса: Starlette запрещает добавлять middleware после
    старта (`RuntimeError: Cannot add middleware after an application has started`).
    """
    if tracer_provider is None:
        return False
    if find_spec("opentelemetry.instrumentation.fastapi") is None:
        logger.warning(
            "FastAPIInstrumentor не установлен — трейсы не будут сгруппированы "
            "по запросам (uv sync --extra tracing)"
        )
        return False

    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    FastAPIInstrumentor.instrument_app(
        app,
        tracer_provider=tracer_provider,
        excluded_urls=excluded_urls or None,
        # Служебные спаны чтения/записи ASGI («http receive/send») в дереве
        # только шумят: в трейсе остаётся один серверный спан на запрос.
        exclude_spans=["receive", "send"],
    )
    logger.info("FastAPI-инструментер подключён (исключены: %s)", excluded_urls or "—")
    return True
