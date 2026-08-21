"""Трейсинг в Phoenix через OpenInference (опциональный runtime-путь).

Включается флагом `PHOENIX_ENABLED=true`; LlamaIndexInstrumentor дополнительно
требует группы зависимостей `tracing` (`uv sync --extra tracing`). По умолчанию
выключено — сервис поднимается без трейсинга, спаны не пишутся.

Инструменторы подключаются один раз при старте (lifespan):
- OpenAI — вызовы /chat, /chats, модерация;
- LlamaIndex — вызовы RAG (retriever с similarity scores, LLM с prompt/usage).
"""

import logging
from importlib.util import find_spec

from app.core.config import Settings

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


def setup_tracing(settings: Settings) -> bool:
    """Регистрирует OpenAI- и LlamaIndex-инструменторы → Phoenix.

    Возвращает True, если трейсинг включён и хотя бы один инструментор поднят.
    """
    if not settings.phoenix_enabled:
        return False
    endpoint = settings.phoenix_collector_endpoint
    if not endpoint:
        logger.warning("phoenix_enabled=true, но PHOENIX_COLLECTOR_ENDPOINT пуст")
        return False

    from openinference.instrumentation.openai import OpenAIInstrumentor
    from phoenix.otel import register

    tracer_provider = register(project_name="diploma-fastapi", endpoint=endpoint)
    OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)

    instrumented = ["OpenAI"]
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
    return True
