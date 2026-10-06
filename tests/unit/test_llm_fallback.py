"""Тесты переключения на резервного LLM-провайдера.

Покрывают `app/services/llm_fallback.py` и его проводку в трёх путях к модели:
чат бота (стрим и condense), синтез RAG и разрешение резерва из настроек.

Отдельно проверяем то, что легко сломать при fallback: резервной Ollama нельзя
отправлять `stream_options` и `thinking` — это поля DeepSeek/OpenAI, а не
общего протокола.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from llama_index.core.schema import NodeWithScore, TextNode
from openai import APIConnectionError, BadRequestError, RateLimitError

from app.chat.service import ChatService
from app.core.exceptions import LLMContentFilterError, LLMTimeoutError
from app.services.llm_fallback import (
    is_availability_error,
    resolve_fallback,
    with_fallback,
)
from app.services.rag import RAGService

_REQ = httpx.Request("POST", "https://api.deepseek.com/chat/completions")


def _http_error(cls, status: int):
    return cls("boom", response=httpx.Response(status, request=_REQ), body=None)


# ── классификация ошибок ─────────────────────────────────────────────────

@pytest.mark.parametrize(
    "exc",
    [
        APIConnectionError(request=_REQ),
        _http_error(RateLimitError, 429),
        LLMTimeoutError("нет ответа"),
    ],
)
def test_availability_errors_are_recognised(exc) -> None:
    """Нет связи, таймаут, 429 и доменные ошибки — провайдер недоступен."""
    assert is_availability_error(exc) is True


def test_content_filter_is_not_availability_error() -> None:
    """Фильтр контента — ответ провайдера по существу, а не сбой доступности."""
    assert is_availability_error(LLMContentFilterError("заблокировано")) is False


def test_bad_request_is_not_availability_error() -> None:
    """400 — ошибка формата запроса: на другой модели повторится."""
    assert is_availability_error(_http_error(BadRequestError, 400)) is False


def test_unknown_error_is_not_availability_error() -> None:
    """Незнакомое исключение — наша ошибка; подменой модели её не прячем."""
    assert is_availability_error(ValueError("баг в коде")) is False


def test_wrapped_cause_is_unwrapped() -> None:
    """llama_index и langchain оборачивают причину — смотрим цепочку."""
    try:
        try:
            raise APIConnectionError(request=_REQ)
        except Exception as inner:  # noqa: BLE001
            raise LLMTimeoutError("обёртка") from inner
    except LLMTimeoutError as wrapped:
        assert is_availability_error(wrapped) is True


# ── разрешение резерва ───────────────────────────────────────────────────

def test_resolve_fallback_returns_pair() -> None:
    assert resolve_fallback("ollama", "qwen2.5:3b", "deepseek") == ("ollama", "qwen2.5:3b")


def test_resolve_fallback_disabled_by_empty_provider() -> None:
    assert resolve_fallback("", "qwen2.5:3b", "deepseek") is None


def test_resolve_fallback_skips_same_provider() -> None:
    """Резерв совпал с основным — подменять нечем."""
    assert resolve_fallback("ollama", "qwen2.5:3b", "ollama") is None


# ── with_fallback ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_with_fallback_returns_primary_result() -> None:
    fallback = AsyncMock(return_value="резерв")
    result = await with_fallback(
        AsyncMock(return_value="основной"), fallback, context="test"
    )
    assert result == "основной"
    fallback.assert_not_awaited()


@pytest.mark.asyncio
async def test_with_fallback_switches_on_availability_error() -> None:
    async def primary():
        raise APIConnectionError(request=_REQ)

    result = await with_fallback(
        primary, AsyncMock(return_value="резерв"), context="test", fallback_name="qwen2.5:3b"
    )
    assert result == "резерв"


@pytest.mark.asyncio
async def test_with_fallback_reraises_content_filter() -> None:
    """Фильтр контента не подменяется — иначе смена модели обходила бы модерацию."""
    fallback = AsyncMock(return_value="резерв")

    async def primary():
        raise LLMContentFilterError("заблокировано")

    with pytest.raises(LLMContentFilterError):
        await with_fallback(primary, fallback, context="test")
    fallback.assert_not_awaited()


@pytest.mark.asyncio
async def test_with_fallback_without_target_reraises() -> None:
    async def primary():
        raise APIConnectionError(request=_REQ)

    with pytest.raises(APIConnectionError):
        await with_fallback(primary, None, context="test")


@pytest.mark.asyncio
async def test_with_fallback_propagates_fallback_failure() -> None:
    """Резерв тоже упал — наружу уходит его ошибка, а не исходная."""
    async def primary():
        raise APIConnectionError(request=_REQ)

    async def fallback():
        raise LLMTimeoutError("резерв не ответил")

    with pytest.raises(LLMTimeoutError):
        await with_fallback(primary, fallback, context="test")


# ── проводка: чат бота ───────────────────────────────────────────────────

class _FakeLLM:
    """AsyncOpenAI-подобный объект для ChatService."""

    def __init__(self, error: Exception | None = None, chunks=None, completion=None) -> None:
        self.error = error
        self.chunks = chunks or ["чанк"]
        # Нестриминговый ответ (condense) — если задан, отдаём его.
        self.completion = completion
        self.calls: list[dict] = []
        self.chat = self
        self.completions = self

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        if self.completion is not None:
            return self.completion
        return _Stream(self.chunks)


class _Stream:
    def __init__(self, chunks) -> None:
        self._chunks = list(chunks)
        self._i = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._i >= len(self._chunks):
            raise StopAsyncIteration
        self._i += 1
        return self._chunks[self._i - 1]


def _chat_service(primary, fallback, fallback_provider: str = "ollama") -> ChatService:
    return ChatService(
        repository=AsyncMock(),
        llm_ollama=fallback,
        llm_openai=None,
        llm_openrouter=None,
        llm_deepseek=primary,
        default_provider="deepseek",
        default_model="deepseek-v4-flash",
        fallback_provider=fallback_provider,
        fallback_model="qwen2.5:3b",
    )


@pytest.mark.asyncio
async def test_chat_stream_switches_to_fallback() -> None:
    primary = _FakeLLM(error=APIConnectionError(request=_REQ))
    fallback = _FakeLLM(chunks=["ответ"])
    svc = _chat_service(primary, fallback)

    stream = await svc._create_stream(
        primary, "deepseek", "deepseek-v4-flash", [{"role": "user", "content": "привет"}]
    )

    assert [c async for c in stream] == ["ответ"]
    assert len(primary.calls) == 1 and len(fallback.calls) == 1
    assert fallback.calls[0]["model"] == "qwen2.5:3b"


@pytest.mark.asyncio
async def test_chat_stream_does_not_send_stream_options_to_ollama() -> None:
    """`stream_options` — поле облачных провайдеров; Ollama его не понимает."""
    primary = _FakeLLM(error=APIConnectionError(request=_REQ))
    fallback = _FakeLLM()
    svc = _chat_service(primary, fallback)

    await svc._create_stream(primary, "deepseek", "deepseek-v4-flash", [])

    assert primary.calls[0]["stream_options"] == {"include_usage": True}
    assert "stream_options" not in fallback.calls[0]


@pytest.mark.asyncio
async def test_chat_stream_does_not_switch_on_content_filter() -> None:
    primary = _FakeLLM(error=LLMContentFilterError("заблокировано"))
    fallback = _FakeLLM()
    svc = _chat_service(primary, fallback)

    with pytest.raises(LLMContentFilterError):
        await svc._create_stream(primary, "deepseek", "deepseek-v4-flash", [])

    assert fallback.calls == []


@pytest.mark.asyncio
async def test_chat_stream_without_fallback_configured() -> None:
    """Резерв выключен — ошибка уходит наружу, лишнего вызова нет."""
    primary = _FakeLLM(error=APIConnectionError(request=_REQ))
    fallback = _FakeLLM()
    svc = _chat_service(primary, fallback, fallback_provider="")

    with pytest.raises(APIConnectionError):
        await svc._create_stream(primary, "deepseek", "deepseek-v4-flash", [])

    assert fallback.calls == []


@pytest.mark.asyncio
async def test_chat_condense_switches_to_fallback() -> None:
    """condense — служебный вызов; на резерве он тоже должен отвечать."""
    primary = _FakeLLM(error=APIConnectionError(request=_REQ))
    fallback = _FakeLLM(
        completion=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="переписанный вопрос"))]
        )
    )
    svc = _chat_service(primary, fallback)
    chat = SimpleNamespace(provider="deepseek", model="deepseek-v4-flash")
    history = [
        SimpleNamespace(role="user", content="что по тарифам"),
        SimpleNamespace(role="assistant", content="ответ"),
    ]

    result = await svc._condense(chat, "а по Малахиту?", history, primary)

    assert result == "переписанный вопрос"
    assert "thinking" not in fallback.calls[0]


# ── проводка: синтез RAG ─────────────────────────────────────────────────

class _FakeLlamaLLM:
    """Объект с `acomplete` — как llama_index LLM в `_synthesize`."""

    def __init__(self, error: Exception | None = None, text: str = "Ответ [1]") -> None:
        self.error = error
        self.text = text
        self.calls = 0

    async def acomplete(self, prompt: str):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.text


def _rag_service(primary, fallback) -> RAGService:
    svc = object.__new__(RAGService)
    svc._llm = primary
    svc._fallback_llm = fallback
    svc._settings = SimpleNamespace(
        rag_score_threshold=0.5,
        llm=SimpleNamespace(fallback_model="qwen2.5:3b"),
    )
    return svc


def _nodes() -> list[NodeWithScore]:
    node = TextNode(text="Тарифы версии 2.4 вступают в силу с 1 марта.", metadata={"source": "t.pdf", "page": 1})
    return [NodeWithScore(node=node, score=0.9)]


@pytest.mark.asyncio
async def test_rag_synthesize_switches_to_fallback() -> None:
    primary = _FakeLlamaLLM(error=LLMTimeoutError("DeepSeek не ответил"))
    fallback = _FakeLlamaLLM(text="Тарифы вступают в силу с 1 марта [1].")
    svc = _rag_service(primary, fallback)

    result = await svc._synthesize("когда вступают тарифы?", _nodes())

    assert primary.calls == 1 and fallback.calls == 1
    assert result["confident"] is True
    assert "1 марта" in result["answer"]


@pytest.mark.asyncio
async def test_rag_synthesize_without_fallback_raises() -> None:
    primary = _FakeLlamaLLM(error=LLMTimeoutError("DeepSeek не ответил"))
    svc = _rag_service(primary, None)

    with pytest.raises(LLMTimeoutError):
        await svc._synthesize("когда вступают тарифы?", _nodes())


@pytest.mark.asyncio
async def test_rag_synthesize_refuses_without_llm_call() -> None:
    """score-guard срабатывает до модели — резерв тоже не зовём."""
    primary = _FakeLlamaLLM()
    fallback = _FakeLlamaLLM()
    svc = _rag_service(primary, fallback)

    result = await svc._synthesize("вопрос вне базы", [NodeWithScore(node=TextNode(text="x"), score=0.1)])

    assert result["confident"] is False
    assert primary.calls == 0 and fallback.calls == 0


# ── проводка: резервная модель агента ────────────────────────────────────

def test_agent_fallback_kwargs_for_ollama() -> None:
    """У агента своя модель (LangChain) — параметры под провайдера собираются
    отдельно от клиентов LLMService."""
    from app.core.config import get_settings
    from app.main import _provider_chat_kwargs

    cfg = get_settings()
    kwargs = _provider_chat_kwargs(cfg, "ollama", "qwen2.5:3b")

    assert kwargs["model"] == "qwen2.5:3b"
    assert kwargs["base_url"] == cfg.llm.ollama_base_url
    assert kwargs["api_key"] == "ollama"


def test_agent_fallback_kwargs_unknown_provider() -> None:
    """Незнакомый провайдер — None: агент остаётся без резерва, а не падает."""
    from app.core.config import get_settings
    from app.main import _provider_chat_kwargs

    assert _provider_chat_kwargs(get_settings(), "нет-такого", "x") is None
