"""Тесты диалогового RAG-пути в ChatService (Б5.5).

Без живой инфраструктуры: фейковые репозиторий / rag-сервис / LLM.
Покрывают:
- _rag_active (включается только при rag_service + rag_enable_chat);
- _build_rag_messages (system + предыстория + контекст в финальном вопросе);
- _condense (переписывает follow-up с учётом истории);
- _send_rag: отказ по score-guard и генерация со стримингом + sources.

Внимание: намеренно НЕ импортируем llama_index/sentence_transformers —
их нативный импорт (pyarrow) падает на Windows при одновременной загрузке
с app.chat.service. Фейковой ноде достаточно score/metadata/get_content.
"""

from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.chat.domain import ChatMessage
from app.chat.repositories.json_repo import JsonChatRepository
from app.chat.service import ChatService

# Локальная копия REFUSAL_TEXT из app/services/rag.py (см. комментарий выше).
REFUSAL_TEXT = "В базе знаний я не нашёл ответа на этот вопрос."


class _FakeNode:
    def __init__(self, text: str, score: float = 0.9) -> None:
        self.text = text
        self.score = score
        self.metadata = {"source": "a.pdf"}

    def get_content(self) -> str:
        return self.text


class _FakeRag:
    def __init__(self, nodes: list) -> None:
        self._nodes = nodes
        self.retrieve_calls: list[tuple] = []

    async def retrieve(self, q: str, filters=None) -> list:
        self.retrieve_calls.append((q, filters))
        return self._nodes


class _Chunk:
    usage = None

    def __init__(self, content: str) -> None:
        self.choices = [SimpleNamespace(delta=SimpleNamespace(content=content))]


class _Stream:
    def __init__(self, chunks: list) -> None:
        self._chunks = list(chunks)
        self._i = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._i >= len(self._chunks):
            raise StopAsyncIteration
        c = self._chunks[self._i]
        self._i += 1
        return c


class _FakeLLM:
    """Минимальный AsyncOpenAI-подобный объект: llm.chat.completions.create(...).

    При `completion` возвращает его (нестриминг, для condense), иначе —
    async-итератор `_Stream` из `stream_chunks`.
    """

    def __init__(self, stream_chunks=None, completion=None) -> None:
        self.stream_chunks = stream_chunks or []
        self.completion = completion
        self.calls: list[dict] = []
        self.chat = self
        self.completions = self

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.completion is not None:
            return self.completion
        return _Stream(self.stream_chunks)


def _service(tmp_path, rag=None, llm=None) -> tuple[ChatService, JsonChatRepository]:
    repo = JsonChatRepository(tmp_path)
    svc = ChatService(
        repository=repo,
        llm_ollama=llm or _FakeLLM(),
        llm_openai=None,
        llm_openrouter=None,
        rag_service=rag,
        rag_enable_chat=True,
        rag_condense_enabled=False,
        rag_score_threshold=0.8,
    )
    return svc, repo


def test_rag_active_requires_service_and_flag(tmp_path) -> None:
    svc_off, _ = _service(tmp_path, rag=_FakeRag([]))
    svc_off.rag_enable_chat = False
    assert svc_off._rag_active() is False

    svc_none, _ = _service(tmp_path, rag=None)
    assert svc_none._rag_active() is False

    svc_on, _ = _service(tmp_path, rag=_FakeRag([]))
    assert svc_on._rag_active() is True


def test_build_rag_messages_injects_context_and_keeps_history(tmp_path) -> None:
    svc, _ = _service(tmp_path)
    chat_id = uuid4()
    history = [
        ChatMessage(chat_id=chat_id, role="user", content="что такое ТРОИС?"),
        ChatMessage(chat_id=chat_id, role="assistant", content="ТРОИС — это …"),
        ChatMessage(chat_id=chat_id, role="user", content="а для чего?"),
    ]
    msgs = svc._build_rag_messages(history, "а для чего?", "[1] контекст про ТРОИС")
    assert msgs[0]["role"] == "system"
    # system + предыстория (history[:-1]) + финальный вопрос
    assert len(msgs) == 4
    assert msgs[1]["role"] == "user" and msgs[1]["content"] == "что такое ТРОИС?"
    final = msgs[-1]
    assert final["role"] == "user"
    assert "[1] контекст про ТРОИС" in final["content"]
    assert "а для чего?" in final["content"]


@pytest.mark.asyncio
async def test_condense_rewrites_followup(tmp_path) -> None:
    completion = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="состав ТРОИС"))]
    )
    llm = _FakeLLM(completion=completion)
    svc, _ = _service(tmp_path, llm=llm)
    chat_id = uuid4()
    history = [
        ChatMessage(chat_id=chat_id, role="user", content="Что входит в КПС «Тарифы»?"),
        ChatMessage(chat_id=chat_id, role="assistant", content="…"),
        ChatMessage(chat_id=chat_id, role="user", content="а для них?"),
    ]
    chat = SimpleNamespace(id=chat_id, model="qwen2.5:3b")
    out = await svc._condense(chat, "а для них?", history, llm)
    assert out == "состав ТРОИС"


@pytest.mark.asyncio
async def test_send_rag_refuses_when_no_nodes(tmp_path) -> None:
    rag = _FakeRag([])  # пустой retrieval → score-guard
    llm = _FakeLLM(stream_chunks=[_Chunk("не должен вызываться")])
    svc, _ = _service(tmp_path, rag=rag, llm=llm)
    chat = await svc.get_or_create_chat(
        owner_external_id="u1", interface="telegram", provider="ollama", model="qwen2.5:3b"
    )
    events = []
    async for e in svc.send_message(chat.id, "вопрос"):
        events.append(e)

    tokens = "".join(e.get("delta", "") for e in events if e.get("type") == "token")
    srcs = [e for e in events if e.get("type") == "sources"]
    saved = [e for e in events if e.get("type") == "message_saved"]

    assert tokens == REFUSAL_TEXT
    assert srcs and srcs[0]["sources"] == []
    assert saved
    assert llm.calls == []  # LLM не вызывался при отказе


@pytest.mark.asyncio
async def test_send_rag_streams_and_emits_sources(tmp_path) -> None:
    rag = _FakeRag([_FakeNode("контекст про ТРОИС", score=0.9)])
    llm = _FakeLLM(stream_chunks=[_Chunk("ответ "), _Chunk("с цитатой [1]")])
    svc, _ = _service(tmp_path, rag=rag, llm=llm)
    chat = await svc.get_or_create_chat(
        owner_external_id="u2", interface="telegram", provider="ollama", model="qwen2.5:3b"
    )
    events = []
    async for e in svc.send_message(chat.id, "Что входит в состав ТРОИС?"):
        events.append(e)

    tokens = "".join(e.get("delta", "") for e in events if e.get("type") == "token")
    srcs = [e for e in events if e.get("type") == "sources"]
    saved = [e for e in events if e.get("type") == "message_saved"]

    assert tokens == "ответ с цитатой [1]"
    assert srcs and len(srcs[0]["sources"]) == 1
    assert srcs[0]["sources"][0]["file_name"] == "a.pdf"
    assert saved
    assert rag.retrieve_calls == [("Что входит в состав ТРОИС?", None)]


@pytest.mark.asyncio
async def test_send_rag_passes_category_filter_to_retrieve(tmp_path) -> None:
    """При заданной категории retrieval получает MetadataFilters по category."""
    rag = _FakeRag([_FakeNode("контекст про Тарифы", score=0.9)])
    llm = _FakeLLM(stream_chunks=[_Chunk("ответ")])
    svc, _ = _service(tmp_path, rag=rag, llm=llm)
    chat = await svc.get_or_create_chat(
        owner_external_id="u3", interface="telegram", provider="ollama", model="qwen2.5:3b"
    )
    async for _ in svc.send_message(chat.id, "вопрос", category="tarify"):
        pass

    assert rag.retrieve_calls
    _, filters = rag.retrieve_calls[0]
    assert filters is not None
    keys = {f.key for f in filters.filters}
    assert keys == {"visibility", "category"}
    cat = next(f for f in filters.filters if f.key == "category")
    assert cat.value == ["tarify"]
