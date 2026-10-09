"""Тесты контекста и обработки ошибок ChatService (не-RAG путь).

Файл задумывался под «Postgres/lifespan»-прогон, но оставался пустым с ДЗ 4.1.
Здесь — поведение sliding window и мягкая обработка вложений: всё на
JsonChatRepository + фейковом LLM, без живой инфраструктуры.
"""

from io import BytesIO
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import UploadFile
from pypdf import PdfWriter

from app.chat.domain import ChatMessage
from app.chat.repositories.json_repo import JsonChatRepository
from app.chat.service import ChatService


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
    """AsyncOpenAI-подобная заглушка: create() отдаёт поток чанков и пишет kwargs."""

    def __init__(self, stream_chunks=None) -> None:
        self.stream_chunks = stream_chunks or [_Chunk("ок")]
        self.calls: list[dict] = []
        self.chat = self
        self.completions = self

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return _Stream(self.stream_chunks)


def _service(tmp_path, llm=None, context_window: int = 10):
    svc = ChatService(
        repository=JsonChatRepository(tmp_path),
        llm_ollama=llm or _FakeLLM(),
        llm_openai=None,
        llm_openrouter=None,
        context_window=context_window,
        rag_service=None,
        rag_enable_chat=False,
    )
    return svc, svc.repository


def _upload(content_type: str, data: bytes, filename: str = "file") -> UploadFile:
    return UploadFile(
        filename=filename,
        file=BytesIO(data),
        headers={"content-type": content_type},
    )


def _minimal_pdf_bytes() -> bytes:
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    buf = BytesIO()
    writer.write(buf)
    return buf.getvalue()


@pytest.mark.asyncio
async def test_unknown_chat_raises_value_error(tmp_path) -> None:
    svc, _ = _service(tmp_path)
    with pytest.raises(ValueError, match="not found"):
        async for _ in svc.send_message(uuid4(), "привет"):
            pass


@pytest.mark.asyncio
async def test_broken_media_is_reported_not_crash(tmp_path) -> None:
    """Неподдерживаемое вложение → понятное сообщение, а не обрыв SSE.

    Регрессия: исключение из media_to_part вылетало из генератора до первого
    события — бот показывал «Не получилось получить ответ от модели».
    """
    svc, repo = _service(tmp_path)
    chat = await svc.get_or_create_chat(
        owner_external_id="u1", interface="telegram", provider="ollama", model="m"
    )

    events = []
    async for e in svc.send_message(
        chat.id, "смотри видео", media=_upload("video/mp4", b"\x00\x01", "clip.mp4")
    ):
        events.append(e)

    assert len(events) == 1
    assert events[0]["type"] == "token"
    assert "Не удалось обработать вложение" in events[0]["delta"]
    # сообщение не сохранено — диалог не отравлен наполовину
    assert await repo.list_messages(chat.id) == []


@pytest.mark.asyncio
async def test_context_window_limits_history(tmp_path) -> None:
    """В LLM уходит не больше CHAT_CONTEXT_WINDOW последних сообщений."""
    llm = _FakeLLM()
    svc, repo = _service(tmp_path, llm=llm, context_window=3)
    chat = await svc.get_or_create_chat(
        owner_external_id="u2", interface="telegram", provider="ollama", model="m"
    )
    for i in range(4):
        await repo.append_message(
            chat.id, ChatMessage(chat_id=chat.id, role="user", content=f"старое {i}")
        )

    async for _ in svc.send_message(chat.id, "новый"):
        pass

    sent = llm.calls[0]["messages"]
    assert len(sent) == 3  # окно: «старое 2», «старое 3», «новый»
    assert sent[-1]["content"] == "новый"
    assert all("старое 0" not in str(m["content"]) for m in sent)


@pytest.mark.asyncio
async def test_pdf_media_reaches_llm_as_part(tmp_path) -> None:
    """PDF последнего сообщения доходит до LLM как text-part (не-RAG путь)."""
    llm = _FakeLLM()
    svc, _ = _service(tmp_path, llm=llm)
    chat = await svc.get_or_create_chat(
        owner_external_id="u3", interface="telegram", provider="ollama", model="m"
    )

    async for _ in svc.send_message(
        chat.id, "что в документе?", media=_upload("application/pdf", _minimal_pdf_bytes(), "a.pdf")
    ):
        pass

    parts = llm.calls[0]["messages"][-1]["content"]
    assert isinstance(parts, list)
    assert parts[0] == {"type": "text", "text": "что в документе?"}
    assert parts[1]["text"].startswith("[документ PDF]")
