"""HTTP-контракт /chats: создание, SSE-поток сообщения, история, очистка, feedback.

Без PG/Redis/LLM: ChatService подменяется фейком через dependency_overrides —
проверяется транспортный слой (JSON-схемы, кадры SSE, коды ответов).
Центральный эндпоинт бота — `POST /chats/{id}/messages` — до этого не был
покрыт ни одним тестом: файл пустовал с ДЗ 4.1, а Makefile его игнорировал.
"""

import json
from uuid import UUID, uuid4

import pytest
from httpx import ASGITransport, AsyncClient

from app.chat.deps import get_chat_service
from app.chat.domain import Chat, ChatMessage
from app.main import app
from app.moderation.domain import ModerationResult


class _FakeChatService:
    """Минимальный двойник ChatService: только то, что зовут роуты."""

    def __init__(self) -> None:
        self.chat = Chat(
            owner_external_id="u1",
            interface="telegram",
            provider="ollama",
            model="gemma3:4b",
        )
        self.messages: list[ChatMessage] = []
        self.send_calls: list[dict] = []
        self.moderation_allowed = True
        self.cleared = False

    async def get_or_create_chat(
        self, owner_external_id, interface, provider, model, system_prompt=None
    ) -> Chat:
        return self.chat

    async def create_chat(
        self, owner_external_id, interface, provider, model, system_prompt=None
    ) -> Chat:
        return self.chat

    async def get_chat(self, chat_id: UUID) -> Chat | None:
        return self.chat if chat_id == self.chat.id else None

    async def list_messages(self, chat_id: UUID, limit: int = 50) -> list[ChatMessage]:
        return self.messages[:limit]

    async def clear_history(self, chat_id: UUID) -> None:
        self.cleared = True

    async def check_input(self, content: str, owner_external_id: str | None = None):
        if self.moderation_allowed:
            return ModerationResult(allowed=True, layer="passed")
        return ModerationResult(allowed=False, categories=["test"], layer="regex")

    async def send_message(self, chat_id, user_content, media=None, category=None):
        self.send_calls.append(
            {
                "chat_id": chat_id,
                "content": user_content,
                "category": category,
                "has_media": media is not None,
            }
        )
        yield {"type": "token", "delta": "привет "}
        yield {"type": "token", "delta": "мир"}
        yield {"type": "message_saved", "message_id": str(uuid4())}


@pytest.fixture
def fake_service():
    svc = _FakeChatService()
    app.dependency_overrides[get_chat_service] = lambda: svc
    yield svc
    app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def _no_session_factory():
    """Фиксируем отсутствие PG, не завязываясь на порядок тестов.

    Другие тесты (admin, ratelimit, categories) выставляют app.state.session_factory
    и могут оставить его между прогонами.
    """
    had = hasattr(app.state, "session_factory")
    prev = getattr(app.state, "session_factory", None)
    app.state.session_factory = None
    yield
    if had:
        app.state.session_factory = prev
    else:
        try:
            del app.state.session_factory
        except AttributeError:
            pass


async def _client() -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


@pytest.mark.asyncio
async def test_create_chat_returns_id(fake_service) -> None:
    async with await _client() as ac:
        resp = await ac.post(
            "/chats", json={"owner_external_id": "u1", "interface": "telegram"}
        )
    assert resp.status_code == 200
    assert UUID(resp.json()["chat_id"]) == fake_service.chat.id


@pytest.mark.asyncio
async def test_message_streams_sse_frames(fake_service) -> None:
    """Кадры SSE и порядок — контракт бота: токены → message_saved → done."""
    async with await _client() as ac:
        resp = await ac.post(
            f"/chats/{fake_service.chat.id}/messages", data={"content": "привет"}
        )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")

    frames = [
        json.loads(line.removeprefix("data: "))
        for line in resp.text.splitlines()
        if line.startswith("data: ")
    ]
    assert [f["type"] for f in frames] == ["token", "token", "message_saved", "done"]
    assert "".join(f.get("delta", "") for f in frames) == "привет мир"
    assert frames[-1] == {"type": "done"}  # done всегда последний


@pytest.mark.asyncio
async def test_message_passes_content_and_category(fake_service) -> None:
    async with await _client() as ac:
        await ac.post(
            f"/chats/{fake_service.chat.id}/messages",
            data={"content": "вопрос", "category": "tarify"},
        )
    assert fake_service.send_calls == [
        {
            "chat_id": fake_service.chat.id,
            "content": "вопрос",
            "category": "tarify",
            "has_media": False,
        }
    ]


@pytest.mark.asyncio
async def test_message_moderation_blocked_before_stream(fake_service) -> None:
    """403 отдаётся ДО старта стрима (иначе статус уже отправлен как 200)."""
    fake_service.moderation_allowed = False
    async with await _client() as ac:
        resp = await ac.post(
            f"/chats/{fake_service.chat.id}/messages", data={"content": "запрещённое"}
        )
    assert resp.status_code == 403
    assert resp.json()["detail"]["code"] == "moderation_blocked"
    assert fake_service.send_calls == []  # генератор не запускался


@pytest.mark.asyncio
async def test_history_returns_messages(fake_service) -> None:
    fake_service.messages = [
        ChatMessage(chat_id=fake_service.chat.id, role="user", content="привет")
    ]
    async with await _client() as ac:
        resp = await ac.get(f"/chats/{fake_service.chat.id}/messages")
    assert resp.status_code == 200
    body = resp.json()
    assert body[0]["role"] == "user"
    assert body[0]["content"] == "привет"


@pytest.mark.asyncio
async def test_delete_history_calls_clear(fake_service) -> None:
    async with await _client() as ac:
        resp = await ac.delete(f"/chats/{fake_service.chat.id}/messages")
    assert resp.status_code == 200
    assert fake_service.cleared is True


@pytest.mark.asyncio
async def test_get_chat_unknown_returns_404(fake_service) -> None:
    async with await _client() as ac:
        resp = await ac.get(f"/chats/{uuid4()}")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_feedback_requires_postgres(fake_service) -> None:
    """Без PG фича недоступна — честный 503, а не 500."""
    async with await _client() as ac:
        resp = await ac.post(
            f"/chats/{uuid4()}/messages/{uuid4()}/feedback",
            json={"owner_external_id": "u1", "value": "up"},
        )
    assert resp.status_code == 503


@pytest.mark.asyncio
async def test_feedback_rejects_unknown_value(fake_service) -> None:
    async with await _client() as ac:
        resp = await ac.post(
            f"/chats/{uuid4()}/messages/{uuid4()}/feedback",
            json={"owner_external_id": "u1", "value": "bogus"},
        )
    assert resp.status_code == 422
