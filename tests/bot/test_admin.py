"""Tests for admin bot commands: /stats, /users, /broadcast."""

import pytest
from unittest.mock import AsyncMock, MagicMock

import httpx
from aiogram.types import Message, Chat, User
from bot.handlers.admin import IsAdmin, cmd_stats, cmd_users, cmd_broadcast
from bot.services.backend_client import BackendClient


# ── helpers ──────────────────────────────────────────────────────────────

def _make_message(
    user_id: int = 67890,
    chat_id: int = 12345,
    text: str = "/stats",
) -> MagicMock:
    """Build a Message mock."""
    msg = AsyncMock(spec=Message)
    msg.from_user = User(id=user_id, is_bot=False, first_name="Test")
    msg.chat = Chat(id=chat_id, type="private")
    msg.text = text
    msg.answer = AsyncMock()
    return msg


def _make_backend(
    stats: dict | None = None,
    users: list[dict] | None = None,
) -> BackendClient:
    """Build BackendClient with mocked HTTP."""
    http = AsyncMock()
    backend = BackendClient(http, admin_token="test-token")
    backend.get_admin_stats = AsyncMock(return_value=stats if stats is not None else {
        "total_messages": 100, "active_users": 10,
        "avg_latency_ms": 250.0, "moderation_block_rate": 0.02, "feedback_ratio": 0.85,
    })
    backend.list_admin_users = AsyncMock(return_value=users if users is not None else [
        {"owner_external_id": "111", "interface": "telegram", "last_seen_at": "2026-07-12T12:00:00"},
        {"owner_external_id": "222", "interface": "telegram", "last_seen_at": "2026-07-12T11:00:00"},
    ])
    backend.broadcast = AsyncMock(return_value={
        "sent": 0, "failed": 0, "detail": "broadcast #1 queued (10 recipients)",
    })
    return backend


def _mock_http_error(status: int = 500) -> httpx.HTTPStatusError:
    """Create a realistic httpx.HTTPStatusError."""
    request = MagicMock()
    response = MagicMock()
    response.status_code = status
    response.json.return_value = {"detail": "test error"}
    response.headers = {}
    return httpx.HTTPStatusError(
        f"Server error '{status}'", request=request, response=response,
    )


# ── IsAdmin filter ───────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_isadmin_user_in_list(monkeypatch):
    """Admin user passes filter."""
    monkeypatch.setattr(
        "bot.handlers.admin.get_bot_settings",
        lambda: MagicMock(bot_admin_ids=[67890]),
    )
    f = IsAdmin()
    msg = _make_message(user_id=67890)
    result = await f(msg)
    assert result is True


@pytest.mark.anyio
async def test_isadmin_user_not_in_list(monkeypatch):
    """Non-admin user is rejected."""
    monkeypatch.setattr(
        "bot.handlers.admin.get_bot_settings",
        lambda: MagicMock(bot_admin_ids=[11111]),
    )
    f = IsAdmin()
    msg = _make_message(user_id=67890)
    result = await f(msg)
    assert result is False


@pytest.mark.anyio
async def test_isadmin_no_from_user():
    """Message without from_user → False."""
    msg = _make_message()
    msg.from_user = None
    f = IsAdmin()
    result = await f(msg)
    assert result is False


# ── /stats command ───────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_cmd_stats_renders_html():
    """/stats formats stats as HTML."""
    backend = _make_backend()
    msg = _make_message()

    await cmd_stats(msg, backend=backend)

    msg.answer.assert_awaited_once()
    call_text = msg.answer.call_args.args[0]
    assert "Статистика" in call_text
    assert "100" in call_text
    backend.get_admin_stats.assert_awaited_once()


@pytest.mark.anyio
async def test_cmd_stats_backend_error():
    """Backend HTTP error → user-friendly message."""
    backend = _make_backend()
    backend.get_admin_stats = AsyncMock(side_effect=_mock_http_error(500))
    msg = _make_message()

    await cmd_stats(msg, backend=backend)
    msg.answer.assert_awaited_once()
    assert "500" in msg.answer.call_args.args[0]


@pytest.mark.anyio
async def test_cmd_stats_network_error():
    """Network error → unavailable message."""
    backend = _make_backend()
    backend.get_admin_stats = AsyncMock(
        side_effect=httpx.ConnectError("no connection")
    )
    msg = _make_message()

    await cmd_stats(msg, backend=backend)
    msg.answer.assert_awaited_once()
    assert "недоступен" in msg.answer.call_args.args[0].lower()


# ── /users command ───────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_cmd_users_shows_list():
    """/users displays user list."""
    backend = _make_backend()
    msg = _make_message(text="/users")

    await cmd_users(msg, backend=backend)

    msg.answer.assert_awaited_once()
    call_text = msg.answer.call_args.args[0]
    assert "Последние пользователи" in call_text
    assert "111" in call_text
    assert "222" in call_text


@pytest.mark.anyio
async def test_cmd_users_empty():
    """No users → special message."""
    backend = _make_backend(users=[])
    msg = _make_message(text="/users")

    await cmd_users(msg, backend=backend)
    msg.answer.assert_awaited_once()
    assert "Нет данных" in msg.answer.call_args.args[0]


@pytest.mark.anyio
async def test_cmd_users_backend_error():
    """Backend error for /users."""
    backend = _make_backend()
    backend.list_admin_users = AsyncMock(side_effect=_mock_http_error(503))
    msg = _make_message(text="/users")

    await cmd_users(msg, backend=backend)
    msg.answer.assert_awaited_once()
    assert "503" in msg.answer.call_args.args[0]


# ── /broadcast command ───────────────────────────────────────────────────

@pytest.mark.anyio
async def test_cmd_broadcast_no_text():
    """/broadcast without text → usage hint."""
    backend = _make_backend()
    msg = _make_message(text="/broadcast")
    cmd_mock = MagicMock()
    cmd_mock.args = ""

    await cmd_broadcast(msg, command=cmd_mock, backend=backend)
    msg.answer.assert_awaited_once()
    assert "Использование" in msg.answer.call_args.args[0]


@pytest.mark.anyio
async def test_cmd_broadcast_success():
    """/broadcast with text → success message."""
    backend = _make_backend()
    msg = _make_message(text="/broadcast привет")
    cmd_mock = MagicMock()
    cmd_mock.args = "привет"

    await cmd_broadcast(msg, command=cmd_mock, backend=backend)
    msg.answer.assert_awaited_once()
    call_text = msg.answer.call_args.args[0]
    assert "Рассылка" in call_text
    backend.broadcast.assert_awaited_once_with("привет", interface_filter="telegram")


@pytest.mark.anyio
async def test_cmd_broadcast_error():
    """/broadcast with backend error."""
    backend = _make_backend()
    backend.broadcast = AsyncMock(side_effect=_mock_http_error(500))
    msg = _make_message(text="/broadcast тест")
    cmd_mock = MagicMock()
    cmd_mock.args = "тест"

    await cmd_broadcast(msg, command=cmd_mock, backend=backend)
    msg.answer.assert_awaited_once()
    assert "500" in msg.answer.call_args.args[0]
