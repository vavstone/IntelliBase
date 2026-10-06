"""Тесты гейта доступа к боту (bot/middlewares/access.py).

Проверяем, что посторонний не доходит до обработчиков, что отказ содержит id
пользователя (его передают администратору) и что недоступный бэкенд не
превращается в ложное «доступ не выдан».
"""

from unittest.mock import AsyncMock

import pytest
from aiogram.types import Chat, Message, User

from bot.middlewares.access import AccessMiddleware


def _make_message(chat_id: int = 12345, text: str = "/ask привет") -> Message:
    msg = AsyncMock(spec=Message)
    msg.chat = Chat(id=chat_id, type="private")
    msg.from_user = User(id=chat_id, is_bot=False, first_name="Test")
    msg.text = text
    return msg


def _make_backend(allowed: bool = True, error: Exception | None = None):
    backend = AsyncMock()
    if error is not None:
        backend.check_access = AsyncMock(side_effect=error)
    else:
        backend.check_access = AsyncMock(return_value=allowed)
    return backend


@pytest.mark.asyncio
async def test_allowed_user_reaches_handler():
    backend = _make_backend(allowed=True)
    mw = AccessMiddleware(backend)
    handler = AsyncMock(return_value="handled")
    msg = _make_message()

    result = await mw(handler, msg, {})

    assert result == "handled"
    handler.assert_awaited_once()
    backend.check_access.assert_awaited_once_with(12345)


@pytest.mark.asyncio
async def test_denied_user_is_stopped_with_his_id():
    backend = _make_backend(allowed=False)
    mw = AccessMiddleware(backend)
    handler = AsyncMock()
    msg = _make_message(chat_id=777)

    result = await mw(handler, msg, {})

    assert result is None
    handler.assert_not_awaited()           # до обработчиков не дошло
    text = msg.answer.call_args.args[0]
    assert "Доступ не выдан" in text
    assert "777" in text                   # id нужен администратору


@pytest.mark.asyncio
async def test_backend_error_is_not_reported_as_denied():
    backend = _make_backend(error=RuntimeError("connection refused"))
    mw = AccessMiddleware(backend)
    handler = AsyncMock()
    msg = _make_message()

    assert await mw(handler, msg, {}) is None
    handler.assert_not_awaited()
    text = msg.answer.call_args.args[0]
    assert "недоступен" in text
    assert "Доступ не выдан" not in text   # не путаем отказ с недоступностью


@pytest.mark.asyncio
async def test_verdict_is_cached_between_updates():
    backend = _make_backend(allowed=True)
    mw = AccessMiddleware(backend, cache_ttl=60)
    handler = AsyncMock()

    await mw(handler, _make_message(), {})
    await mw(handler, _make_message(), {})

    backend.check_access.assert_awaited_once()  # второй апдейт — из кэша
    assert handler.await_count == 2


@pytest.mark.asyncio
async def test_cache_does_not_hide_revoked_access():
    """После истечения TTL доступ перепроверяется (отзыв доступа работает)."""
    backend = _make_backend(allowed=True)
    mw = AccessMiddleware(backend, cache_ttl=0)
    handler = AsyncMock()

    await mw(handler, _make_message(), {})
    backend.check_access.return_value = False
    await mw(handler, _make_message(), {})

    assert backend.check_access.await_count == 2
    assert handler.await_count == 1  # второй апдейт отклонён


@pytest.mark.asyncio
async def test_message_without_chat_is_passed_through():
    """Не у всех апдейтов есть чат (например, inline-запросы) — не блокируем."""
    backend = _make_backend()
    mw = AccessMiddleware(backend)
    handler = AsyncMock(return_value="handled")

    assert await mw(handler, object(), {}) == "handled"
    backend.check_access.assert_not_awaited()
