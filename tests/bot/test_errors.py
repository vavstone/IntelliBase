"""Тесты глобального обработчика ошибок бота (bot/handlers/errors.py).

Проверяют, что непойманное исключение хендлера превращается в честный ответ
пользователю (а не в молчание), и что сбой самого ответа не пробрасывается.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from bot.handlers.errors import USER_MESSAGE, on_unhandled_error


def _event(message=None, callback_query=None) -> SimpleNamespace:
    update = SimpleNamespace(message=message, callback_query=callback_query)
    return SimpleNamespace(update=update, exception=RuntimeError("backend down"))


@pytest.mark.asyncio
async def test_message_error_answers_user() -> None:
    message = SimpleNamespace(answer=AsyncMock())
    handled = await on_unhandled_error(_event(message=message))
    assert handled is True  # исключение погашено
    message.answer.assert_awaited_once_with(USER_MESSAGE)


@pytest.mark.asyncio
async def test_callback_error_shows_alert() -> None:
    callback = SimpleNamespace(answer=AsyncMock())
    handled = await on_unhandled_error(_event(callback_query=callback))
    assert handled is True
    callback.answer.assert_awaited_once()
    assert callback.answer.await_args.kwargs.get("show_alert") is True


@pytest.mark.asyncio
async def test_send_failure_does_not_propagate() -> None:
    """Если и ответ не ушёл (Telegram недоступен) — обработчик всё равно True."""
    message = SimpleNamespace(answer=AsyncMock(side_effect=RuntimeError("tg down")))
    assert await on_unhandled_error(_event(message=message)) is True


@pytest.mark.asyncio
async def test_update_without_message_or_callback_is_quiet() -> None:
    assert await on_unhandled_error(_event()) is True
