"""Shared fixtures for bot tests."""

import pytest
from unittest.mock import AsyncMock


@pytest.fixture
def mock_httpx_client():
    """httpx.AsyncClient mock — предотвращает реальные HTTP-запросы."""
    client = AsyncMock()
    client.post = AsyncMock()
    client.delete = AsyncMock()
    client.aclose = AsyncMock()
    return client


@pytest.fixture
def mock_message():
    """aiogram Message mock."""
    msg = AsyncMock()
    msg.chat.id = 12345
    msg.text = "test message"
    msg.answer = AsyncMock()
    msg.edit_text = AsyncMock()
    msg.bot.send_message = AsyncMock()
    msg.bot.send_chat_action = AsyncMock()
    msg.bot.send_message_draft = AsyncMock()
    return msg


@pytest.fixture
def mock_callback_query():
    """aiogram CallbackQuery mock."""
    cb = AsyncMock()
    cb.data = "topic:proj_doc"
    cb.message.chat.id = 12345
    cb.message.edit_text = AsyncMock()
    cb.answer = AsyncMock()
    return cb


@pytest.fixture
def mock_state():
    """aiogram FSMContext mock."""
    state = AsyncMock()
    state.set_state = AsyncMock()
    state.get_state = AsyncMock(return_value=None)
    state.get_data = AsyncMock(return_value={})
    state.update_data = AsyncMock()
    state.clear = AsyncMock()
    return state
