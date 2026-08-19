"""
Tests for AskFlow FSM scenario.

Verifies:
  - /ask sets state → AskFlow.waiting_for_topic and sends keyboard
  - topic selection → state goes to AskFlow.waiting_for_question,
    FSM data contains the selected topic
  - cancel topic → state cleared, message edited
"""

import pytest
from unittest.mock import AsyncMock

from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.fsm.context import FSMContext
from aiogram.types import Message, CallbackQuery, User, Chat

from bot.handlers.fsm import cmd_ask, on_topic_selected
from bot.states import AskFlow


# ── helpers ──────────────────────────────────────────────────────────────

def _make_message(text: str = "/ask") -> Message:
    """Build a realistic Message mock for FSM tests."""
    msg = AsyncMock(spec=Message)
    msg.chat = Chat(id=12345, type="private")
    msg.from_user = User(id=67890, is_bot=False, first_name="Test")
    msg.text = text
    msg.answer = AsyncMock()
    msg.edit_text = AsyncMock()
    msg.bot = AsyncMock()
    msg.bot.send_message = AsyncMock()
    msg.bot.send_chat_action = AsyncMock()
    msg.bot.send_message_draft = AsyncMock()
    return msg


def _make_callback(data: str = "topic:proj_doc") -> CallbackQuery:
    """Build a realistic CallbackQuery mock."""
    cb = AsyncMock(spec=CallbackQuery)
    cb.data = data
    cb.message = _make_message()
    cb.answer = AsyncMock()
    return cb


def _make_fsm_context() -> FSMContext:
    """Create FSMContext backed by MemoryStorage."""
    storage = MemoryStorage()
    # Use a unique key so tests don't interfere
    return FSMContext(storage=storage, key="test:67890")


def _make_backend(categories: list[dict] | None = None) -> AsyncMock:
    """BackendClient mock with list_categories."""
    backend = AsyncMock()
    backend.list_categories = AsyncMock(return_value=categories or [])
    return backend


# ── /ask command ─────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_cmd_ask_sets_state_and_sends_keyboard():
    """/ask sets AskFlow.waiting_for_topic and answers with keyboard."""
    msg = _make_message("/ask")
    ctx = _make_fsm_context()
    backend = _make_backend([{"slug": "tarify", "title": "Тарифы"}])

    await cmd_ask(msg, state=ctx, backend=backend)

    current = await ctx.get_state()
    assert current == AskFlow.waiting_for_topic

    msg.answer.assert_awaited_once()
    call_args = msg.answer.call_args
    assert call_args.kwargs.get("reply_markup") is not None
    assert "Выберите тему" in call_args.args[0]


@pytest.mark.anyio
async def test_cmd_ask_falls_back_when_backend_fails():
    """При сбое GET /categories меню строится из seed-категорий (fallback)."""
    msg = _make_message("/ask")
    ctx = _make_fsm_context()
    backend = _make_backend()
    backend.list_categories = AsyncMock(side_effect=Exception("down"))

    await cmd_ask(msg, state=ctx, backend=backend)

    data = await ctx.get_data()
    # fallback — 7 seed-категорий из DEFAULT_CATEGORIES
    assert len(data["categories"]) == 7
    assert data["categories"][0]["slug"] == "tarify"


# ── topic selection ──────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_on_topic_selected_transitions_to_waiting_for_question():
    """Selecting a topic updates FSM data and sets waiting_for_question."""
    cb = _make_callback("topic:proj_doc")
    ctx = _make_fsm_context()
    # Pre-set state to waiting_for_topic
    await ctx.set_state(AskFlow.waiting_for_topic)

    await on_topic_selected(cb, state=ctx)

    current = await ctx.get_state()
    assert current == AskFlow.waiting_for_question

    data = await ctx.get_data()
    assert data["category"] == "proj_doc"

    cb.message.edit_text.assert_awaited_once()
    cb.answer.assert_awaited_once()


@pytest.mark.anyio
async def test_on_topic_selected_slug_is_stored():
    """Different topic slug is properly stored in FSM data."""
    cb = _make_callback("topic:db_desc")
    ctx = _make_fsm_context()
    await ctx.set_state(AskFlow.waiting_for_topic)

    await on_topic_selected(cb, state=ctx)

    data = await ctx.get_data()
    assert data["category"] == "db_desc"


@pytest.mark.anyio
async def test_on_topic_all_sets_no_category():
    """"Все ПС" → category=None (поиск без фильтра по категории)."""
    cb = _make_callback("topic:all")
    ctx = _make_fsm_context()
    await ctx.set_state(AskFlow.waiting_for_topic)

    await on_topic_selected(cb, state=ctx)

    current = await ctx.get_state()
    assert current == AskFlow.waiting_for_question

    data = await ctx.get_data()
    assert data.get("category") is None

    cb.message.edit_text.assert_awaited_once()
    assert "Все ПС" in cb.message.edit_text.call_args.args[0]


# ── cancel from topic selection ──────────────────────────────────────────

@pytest.mark.anyio
async def test_on_topic_cancel_clears_state():
    """topic:cancel clears FSM state and edits message to 'Отменено'."""
    cb = _make_callback("topic:cancel")
    ctx = _make_fsm_context()
    await ctx.set_state(AskFlow.waiting_for_topic)

    await on_topic_selected(cb, state=ctx)

    current = await ctx.get_state()
    assert current is None

    cb.message.edit_text.assert_awaited_once()
    assert "Отменено" in cb.message.edit_text.call_args.args[0]

    cb.answer.assert_awaited_once()
