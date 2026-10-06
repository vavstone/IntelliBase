"""Тесты агентного сценария бота с подтверждением (HIL).

Проверяем шаг 4 демо: `/agent <задача>` → пауза на опасном действии →
превью черновика + inline-кнопки → `hil:approve|reject` → `/agent/resume`.

Backend замокан: тесты не ходят в FastAPI и не поднимают LangGraph.
"""

from unittest.mock import AsyncMock

import httpx
import pytest
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramBadRequest
from aiogram.types import CallbackQuery, Chat, Message, User

from bot.handlers.agent import (
    HIL_HINT,
    cmd_agent,
    new_thread_id,
    on_hil_decision,
    render_interrupt,
    render_result,
)
from bot.keyboards.inline import HIL_CB_PREFIX, hil_kb

# ── helpers ──────────────────────────────────────────────────────────────

INTERRUPT_RESULT = {
    "status": "interrupted",
    "thread_id": "tg12345-abcdef01",
    "answer": None,
    "tool_results": [
        {
            "name": "search_knowledge_base",
            "args": {"query": "тарифы"},
            "result": '{"top_score": 0.86}',
        }
    ],
    "interrupt": {
        "type": "approve_send",
        "preview": {
            "chat_id": "12345",
            "text": "Отчёт по тарифам готов",
            "tool_call_id": "call-1",
        },
    },
}

DONE_RESULT = {
    "status": "done",
    "thread_id": "tg12345-abcdef01",
    "answer": "Сообщение отправлено клиенту.",
    "tool_results": [
        {"name": "search_knowledge_base", "args": {}, "result": "..."},
        {
            "name": "send_telegram_message",
            "args": {"chat_id": "12345"},
            "result": "сообщение отправлено: Отчёт по тарифам готов",
        },
    ],
    "interrupt": None,
}


def _make_message(text: str = "/agent отправь уведомление") -> Message:
    msg = AsyncMock(spec=Message)
    msg.chat = Chat(id=12345, type="private")
    msg.from_user = User(id=67890, is_bot=False, first_name="Test")
    msg.text = text
    # edit_text/edit_reply_markup в aiogram — sync-методы, возвращающие
    # awaitable; под spec=Message они становятся MagicMock, поэтому задаём явно.
    msg.answer = AsyncMock()
    msg.edit_text = AsyncMock()
    msg.edit_reply_markup = AsyncMock()
    msg.bot = AsyncMock()
    msg.bot.send_chat_action = AsyncMock()
    return msg


def _make_callback(data: str) -> CallbackQuery:
    cb = AsyncMock(spec=CallbackQuery)
    cb.data = data
    cb.message = _make_message()
    cb.answer = AsyncMock()
    return cb


def _make_backend(**kwargs) -> AsyncMock:
    backend = AsyncMock()
    backend.agent_chat = AsyncMock(return_value=kwargs.get("chat", INTERRUPT_RESULT))
    backend.agent_resume = AsyncMock(
        return_value=kwargs.get("resume", DONE_RESULT)
    )
    return backend


def _plain(md: str) -> str:
    """MarkdownV2 → текст без экранирования (для читаемых ассертов)."""
    return md.replace("\\", "")


def _command(args: str):
    """CommandObject как его собирает aiogram-фильтр Command."""
    from aiogram.filters import CommandObject

    return CommandObject(prefix="/", command="agent", args=args or None)


# ── /agent: запуск ───────────────────────────────────────────────────────


def test_new_thread_id_fits_callback_data():
    """thread_id идёт в callback_data — Telegram разрешает не больше 64 байт."""
    thread_id = new_thread_id(123456789012345)
    assert len(f"hil:approve:{thread_id}".encode()) <= 64
    assert thread_id.startswith("tg123456789012345-")


def test_new_thread_id_is_unique_per_run():
    """Повторный /agent не должен попадать в тот же thread чекпоинтера."""
    assert new_thread_id(1) != new_thread_id(1)


@pytest.mark.asyncio
async def test_cmd_agent_without_args_shows_hint():
    msg = _make_message("/agent")
    backend = _make_backend()

    await cmd_agent(msg, command=_command(""), backend=backend)

    backend.agent_chat.assert_not_awaited()
    msg.answer.assert_awaited_once()
    assert msg.answer.call_args.args[0] == HIL_HINT
    # parse_mode=None: в подсказке есть «<задача>», и HTML-парсер Telegram
    # отверг бы сообщение (Unsupported start tag) — регрессия 06.10.
    assert msg.answer.call_args.kwargs["parse_mode"] is None
    assert "<задача>" in HIL_HINT


@pytest.mark.asyncio
async def test_cmd_agent_interrupt_shows_preview_and_buttons():
    """Пауза: в сообщении — черновик из interrupt, на кнопках — thread_id."""
    msg = _make_message()
    backend = _make_backend()

    await cmd_agent(
        msg, command=_command("отправь уведомление"), backend=backend
    )

    backend.agent_chat.assert_awaited_once()
    sent_task, thread_id, initiator_chat_id = backend.agent_chat.call_args.args
    # Бот передаёт задачу как есть: служебную подсказку о чате-инициаторе
    # добавляет backend, поэтому в тексте её быть не должно (иначе она
    # задвоится и снова начнёт объявлять получателем чат инициатора).
    assert sent_task == "отправь уведомление"
    assert "chat_id" not in sent_task
    # ...а сам chat_id уходит отдельным полем: по нему backend решает, кому
    # можно отправлять, и подставляет его в подсказку для «отправь мне»
    assert initiator_chat_id == 12345

    text = _plain(msg.answer.call_args.args[0])
    assert "нужно подтверждение" in text
    assert "Отчёт по тарифам готов" in text
    assert "search_knowledge_base" in text

    markup = msg.answer.call_args.kwargs["reply_markup"]
    callbacks = [b.callback_data for row in markup.inline_keyboard for b in row]
    assert callbacks == [
        f"{HIL_CB_PREFIX}:approve:{thread_id}",
        f"{HIL_CB_PREFIX}:reject:{thread_id}",
    ]
    assert msg.answer.call_args.kwargs["parse_mode"] is ParseMode.MARKDOWN_V2


@pytest.mark.asyncio
async def test_cmd_agent_done_shows_answer_without_buttons():
    msg = _make_message()
    backend = _make_backend(chat=DONE_RESULT)

    await cmd_agent(msg, command=_command("сколько времени?"), backend=backend)

    text = _plain(msg.answer.call_args.args[0])
    assert "Сообщение отправлено клиенту." in text
    assert msg.answer.call_args.kwargs.get("reply_markup") is None


@pytest.mark.asyncio
async def test_cmd_agent_falls_back_to_plain_text():
    """MarkdownV2 не распарсился — ответ всё равно должен дойти."""
    msg = _make_message()
    backend = _make_backend(chat=DONE_RESULT)
    msg.answer = AsyncMock(
        side_effect=[
            TelegramBadRequest(method=None, message="can't parse entities"),
            None,
        ]
    )

    await cmd_agent(msg, command=_command("задача"), backend=backend)

    assert msg.answer.await_count == 2
    assert msg.answer.call_args.kwargs["parse_mode"] is None


@pytest.mark.asyncio
async def test_cmd_agent_backend_error_is_reported():
    msg = _make_message()
    backend = _make_backend()
    backend.agent_chat = AsyncMock(side_effect=httpx.ConnectError("down"))

    await cmd_agent(msg, command=_command("задача"), backend=backend)

    assert "Сервис недоступен" in msg.answer.call_args.args[0]


# ── callback: подтверждение / отклонение ─────────────────────────────────


@pytest.mark.asyncio
async def test_approve_resumes_with_true():
    cb = _make_callback(f"{HIL_CB_PREFIX}:approve:tg1-abcd1234")
    backend = _make_backend()

    await on_hil_decision(cb, backend=backend)

    backend.agent_resume.assert_awaited_once_with("tg1-abcd1234", True, 12345)
    text = cb.message.edit_text.call_args.args[0]
    assert text.startswith("✅ Подтверждено")
    assert "сообщение отправлено" in text
    # пока агент думает — в сообщении индикатор, а не старый вопрос
    pending = cb.message.edit_text.call_args_list[0].args[0]
    assert "выполняет" in pending
    # кнопки сняты до resume — от повторного клика
    cb.message.edit_reply_markup.assert_awaited_once()


@pytest.mark.asyncio
async def test_reject_resumes_with_false():
    cb = _make_callback(f"{HIL_CB_PREFIX}:reject:tg1-abcd1234")
    cancelled = {
        **DONE_RESULT,
        "answer": "Отправка отменена.",
        "tool_results": [
            {
                "name": "send_telegram_message",
                "args": {"chat_id": "12345"},
                "result": "отправка отменена пользователем",
            }
        ],
    }
    backend = _make_backend(resume=cancelled)

    await on_hil_decision(cb, backend=backend)

    backend.agent_resume.assert_awaited_once_with("tg1-abcd1234", False, 12345)
    text = cb.message.edit_text.call_args.args[0]
    assert text.startswith("❌ Отменено")
    assert "отправка отменена пользователем" in text


@pytest.mark.asyncio
async def test_stale_approval_is_reported():
    """Повторный клик по старой кнопке: backend вернул 5xx — не падаем."""
    cb = _make_callback(f"{HIL_CB_PREFIX}:approve:tg1-abcd1234")
    backend = _make_backend()
    request = httpx.Request("POST", "http://app:8000/agent/resume")
    backend.agent_resume = AsyncMock(
        side_effect=httpx.HTTPStatusError(
            "500", request=request, response=httpx.Response(500, request=request)
        )
    )

    await on_hil_decision(cb, backend=backend)

    assert "неактуально" in cb.message.edit_text.call_args.args[0]


@pytest.mark.asyncio
async def test_unknown_callback_is_ignored():
    cb = _make_callback(f"{HIL_CB_PREFIX}:delete:tg1-abcd1234")
    backend = _make_backend()

    await on_hil_decision(cb, backend=backend)

    backend.agent_resume.assert_not_awaited()
    cb.answer.assert_awaited_once()


# ── рендеринг ────────────────────────────────────────────────────────────


def test_render_interrupt_clips_long_draft():
    payload = {"type": "approve_send", "preview": {"text": "я" * 5000}}
    text = render_interrupt(payload, [])

    assert "я" * 900 in text
    assert len(text) < 1100


def test_render_interrupt_handles_empty_payload():
    """Пустой interrupt (или другой тип) не должен ронять рендер."""
    text = render_interrupt({}, [])

    assert "нужно подтверждение" in text
    assert "—" in text  # chat_id отсутствует → прочерк


def test_hil_kb_callback_data_within_limit():
    markup = hil_kb("tg" + "9" * 16 + "-abcdef01")
    for row in markup.inline_keyboard:
        for button in row:
            assert len(button.callback_data.encode()) <= 64


def test_render_result_includes_dangerous_tool_outcome():
    text = render_result(DONE_RESULT)

    assert "send_telegram_message" in text
    assert "сообщение отправлено" in text
