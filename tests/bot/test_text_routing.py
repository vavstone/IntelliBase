"""Автомаршрут свободного текста: просьба отправить → агент, вопрос → чат.

Обычный чат отвечает только по базе знаний и отправить ничего не может, поэтому
«отправь документ Крутикову» без команды `/agent` раньше получал ответ
«сведений о такой функции нет». Здесь проверяется, что такие сообщения уходят
агенту, а вопросы — по-прежнему в чат.
"""

from unittest.mock import AsyncMock, patch

import pytest
from aiogram.types import Chat, Message, User

from bot.handlers.agent import looks_like_action
from bot.handlers.text import on_text


def _make_message(text: str, chat_id: int = 111) -> Message:
    msg = AsyncMock(spec=Message)
    msg.chat = Chat(id=chat_id, type="private")
    msg.from_user = User(id=chat_id, is_bot=False, first_name="Test")
    msg.text = text
    # answer/bot в aiogram — sync-атрибуты, под spec=Message они становятся
    # MagicMock, поэтому задаём явно (как в tests/bot/test_agent_hil.py).
    msg.answer = AsyncMock()
    msg.bot = AsyncMock()
    msg.bot.send_chat_action = AsyncMock()
    return msg


def _make_state():
    state = AsyncMock()
    state.get_state = AsyncMock(return_value=None)  # не в FSM-сценарии
    return state


def _make_backend() -> AsyncMock:
    backend = AsyncMock()
    backend.agent_chat = AsyncMock(return_value={"status": "done", "answer": "ок"})
    backend.get_or_create_chat = AsyncMock(return_value="chat-uuid")
    backend.send_message = AsyncMock(return_value=iter([]))
    return backend


@pytest.mark.parametrize(
    "text",
    [
        "отправь ПС Малахит. Технические требования. Версия 1.9 мне и Крутикову",
        "Отправь мне отчёт по тарифам",
        "перешли документ коллеге",
        "скинь выжимку по Малахиту",
        "разошли всем",
    ],
)
def test_action_requests_are_detected(text: str) -> None:
    assert looks_like_action(text)


@pytest.mark.parametrize(
    "text",
    [
        "какие фт есть по нашим ПС в базе?",
        "что сказано про тарифы в регламенте?",
        "расскажи про ПС Малахит",
        "какие требования к хранению документов?",
    ],
)
def test_questions_are_not_actions(text: str) -> None:
    assert not looks_like_action(text)


@pytest.mark.asyncio
async def test_action_request_goes_to_agent() -> None:
    message = _make_message("отправь отчёт мне и Крутикову")
    backend = _make_backend()

    await on_text(message, backend=backend, state=_make_state())

    backend.agent_chat.assert_awaited_once()
    task, thread_id, chat_id = backend.agent_chat.call_args.args
    assert task == "отправь отчёт мне и Крутикову"
    assert thread_id.startswith("tg111-")   # новый thread на запуск
    assert chat_id == 111
    backend.send_message.assert_not_awaited()  # в обычный чат не пошли
    # Ответ помечен: видно, что это агент, а не обычный ответ чата.
    assert "Обработано агентом" in message.answer.call_args.args[0]


@pytest.mark.asyncio
async def test_question_goes_to_chat() -> None:
    message = _make_message("какие фт есть по нашим ПС в базе?")
    backend = _make_backend()

    with patch("bot.handlers.text.stream_to_chat", AsyncMock()) as stream:
        await on_text(message, backend=backend, state=_make_state())

    backend.get_or_create_chat.assert_awaited_once()
    # send_message — async-генератор, поэтому called, а не awaited
    backend.send_message.assert_called_once()
    stream.assert_awaited_once()
    backend.agent_chat.assert_not_awaited()


@pytest.mark.asyncio
async def test_fsm_state_has_priority() -> None:
    """Внутри сценария /ask текст обрабатывает fsm-роутер, а не агент."""
    message = _make_message("отправь отчёт")
    backend = _make_backend()
    state = AsyncMock()
    state.get_state = AsyncMock(return_value="AskCategory:category")

    await on_text(message, backend=backend, state=state)

    backend.agent_chat.assert_not_awaited()
    backend.send_message.assert_not_awaited()
