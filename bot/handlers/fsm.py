"""
FSM-сценарий /ask: выбор категории (ПС) → текст вопроса → отправка с category.

Категории запрашиваются у backend (GET /categories) и строят inline-меню;
выбранный slug уходит в RAG как отдельное поле `category` (строгая фильтрация
поиска на уровне векторного хранилища), а не текстовым префиксом «Тема: …».
"""

import asyncio
import logging

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, Message

from bot.keyboards.inline import (
    ALL_CATEGORY,
    ALL_CATEGORY_LABEL,
    DEFAULT_CATEGORIES,
    topics_kb,
)
from bot.services.backend_client import BackendClient
from bot.services.error_handling import handle_backend_error
from bot.services.streaming import stream_to_chat
from bot.services.typing import typing_until
from bot.states import AskFlow

router = Router(name="fsm")
log = logging.getLogger(__name__)


async def _fetch_categories(backend: BackendClient) -> list[dict]:
    """Возвращает [{slug, title}] от backend; fallback — seed-категории."""
    try:
        cats = await backend.list_categories()
        if cats:
            return cats
    except Exception as exc:
        log.warning("не удалось получить категории из backend: %s", exc)
    return [{"slug": s, "title": t} for t, s in DEFAULT_CATEGORIES]


@router.message(Command("ask"))
async def cmd_ask(message: Message, state: FSMContext, backend: BackendClient) -> None:
    await state.set_state(AskFlow.waiting_for_topic)
    categories = await _fetch_categories(backend)
    await state.update_data(categories=categories)
    await message.answer(
        "Выберите тему:",
        reply_markup=topics_kb([(c["title"], c["slug"]) for c in categories]),
    )


@router.callback_query(F.data.startswith("topic:"), AskFlow.waiting_for_topic)
async def on_topic_selected(cb: CallbackQuery, state: FSMContext) -> None:
    _, slug = cb.data.split(":", 1)
    if slug == "cancel":
        await state.clear()
        if cb.message is not None:
            await cb.message.edit_text("Отменено.")
        await cb.answer()
        return
    # «Все ПС» — поиск без фильтра по категории (category=None уходит в backend).
    if slug == ALL_CATEGORY:
        await state.update_data(category=None)
        await state.set_state(AskFlow.waiting_for_question)
        if cb.message is not None:
            await cb.message.edit_text(
                f"Тема: {ALL_CATEGORY_LABEL}\nЗадайте ваш вопрос текстом."
            )
        await cb.answer()
        return
    data = await state.get_data()
    categories = data.get("categories") or []
    title = next((c["title"] for c in categories if c["slug"] == slug), slug)
    await state.update_data(category=slug)
    await state.set_state(AskFlow.waiting_for_question)
    if cb.message is not None:
        await cb.message.edit_text(f"Тема: {title}\nЗадайте ваш вопрос текстом.")
    await cb.answer()


@router.message(AskFlow.waiting_for_question, F.text)
async def on_question_received(
    message: Message,
    backend: BackendClient,
    state: FSMContext,
) -> None:
    data = await state.get_data()
    category = data.get("category")

    chat_id = await backend.get_or_create_chat(
        owner_external_id=str(message.chat.id),
        interface="telegram",
    )
    stop = asyncio.Event()
    typing_task = asyncio.create_task(
        typing_until(message.bot, message.chat.id, stop)
    )
    try:
        events = backend.send_message(
            chat_id,
            message.text,
            owner_external_id=str(message.chat.id),
            category=category,
        )
        await stream_to_chat(message, events, chat_id=chat_id)
    except Exception as exc:
        await handle_backend_error(message, exc)
    finally:
        stop.set()
        await typing_task
        await state.clear()
