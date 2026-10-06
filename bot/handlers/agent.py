"""Агентный сценарий с человеком в цикле (HIL): `/agent` + inline-подтверждение.

Шаг 4 демо. Опасное действие агента (`send_telegram_message`) не выполняется
само: граф останавливается на `interrupt()`, бот показывает превью черновика и
кнопки «Отправить»/«Отменить», а решение человека уходит в `POST /agent/resume`
(см. `app/services/agent_persistent.py`, `app/routers/agent.py`).

- `/agent <задача>` — один шаг агента в НОВОМ thread'е. Id генерится здесь
  (`tg<chat_id>-<hex8>`), чтобы повторные запуски и демо-прогоны не смешивались
  в чекпоинтере. Получатель сообщения — сам чат с ботом: его chat_id уходит
  в `chat_id` запроса (и подсказкой в задаче, чтобы модель не тратила круг на
  отказ инструмента). Решение о допустимых получателях принимает backend.
- `hil:<approve|reject>:<thread_id>` — callback кнопок, resume того же thread'а.

Текст от LLM прогоняется через `telegramify_markdown` (как в стриминге чата) и
уходит в MarkdownV2; на ошибке парсинга — plain-текст, чтобы ответ дошёл.
"""

import asyncio
import logging
import re
import uuid

import httpx
from aiogram import F, Router
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command, CommandObject
from aiogram.types import CallbackQuery, Message

from bot.keyboards.inline import (
    HIL_APPROVE,
    HIL_CB_PREFIX,
    HIL_VALUES,
    hil_kb,
)
from bot.services.backend_client import BackendClient
from bot.services.error_handling import handle_backend_error
from bot.services.streaming import to_tg_markdown
from bot.services.typing import typing_until

router = Router(name="agent")
log = logging.getLogger(__name__)

# Telegram ограничивает сообщение 4096 символами — режем с запасом.
TELEGRAM_LIMIT = 4000
# Сколько символов черновика показывать в превью (остальное — многоточие).
PREVIEW_LIMIT = 900
# Имя опасного инструмента в `tool_results` (совпадает с app/tools/graph_tools.py).
DANGEROUS_TOOL = "send_telegram_message"
RESULT_LIMIT = 200

HIL_HINT = (
    "🤖 Агентный режим с подтверждением (HIL).\n"
    "\n"
    "Использование: /agent <задача>\n"
    "Например: /agent Подготовь уведомление о готовности отчёта "
    "и отправь его клиенту\n"
    "\n"
    "Что делает агент: ищет данные в базе знаний, формирует сообщение и "
    "просит подтверждение перед отправкой. Без нажатия «Отправить» "
    "сообщение не уходит.\n"
    "\n"
    "Команду можно не писать: если в сообщении есть просьба отправить или "
    "переслать, задача уходит агенту автоматически (в ответе будет пометка "
    "«🤖 Обработано агентом»)."
)

# Просьба совершить действие в свободном тексте: «отправь отчёт Крутикову».
# Нужна, чтобы человеку не приходилось знать про /agent: обычный чат отвечает
# только по базе знаний и отправить ничего не может, а агент умеет и то, и другое.
ACTION_INTENT_RE = re.compile(
    r"\b(отправ\w*|пришл\w*|перешл\w*|скинь\w*|скин\w*|разошл\w*|вышл\w*|"
    r"разосл\w*|отправить)\b",
    re.IGNORECASE,
)


def looks_like_action(text: str) -> bool:
    """Похоже ли сообщение на задачу агенту (а не на вопрос по базе знаний)."""
    return bool(ACTION_INTENT_RE.search(text or ""))


def new_thread_id(chat_id: int) -> str:
    """Новый thread на каждый запуск: прогоны не смешиваются в чекпоинтере.

    Идёт в callback_data кнопок, поэтому держим коротким: Telegram разрешает
    не больше 64 байт (`hil:approve:` + id).
    """
    return f"tg{chat_id}-{uuid.uuid4().hex[:8]}"


def _clip(text: str, limit: int = TELEGRAM_LIMIT) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _tool_lines(tool_results: list[dict], *, with_result: bool) -> list[str]:
    """Строки про инструменты агента. Для опасного инструмента показываем
    результат: «отправка отменена пользователем» / «сообщение отправлено» —
    это и есть свидетельство, что до подтверждения side-effect не случился."""
    if not tool_results:
        return []
    names = [str(r.get("name") or "?") for r in tool_results]
    lines = [f"🔧 Инструменты: {', '.join(names)}"]
    for r in tool_results:
        if not with_result or r.get("name") != DANGEROUS_TOOL:
            continue
        result = str(r.get("result") or "").strip()
        if len(result) > RESULT_LIMIT:
            result = result[:RESULT_LIMIT] + "…"
        lines.append(f"   ↳ {DANGEROUS_TOOL}: {result}")
    return lines


def _format_recipients(preview: dict) -> str:
    """«Иванов Пётр (111222333), Петров Пётр (444555666)» или один chat_id.

    Понимает и старый одиночный формат (`recipient_title` + `chat_id`), и новый
    список (`recipients`) — черновики в чекпоинтере могли остаться от прежней версии.
    """
    items = preview.get("recipients")
    if not items:
        title = str(preview.get("recipient_title") or "").strip()
        chat_id = str(preview.get("chat_id") or "—")
        return f"{title}, chat_id {chat_id}" if title else chat_id
    return ", ".join(
        f"{r.get('title')} ({r.get('chat_id')})" if r.get("title") else str(r.get("chat_id"))
        for r in items
    ) or "—"


def render_interrupt(payload: dict, tool_results: list[dict]) -> str:
    """Превью черновика из HIL-паузы (кнопки вешает вызывающий)."""
    preview = payload.get("preview") or {}
    is_document = preview.get("kind") == "document"
    draft = str(
        preview.get("caption") if is_document else preview.get("text") or ""
    ).strip()
    if len(draft) > PREVIEW_LIMIT:
        draft = draft[:PREVIEW_LIMIT] + "…"
    # Получателей показываем именами (из списка пользователей), chat_id — рядом:
    # подтверждать нужно по имени, иначе человек сверяет цифры, которые ни о чём
    # не говорят и легко перепутать. Получателей может быть несколько
    # («мне и Иванову») — тогда перечисляем всех: подтверждение одно на всех.
    recipient = _format_recipients(preview)
    if is_document:
        head = (
            f"📎 Черновик отправки файла (получатель: {recipient}):"
            f"\n\nФайл: «{preview.get('file_name') or '—'}»"
        )
    else:
        head = f"📨 Черновик сообщения (получатель: {recipient}):"
    blocks = [
        "⏸ Агент остановлен: нужно подтверждение",
        f"{head}\n\n«{draft}»" if draft else head,
    ]
    tools = _tool_lines(tool_results, with_result=True)
    if tools:
        blocks.append("\n".join(tools))
    blocks.append("Отправить сообщение?")
    return "\n\n".join(blocks)


def render_result(result: dict) -> str:
    answer = str(result.get("answer") or "").strip() or "(пустой ответ)"
    lines = ["✅ Агент завершил шаг", "", answer]
    tools = _tool_lines(result.get("tool_results") or [], with_result=True)
    if tools:
        lines += ["", *tools]
    return "\n".join(lines)


async def _answer(
    message: Message, text: str, reply_markup=None
) -> None:
    """Markdown от LLM → MarkdownV2; при ошибке парсинга — plain-текст.

    Ответ агента содержит пользовательский Markdown (`**жирный**`, `> цитата`),
    поэтому переводим его тем же конвертером, что и стриминг чата, а при
    TelegramBadRequest отдаём текст как есть — без разметки, но доставленный.
    """
    body = _clip(text)
    try:
        await message.answer(
            to_tg_markdown(body),
            reply_markup=reply_markup,
            parse_mode=ParseMode.MARKDOWN_V2,
        )
    except TelegramBadRequest as exc:
        log.warning("MarkdownV2 parse failed, fallback to plain: %s", exc)
        await message.answer(body, reply_markup=reply_markup, parse_mode=None)


async def _edit(cb: CallbackQuery, text: str) -> None:
    if cb.message is None:
        return
    body = _clip(text)
    try:
        await cb.message.edit_text(
            to_tg_markdown(body), parse_mode=ParseMode.MARKDOWN_V2
        )
    except TelegramBadRequest as exc:
        log.debug("agent edit_text fallback to plain: %s", exc)
        try:
            await cb.message.edit_text(body, parse_mode=None)
        except TelegramBadRequest as exc2:
            # Сообщение старше 48 ч / уже отредактировано — не критично.
            log.debug("agent edit_text failed: %s", exc2)


async def run_agent_task(
    message: Message,
    task: str,
    backend: BackendClient,
    *,
    auto_routed: bool = False,
) -> None:
    """Один шаг агента: пауза с кнопками либо финальный ответ.

    Используется и командой `/agent`, и автомаршрутом из свободного текста
    (`bot/handlers/text.py`): человек пишет «отправь документ Крутикову» без
    команды, а задача всё равно попадает агенту. `auto_routed=True` помечает
    такой ответ, чтобы было видно, почему он отличается от обычного ответа чата.
    """
    thread_id = new_thread_id(message.chat.id)
    # Задача уходит как есть: служебную подсказку о чате-инициаторе добавляет
    # backend (app/routers/agent.py) — она нужна любому клиенту, не только боту.
    # chat_id передаём отдельным полем: по нему backend решает, кому можно
    # отправлять, и подставляет его в подсказку для «отправь мне».
    stop = asyncio.Event()
    typing_task = asyncio.create_task(
        typing_until(message.bot, message.chat.id, stop)
    )
    try:
        result = await backend.agent_chat(task, thread_id, message.chat.id)
    except Exception as exc:
        await handle_backend_error(message, exc)
        return
    finally:
        stop.set()
        await typing_task

    if result.get("status") == "interrupted":
        await _answer(
            message,
            render_interrupt(
                result.get("interrupt") or {},
                result.get("tool_results") or [],
            ),
            reply_markup=hil_kb(thread_id),
        )
        return
    answer = render_result(result)
    if auto_routed:
        answer = f"🤖 Обработано агентом\n\n{answer}"
    await _answer(message, answer)


@router.message(Command("agent"))
async def cmd_agent(
    message: Message, command: CommandObject, backend: BackendClient
) -> None:
    task = (command.args or "").strip()
    if not task:
        # parse_mode=None: бот по умолчанию парсит HTML, а в подсказке есть
        # «<задача>» — Telegram отверг бы сообщение (Unsupported start tag).
        await message.answer(HIL_HINT, parse_mode=None)
        return
    await run_agent_task(message, task, backend)


@router.callback_query(F.data.startswith(f"{HIL_CB_PREFIX}:"))
async def on_hil_decision(cb: CallbackQuery, backend: BackendClient) -> None:
    parts = (cb.data or "").split(":", 2)
    if len(parts) != 3 or parts[1] not in HIL_VALUES:
        await cb.answer()
        return
    _, decision, thread_id = parts
    approved = decision == HIL_APPROVE

    await cb.answer("Отправляю…" if approved else "Отменяю…")
    if cb.message is not None:
        # Снимаем кнопки до resume: защита от повторного клика, пока
        # бэкенд выполняет граф (resume идемпотентным не является).
        try:
            await cb.message.edit_reply_markup(reply_markup=None)
        except Exception as exc:
            log.debug("clear HIL keyboard failed: %s", exc)
    # Агент может думать десятки секунд (RAG + несколько вызовов LLM) —
    # показываем, что решение принято и работа идёт.
    pending = (
        "✅ Подтверждено — агент выполняет…"
        if approved
        else "❌ Отменено — агент завершает…"
    )
    await _edit(cb, pending)

    # Инициатор = чат, в котором показаны кнопки. Если сообщение недоступно
    # (старое/удалённое), берём id пользователя: в приватном чате это тот же id.
    initiator_chat_id = (
        cb.message.chat.id if cb.message is not None else cb.from_user.id
    )
    try:
        result = await backend.agent_resume(thread_id, approved, initiator_chat_id)
    except httpx.HTTPStatusError as exc:
        log.warning("agent resume failed (%s): %s", thread_id, exc)
        await _edit(cb, "⚠️ Подтверждение уже неактуально — запустите /agent заново.")
        return
    except Exception as exc:
        log.warning("agent resume crashed (%s): %s", thread_id, exc)
        await _edit(cb, "⚠️ Сервис недоступен, попробуйте позже.")
        return

    header = "✅ Подтверждено" if approved else "❌ Отменено"
    await _edit(cb, f"{header}\n\n{render_result(result)}")
