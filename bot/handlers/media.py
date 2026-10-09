"""Обработчики медиа: фото, голос, аудио, документы (PDF/DOCX).

Бот скачивает файл через Telegram Bot API в bytes и отправляет его как
multipart-часть в backend через `BackendClient.send_message`. Бэкенд сам
конвертирует медиа в content-part для chat.completions (см. app/chat/media.py).
"""

import asyncio
import logging
from io import BytesIO

from aiogram import F, Router
from aiogram.exceptions import TelegramNetworkError
from aiogram.types import Message

from bot.services.backend_client import BackendClient
from bot.services.error_handling import handle_backend_error
from bot.services.streaming import stream_to_chat
from bot.services.typing import typing_until

router = Router(name="media")
log = logging.getLogger(__name__)


MAX_PHOTO_BYTES = 2 * 1024 * 1024  # 2 МБ
MAX_DOC_BYTES = 10 * 1024 * 1024  # 10 МБ
ALLOWED_DOC_EXT = (".pdf", ".docx")


def _pick_photo_size(photos):
    """Выбирает самый большой размер фото, что меньше MAX_PHOTO_BYTES."""
    sorted_photos = sorted(photos, key=lambda p: p.file_size or 0, reverse=True)
    for p in sorted_photos:
        if (p.file_size or 0) <= MAX_PHOTO_BYTES:
            return p
    return sorted_photos[-1]  # самый маленький, если все больше лимита


async def _download_to_bytes(bot, file_id: str, *, attempts: int = 2) -> bytes:
    """Скачивает файл из Telegram; на сетевом сбое — один повтор.

    Наблюдали в живом прогоне: `TelegramNetworkError: ServerDisconnectedError`
    на скачивании документа (одиночный обрыв сети/прокси). Без повтора такой
    сбой превращался в «Что-то пошло не так» от глобального обработчика.
    """
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            f = await bot.get_file(file_id)
            buf = BytesIO()
            await bot.download_file(f.file_path, destination=buf)
            return buf.getvalue()
        except TelegramNetworkError as exc:
            last_exc = exc
            log.warning(
                "Сбой скачивания из Telegram (попытка %d/%d): %s",
                attempt + 1, attempts, exc,
            )
            if attempt + 1 < attempts:
                await asyncio.sleep(1.0)
    assert last_exc is not None  # цикл выше всегда либо вернёт данные, либо запишет ошибку
    raise last_exc


async def _safe_download(message: Message, file_id: str) -> bytes | None:
    """Скачивание с человекочитаемым ответом при повторной неудаче."""
    try:
        return await _download_to_bytes(message.bot, file_id)
    except TelegramNetworkError:
        log.warning("Файл не скачан после повтора", exc_info=True)
        await message.answer(
            "Не удалось скачать файл из Telegram — попробуйте, пожалуйста, ещё раз."
        )
        return None


async def _send_media(
    message: Message,
    backend: BackendClient,
    data: bytes,
    mime: str,
    content: str = ""
) -> None:
    chat_id = await backend.get_or_create_chat(
        owner_external_id=str(message.chat.id), interface="telegram",
    )
    stop = asyncio.Event()
    typing_task = asyncio.create_task(
        typing_until(message.bot, message.chat.id, stop)
    )
    try:
        events = backend.send_message(
            chat_id=chat_id, content=content,
            media=data, mime=mime,
            owner_external_id=str(message.chat.id),
        )
        await stream_to_chat(message, events, chat_id=chat_id)
    except Exception as exc:
        await handle_backend_error(message, exc)
    finally:
        stop.set()
        await typing_task


@router.message(F.photo)
async def on_photo(message: Message, backend: BackendClient) -> None:
    photo = _pick_photo_size(message.photo)
    data = await _safe_download(message, photo.file_id)
    if data is None:
        return
    await _send_media(
        message, backend, data, mime="image/jpeg",
        content=message.caption or "Опиши изображение"
    )


@router.message(F.voice)
async def on_voice(message: Message, backend: BackendClient) -> None:
    data = await _safe_download(message, message.voice.file_id)
    if data is None:
        return
    await _send_media(
        message, backend, data, mime="audio/ogg",
        content=message.caption or ""
    )


@router.message(F.audio)
async def on_audio(message: Message, backend: BackendClient) -> None:
    data = await _safe_download(message, message.audio.file_id)
    if data is None:
        return
    mime = message.audio.mime_type or "audio/mpeg"
    await _send_media(
        message, backend, data, mime=mime,
        content=message.caption or ""
    )


@router.message(F.document)
async def on_document(message: Message, backend: BackendClient) -> None:
    doc = message.document
    fname = (doc.file_name or "").lower()
    if not fname.endswith(ALLOWED_DOC_EXT):
        await message.answer(
            f"Поддерживаются только {', '.join(ALLOWED_DOC_EXT)}.",
        )
        return
    if (doc.file_size or 0) > MAX_DOC_BYTES:
        await message.answer(
            f"Файл слишком большой (>{MAX_DOC_BYTES // 1024 // 1024} МБ)."
        )
        return
    data = await _safe_download(message, doc.file_id)
    if data is None:
        return
    mime = doc.mime_type or "application/octet-stream"
    await _send_media(
        message, backend, data, mime=mime,
        content=message.caption or ""
    )
