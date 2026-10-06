"""HTTP-API бота — обратный канал для push'ей от backend.

Один endpoint `POST /notify`, защищён общим секретом `X-Internal-Token`.
Backend (или внутренний admin-flow) может попросить бота отправить
сообщение конкретному пользователю Telegram.
"""

import logging

from aiogram import Bot
from aiogram.exceptions import TelegramAPIError, TelegramForbiddenError
from aiogram.types import BufferedInputFile
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from pydantic import BaseModel

log = logging.getLogger(__name__)

# Лимит Telegram Bot API на документ, отправляемый ботом.
MAX_DOCUMENT_BYTES = 50 * 1024 * 1024


class NotifyRequest(BaseModel):
    chat_id: int
    text: str


def build_api(bot: Bot, internal_token: str) -> FastAPI:
    """Строит FastAPI-приложение, шлющее сообщения через переданный Bot."""
    api = FastAPI(title="bot-notify-api")

    @api.post("/notify")
    async def notify(
        req: NotifyRequest,
        x_internal_token: str = Header(...),
    ) -> dict:
        if x_internal_token != internal_token:
            raise HTTPException(status_code=401, detail="invalid token")
        try:
            await bot.send_message(chat_id=req.chat_id, text=req.text)
        except TelegramForbiddenError:
            raise HTTPException(
                status_code=400, detail="bot blocked by user or chat not found"
            )
        except TelegramAPIError as exc:
            log.warning("/notify failed for chat_id=%s: %s", req.chat_id, exc)
            raise HTTPException(
                status_code=502, detail=f"telegram api error: {exc}"
            )
        return {"ok": True}

    @api.post("/notify/document")
    async def notify_document(
        x_internal_token: str = Header(...),
        chat_id: int = Form(...),
        caption: str = Form(""),
        file: UploadFile = File(...),
    ) -> dict:
        """Отправка документа вложением.

        Байты приходят от app (корпус смонтирован туда, а в контейнер бота — нет),
        поэтому файл передаётся телом запроса, а не путём.
        """
        if x_internal_token != internal_token:
            raise HTTPException(status_code=401, detail="invalid token")
        payload = await file.read()
        if len(payload) > MAX_DOCUMENT_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"document too large: {len(payload)} bytes",
            )
        document = BufferedInputFile(
            payload, filename=file.filename or "document"
        )
        try:
            await bot.send_document(
                chat_id=chat_id, document=document, caption=caption or None
            )
        except TelegramForbiddenError:
            raise HTTPException(
                status_code=400, detail="bot blocked by user or chat not found"
            )
        except TelegramAPIError as exc:
            log.warning(
                "/notify/document failed for chat_id=%s (%s): %s",
                chat_id,
                file.filename,
                exc,
            )
            raise HTTPException(
                status_code=502, detail=f"telegram api error: {exc}"
            )
        return {"ok": True, "file": file.filename, "size": len(payload)}

    @api.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    return api


def build_disabled_api(reason: str, internal_token: str) -> FastAPI:
    """HTTP-заглушка для запуска без BOT_TOKEN.

    Контейнер бота остаётся работоспособным для compose (`/health` отвечает),
    но сообщения не принимает и не отправляет: `/notify` возвращает 503.
    Нужна, чтобы на чистом клоне (в `.env` нет токена) `make up --wait`
    не падал из-за вечно перезапускающегося контейнера.
    """
    api = FastAPI(title="bot-notify-api (disabled)")

    @api.get("/health")
    async def health() -> dict:
        return {"status": "ok", "bot": "disabled", "reason": reason}

    @api.post("/notify")
    async def notify(
        req: NotifyRequest,
        x_internal_token: str = Header(...),
    ) -> dict:
        if x_internal_token != internal_token:
            raise HTTPException(status_code=401, detail="invalid token")
        raise HTTPException(status_code=503, detail=reason)

    @api.post("/notify/document")
    async def notify_document(
        x_internal_token: str = Header(...),
        chat_id: int = Form(...),
        caption: str = Form(""),
        file: UploadFile = File(...),
    ) -> dict:
        if x_internal_token != internal_token:
            raise HTTPException(status_code=401, detail="invalid token")
        raise HTTPException(status_code=503, detail=reason)

    return api