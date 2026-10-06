"""Проверка доступа к боту: `GET /access/{chat_id}`.

Бот не ходит в БД (в его процессе нет ни SQLAlchemy, ни настроек бэкенда) —
он спрашивает бэкенд. Ответ используется гейтом на входе: пользователь, которого
нет в списке `bot_users` и нет в `BOT_ALLOWED_CHAT_IDS`, до обработчиков не
доходит (см. `bot/middlewares/access.py`).

Эндпоинт закрыт `X-Internal-Token`: это оракул доступа (по нему можно перебором
узнать, какие chat_id разрешены), наружу его отдавать незачем.
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel

from app.core.config import get_settings
from app.deps.providers import SessionFactoryDep
from app.services.bot_users import is_allowed

log = logging.getLogger(__name__)

router = APIRouter(prefix="/access", tags=["access"])


class AccessOut(BaseModel):
    chat_id: str
    allowed: bool


async def require_internal_token(
    x_internal_token: Annotated[str | None, Header(alias="X-Internal-Token")] = None,
) -> None:
    expected = get_settings().internal_token.get_secret_value()
    if not x_internal_token or x_internal_token != expected:
        raise HTTPException(status_code=403, detail="forbidden")


@router.get(
    "/{chat_id}",
    response_model=AccessOut,
    dependencies=[Depends(require_internal_token)],
)
async def check_access(chat_id: str, session_factory: SessionFactoryDep) -> AccessOut:
    if not chat_id.strip():
        raise HTTPException(status_code=422, detail="chat_id обязателен")
    settings = get_settings()
    allowed = await is_allowed(
        session_factory, chat_id, settings.bot_allowed_chat_ids
    )
    log.info("Проверка доступа: chat_id=%s → %s", chat_id, allowed)
    return AccessOut(chat_id=chat_id, allowed=allowed)
