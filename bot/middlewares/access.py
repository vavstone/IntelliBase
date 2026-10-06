"""Гейт доступа: бот отвечает только разрешённым пользователям.

Корпоративный бот закрыт: список тех, кому он отвечает, лежит в БД на бэкенде
(таблица `bot_users`, админ-API `/chats/admin/bot-users`, плюс bootstrap
`BOT_ALLOWED_CHAT_IDS`). Middleware вешается на все апдейты (`dp.update`), поэтому
посторонний не доходит ни до одного обработчика — ни `/ask`, ни `/agent`.

Ответ на отказ содержит id пользователя: администратору его нужно передать, чтобы
выдать доступ, а сам пользователь иначе свой id не узнает.

Результат проверки кэшируется на `CACHE_TTL_SECONDS`: без кэша каждый апдейт
(включая нажатия кнопок) давал бы лишний HTTP-запрос. Отзыв доступа начинает
действовать в пределах TTL — для админ-операции это приемлемо.

Недоступный бэкенд — НЕ «доступ не выдан»: пользователь видит понятное
«сервис недоступен», а не ложный отказ в доступе.
"""

import logging
import time
from typing import Any, Awaitable, Callable

from aiogram import BaseMiddleware
from aiogram.types import CallbackQuery, Message, TelegramObject, Update

log = logging.getLogger(__name__)

CACHE_TTL_SECONDS = 60

DENIED_TEXT = (
    "🔒 Доступ не выдан.\n\n"
    "Это корпоративный бот: он отвечает только пользователям из списка доступа.\n"
    f"Ваш ID: {{chat_id}} — передайте его администратору."
)
UNAVAILABLE_TEXT = "⚠️ Сервис недоступен, попробуйте позже."


class AccessMiddleware(BaseMiddleware):
    """Пропускает апдейт дальше, только если у пользователя есть доступ."""

    def __init__(self, backend: Any, cache_ttl: int = CACHE_TTL_SECONDS) -> None:
        self._backend = backend
        self._cache_ttl = cache_ttl
        # chat_id → (allowed, момент проверки). Хранится в памяти процесса:
        # после рестарта бота доступ перепроверяется.
        self._cache: dict[str, tuple[bool, float]] = {}

    async def _is_allowed(self, chat_id: int) -> bool | None:
        """True/False — вердикт бэкенда, None — сервис недоступен."""
        key = str(chat_id)
        cached = self._cache.get(key)
        now = time.monotonic()
        if cached is not None and now - cached[1] < self._cache_ttl:
            return cached[0]
        try:
            allowed = await self._backend.check_access(chat_id)
        except Exception as exc:  # noqa: BLE001 — сеть/5xx: это не «отказ в доступе»
            log.warning("Проверка доступа недоступна (chat_id=%s): %s", chat_id, exc)
            return None
        self._cache[key] = (allowed, now)
        if not allowed:
            log.info("Отказано в доступе: chat_id=%s не в списке", chat_id)
        return allowed

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        chat_id = self._chat_id(event)
        if chat_id is None:
            return await handler(event, data)

        allowed = await self._is_allowed(chat_id)
        if allowed is None:
            await self._reply(event, UNAVAILABLE_TEXT)
            return None
        if not allowed:
            await self._reply(event, DENIED_TEXT.format(chat_id=chat_id))
            return None
        return await handler(event, data)

    @staticmethod
    def _chat_id(event: TelegramObject) -> int | None:
        """chat_id владельца апдейта (у callback — из сообщения с кнопкой)."""
        if isinstance(event, Update):
            if event.message is not None:
                return event.message.chat.id
            if event.callback_query is not None:
                if event.callback_query.message is not None:
                    return event.callback_query.message.chat.id
                return event.callback_query.from_user.id
            if event.edited_message is not None:
                return event.edited_message.chat.id
            return None
        if isinstance(event, Message):
            return event.chat.id
        if isinstance(event, CallbackQuery):
            if event.message is not None:
                return event.message.chat.id
            return event.from_user.id
        return None

    @staticmethod
    async def _reply(event: TelegramObject, text: str) -> None:
        """Ответить на отказ (у callback — всплывающим уведомлением)."""
        try:
            if isinstance(event, Update):
                if event.callback_query is not None:
                    await event.callback_query.answer(text[:200], show_alert=True)
                elif event.message is not None:
                    await event.message.answer(text, parse_mode=None)
            elif isinstance(event, Message):
                await event.answer(text, parse_mode=None)
            elif isinstance(event, CallbackQuery):
                await event.answer(text[:200], show_alert=True)
        except Exception as exc:  # noqa: BLE001 — отказ не должен ронять polling
            log.warning("Не удалось ответить на отказ в доступе: %s", exc)
