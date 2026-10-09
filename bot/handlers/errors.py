"""Глобальный обработчик непойманных исключений (dp.errors).

Раньше падение хендлера (например, бэкенд недоступен в первом вызове вне
try/except) оседало только в логе aiogram — пользователь не получал ничего
и продолжал ждать ответа. Обработчик логирует ошибку с трейсбеком и честно
сообщает пользователю «что-то пошло не так» (или тостом на кнопке).
"""

import logging

from aiogram.types import ErrorEvent

log = logging.getLogger(__name__)

USER_MESSAGE = (
    "Что-то пошло не так на нашей стороне. "
    "Попробуйте, пожалуйста, ещё раз через минуту."
)


async def on_unhandled_error(event: ErrorEvent) -> bool:
    """Последний рубеж обработки ошибок. True — ошибка «разобрана».

    True гасит исключение в polling-цикле: без этого aiogram продолжает
    логировать трейсбек, но мы уже ответили пользователю.
    """
    log.error(
        "Необработанная ошибка в хендлере: %s", event.exception,
        exc_info=event.exception,
    )
    update = event.update
    try:
        message = getattr(update, "message", None)
        callback = getattr(update, "callback_query", None)
        if message is not None:
            await message.answer(USER_MESSAGE)
        elif callback is not None:
            await callback.answer(USER_MESSAGE, show_alert=True)
    except Exception:  # noqa: BLE001 — сбой ответа не должен ломать polling
        log.warning("Не удалось сообщить пользователю об ошибке", exc_info=True)
    return True
