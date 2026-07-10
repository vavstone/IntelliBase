"""
Точка входа: создаёт Bot , Dispatcher , регистрирует роутеры,
запускает polling. Бэкенд-клиент кладётся в dp["backend"] , чтобы middleware
прокидывало его в handlers как параметр.
"""

import asyncio
import logging

from aiogram import Bot, Dispatcher
from aiogram.fsm.storage.memory import MemoryStorage
from bot.config import get_bot_settings
from bot.handlers import register_routers
from bot.services.backend_client import BackendClient
from bot.services.http import build_http_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("bot")


async def main() -> None:
    settings = get_bot_settings()
    bot = Bot(
        token=settings.bot_token.get_secret_value()
    )
    dp = Dispatcher(storage=MemoryStorage())
    http = build_http_client(settings)
    backend = BackendClient(http)
    dp["backend"] = backend
    register_routers(dp)

    try:
        await dp.start_polling(bot=bot)
    finally:
        await backend.aclose()
        await bot.close()


if __name__ == "__main__":
    asyncio.run(main())
