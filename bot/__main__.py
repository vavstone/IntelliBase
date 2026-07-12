"""
Точка входа: создаёт Bot , Dispatcher , регистрирует роутеры,
запускает polling. Бэкенд-клиент кладётся в dp["backend"] , чтобы middleware
прокидывало его в handlers как параметр.
"""

import asyncio
import logging

import uvicorn
from aiogram import Bot, Dispatcher
from aiogram.fsm.storage.memory import MemoryStorage

from bot.config import get_bot_settings
from bot.handlers import register_routers
from bot.services.backend_client import BackendClient
from bot.services.http import build_http_client
from bot.web import build_api

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("bot")


async def main() -> None:
    settings = get_bot_settings()

    token = settings.bot_token.get_secret_value()
    if not token:
        raise ValueError("BOT_TOKEN is empty — задайте токен в .env")

    # Прокси для Telegram API: если задан PROXY_URL — создаём AiohttpSession
    # с ним. aiohttp-socks (уже в зависимостях) поддерживает HTTP-прокси.
    if settings.proxy_url:
        from aiogram.client.session.aiohttp import AiohttpSession
        bot_session = AiohttpSession(proxy=settings.proxy_url)
        log.info("Using proxy for Telegram API: %s", settings.proxy_url)
        bot = Bot(token=token, session=bot_session)
    else:
        bot = Bot(token=token)

    dp = Dispatcher(storage=MemoryStorage())
    http = build_http_client(settings)
    backend = BackendClient(http)
    dp["backend"] = backend
    register_routers(dp)

    api = build_api(bot, settings.internal_token.get_secret_value())
    config = uvicorn.Config(
        api,
        host="0.0.0.0",
        port=settings.bot_api_port,
        log_level="info",
    )
    server = uvicorn.Server(config)

    log.info(
        "Bot starting (backend=%s, notify-port=%s, admin_chat_id=%s)",
        settings.backend_url,
        settings.bot_api_port
    )
    try:
        await asyncio.gather(
            dp.start_polling(bot),
            server.serve()
        )
    finally:
        await backend.aclose()
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
