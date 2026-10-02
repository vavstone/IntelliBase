"""
Точка входа: создаёт Bot , Dispatcher , регистрирует роутеры,
запускает polling. Бэкенд-клиент кладётся в dp["backend"] , чтобы middleware
прокидывало его в handlers как параметр.
"""

import asyncio
import logging

import uvicorn
from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.fsm.storage.memory import MemoryStorage

from bot.config import get_bot_settings
from bot.handlers import register_routers
from bot.services.alert_drain import drain_alerts
from bot.services.backend_client import BackendClient
from bot.services.http import build_http_client
from bot.web import build_api, build_disabled_api

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("bot")


async def _serve_disabled(settings, reason: str) -> None:
    """Поднимает только HTTP-заглушку бота (без Telegram polling)."""
    api = build_disabled_api(reason, settings.internal_token.get_secret_value())
    config = uvicorn.Config(
        api,
        host="0.0.0.0",
        port=settings.bot_api_port,
        log_level="info",
    )
    await uvicorn.Server(config).serve()


async def main() -> None:
    settings = get_bot_settings()

    token = settings.bot_token.get_secret_value()
    if not token:
        # Чистый клон: в .env.example токен пустой. Не падаем (иначе контейнер
        # уходит в рестарт-луп и `make up --wait` завершается ошибкой), а
        # поднимаем HTTP-заглушку: /health отвечает, /notify — 503.
        reason = "BOT_TOKEN не задан: бот запущен в режиме заглушки (без polling)"
        log.warning(reason)
        await _serve_disabled(settings, reason)
        return

    # Прокси для Telegram API: если задан PROXY_URL — создаём AiohttpSession
    # с ним. aiohttp-socks (уже в зависимостях) поддерживает HTTP-прокси.
    if settings.proxy_url:
        from aiogram.client.session.aiohttp import AiohttpSession
        bot_session = AiohttpSession(proxy=settings.proxy_url)
        log.info("Using proxy for Telegram API: %s", settings.proxy_url)
        bot = Bot(token=token, 
		session=bot_session,
		default=DefaultBotProperties(parse_mode=ParseMode.HTML)
		)
    else:
        bot = Bot(
        token=token,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )

    dp = Dispatcher(storage=MemoryStorage())

    http = build_http_client(settings)
    backend = BackendClient(
        http, admin_token=settings.admin_token.get_secret_value()
    )
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
        settings.bot_api_port,
        settings.admin_chat_id,
    )
    try:
        await asyncio.gather(
            dp.start_polling(bot),
            server.serve(),
            drain_alerts(bot, backend, settings.admin_chat_id),
        )
    finally:
        await backend.aclose()
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
