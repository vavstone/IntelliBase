"""
Фабрика общего httpx.AsyncClient для бота.
Один клиент на весь процесс — даёт connection pooling и keep-alive.
Лимиты подобраны под умеренную нагрузку (десятки одновременных диалогов).
"""

import httpx
from bot.config import BotSettings


def build_http_client(settings: BotSettings) -> httpx.AsyncClient:
    """Создаёт httpx.AsyncClient с базовыми timeouts и connection limits.
    Если в настройках задан proxy_url, пробрасывает его.
    """
    return httpx.AsyncClient(
        base_url=settings.backend_url,
        proxy=settings.proxy_url,  # None — без прокси, иначе http://user:pass@host:port
        timeout=httpx.Timeout(
            connect=3.0,
            read=60.0,
            write=10.0,
            pool=5.0,
        ),
        limits=httpx.Limits(
            max_connections=50,
            max_keepalive_connections=20,
            keepalive_expiry=30.0,
        ),
        headers={"X-Client": "telegram-bot/1.0"},
    )
