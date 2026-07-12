"""
Фабрика общего httpx.AsyncClient для бота.
Один клиент на весь процесс — даёт connection pooling и keep-alive.
Лимиты подобраны под умеренную нагрузку (десятки одновременных диалогов).
"""

import httpx
from bot.config import BotSettings


def build_http_client(settings: BotSettings) -> httpx.AsyncClient:
    """Создаёт httpx.AsyncClient с базовыми timeouts и connection limits.
    Прокси используется только для внешних URL (не localhost/127.0.0.1),
    чтобы внешний прокси-сервер не пытался достучаться до локального бэкенда.
    """
    # Прокси включаем только если backend НЕ на localhost
    proxy = settings.proxy_url
    if proxy and settings.backend_url:
        from urllib.parse import urlparse
        host = urlparse(settings.backend_url).hostname or ""
        if host in ("localhost", "127.0.0.1", "0.0.0.0"):
            proxy = None  # localhost — ходим напрямую

    return httpx.AsyncClient(
        base_url=settings.backend_url,
        proxy=proxy,
        timeout=httpx.Timeout(
            connect=3.0,
            read=settings.request_timeout,
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
