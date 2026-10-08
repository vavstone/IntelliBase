"""Пустой `PROXY_URL` не должен ронять старт приложения (регрессия 08.10).

`PROXY_URL=` (пустое значение, как в .env.example) pydantic-settings отдавал
как пустую строку, а `httpx.AsyncClient(proxy="")` падает с
`ValueError: Unknown scheme for proxy URL` — lifespan не проходил, контейнер
уходил в unhealthy, `make up` падал на чистом клоне.
"""

import httpx
import pytest

from app.core.config import Settings
from bot.config import BotSettings


@pytest.mark.parametrize("raw", ["", "   "])
def test_app_empty_proxy_is_none(raw: str) -> None:
    assert Settings(proxy_url=raw).proxy_url is None


def test_app_real_proxy_is_kept() -> None:
    url = "http://127.0.0.1:8080"
    assert Settings(proxy_url=url).proxy_url == url


@pytest.mark.parametrize("raw", ["", "   "])
def test_bot_empty_proxy_is_none(raw: str) -> None:
    settings = BotSettings(bot_token="123:test", proxy_url=raw)
    assert settings.proxy_url is None


@pytest.mark.asyncio
async def test_httpx_client_builds_with_settings_proxy() -> None:
    """Ровно тот вызов, что в lifespan (app/main.py): клиент обязан создаться."""
    settings = Settings(proxy_url="")
    client = httpx.AsyncClient(proxy=settings.proxy_url)
    await client.aclose()
