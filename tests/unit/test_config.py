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


def test_bot_admin_ids_ignores_inline_comment() -> None:
    """`BOT_ADMIN_IDS=  # комментарий` — комментарий приходит значением (08.10)."""
    settings = BotSettings(bot_token="123:test", bot_admin_ids="# админы через запятую")
    assert settings.bot_admin_ids == []


def test_bot_admin_ids_with_trailing_comment() -> None:
    settings = BotSettings(bot_token="123:test", bot_admin_ids="12345,67890  # админы")
    assert settings.bot_admin_ids == [12345, 67890]


def test_bot_token_comment_is_not_a_token() -> None:
    """Пусто с комментарием — это заглушка, а не «токен» из текста комментария."""
    settings = BotSettings(bot_token="# токен от @BotFather")
    assert settings.bot_token.get_secret_value() == ""


@pytest.mark.asyncio
async def test_httpx_client_builds_with_settings_proxy() -> None:
    """Ровно тот вызов, что в lifespan (app/main.py): клиент обязан создаться."""
    settings = Settings(proxy_url="")
    client = httpx.AsyncClient(proxy=settings.proxy_url)
    await client.aclose()
