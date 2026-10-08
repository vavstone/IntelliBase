"""Настройки Telegram-бота.

Бот читает свои переменные из того же .env, что и backend, но через
отдельный класс — чтобы не тащить LLM/DB-зависимости в bot-процесс.
"""

from functools import lru_cache
from typing import Annotated

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


def _strip_inline_comment(value: str) -> str:
    """Отсекает inline-комментарий, попавший в значение.

    `BOT_ADMIN_IDS=  # админы через запятую` читается и docker compose, и
    python-dotenv так, что комментарий становится значением целиком (если перед
    ним самого значения не было). Раньше это роняло бота на старте: `int()` от
    текста комментария — находка проверки чистого клона 08.10.
    """
    return value.split("#", 1)[0].strip()


class BotSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    bot_token: SecretStr
    backend_url: str = "http://app:8000"
    # NoDecode — отключаем штатный JSON-парсинг list[int] для env-переменной,
    # чтобы значение "123,456" попало в наш валидатор как строка.
    bot_admin_ids: Annotated[list[int], NoDecode] = []
    request_timeout: float = 30.0
    # /notify-канал backend → bot. Токен — общий секрет.
    internal_token: SecretStr = SecretStr("change-me")
    bot_api_port: int = 9000
    # X-Admin-Token для вызовов /chats/admin/* из бота.
    admin_token: SecretStr = SecretStr("change-me-admin")
    # Telegram chat_id админ-группы для алертов. None — drain отключён.
    admin_chat_id: int | None = None
    proxy_url: str | None = None

    @field_validator("proxy_url", mode="before")
    @classmethod
    def _empty_proxy_to_none(cls, v):
        """Пустую строку из .env превращаем в None: httpx на `proxy=""` падает."""
        if isinstance(v, str) and v.strip() == "":
            return None
        return v

    @field_validator("bot_token", mode="before")
    @classmethod
    def _clean_bot_token(cls, v):
        """Комментарий рядом с пустым значением не должен выглядеть как токен."""
        return _strip_inline_comment(v) if isinstance(v, str) else v

    @field_validator("bot_admin_ids", mode="before")
    @classmethod
    def _parse_ids(cls, v):
        if isinstance(v, str):
            return [int(x) for x in _strip_inline_comment(v).split(",") if x.strip()]
        return v

    @field_validator("admin_chat_id", mode="before")
    @classmethod
    def _empty_admin_chat_to_none(cls, v):
        """Пустую строку из .env превращаем в None."""
        if v is None:
            return None
        if isinstance(v, str) and v.strip() == "":
            return None
        return int(v)


@lru_cache
def get_bot_settings() -> BotSettings:
    return BotSettings()
