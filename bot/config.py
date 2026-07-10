"""
Настройки Telegram-бота.
Бот читает свои переменные из того же .env, что и backend, но через
отдельный класс — чтобы не тащить LLM/DB-зависимости в bot-процесс.
"""

from functools import lru_cache
from typing import Annotated
from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class BotSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    bot_token: SecretStr
    backend_url: str = "http://app:8000"
    bot_admin_ids: Annotated[list[int], NoDecode] = []

    @field_validator("bot_admin_ids", mode="before")
    @classmethod
    def _parse_ids(cls, v):
        if isinstance(v, str):
            return [int(x) for x in v.split(",") if x.strip()]
        return v

@lru_cache
def get_bot_settings() -> BotSettings:
    return BotSettings()
