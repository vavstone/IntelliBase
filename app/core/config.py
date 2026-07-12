from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LLM_")

    openai_api_key: SecretStr = SecretStr("sk-test-placeholder")
    openrouter_api_key: SecretStr = SecretStr("sk-test-placeholder")
    ollama_base_url: str = "http://localhost:11434/v1"
    openai_base_url: str = "https://api.openai.com/v1"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    default_provider: Literal["openai", "ollama", "openrouter"] = "ollama"
    default_model: str = "qwen2.5:3b"
    request_timeout: float = 30.0
    max_retries: int = 3


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    app_name: str = "llm-service-asdf"
    debug: bool = False
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 3600
    proxy_url: str | None = None
    llm: LLMSettings = Field(default_factory=LLMSettings)

    database_url: str = "postgresql+asyncpg://chat:pswd@localhost:5433/intellibase"
    chat_repository: Literal["json", "postgres"] = "json"
    chat_storage_dir: Path = Path("./var/chats")
    chat_context_window: int = 10

    # Production ---------------------------------------------------------
    # Service-to-service: backend ↔ bot (общий с bot /notify).
    internal_token: SecretStr = SecretStr("change-me-internal")
    # Базовый URL bot-сервиса (для broadcast и notify-вызовов из backend).
    bot_url: str = "http://bot:9000"

@lru_cache
def get_settings() -> Settings:
    return Settings()
