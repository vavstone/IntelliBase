"""Доменные модели чата.

Чистые Pydantic v2-модели — никаких импортов из SQLAlchemy / FastAPI / aiofiles.
Граница между доменом и инфраструктурой проходит через ORM-границу
(`ChatMessage.model_validate(row, from_attributes=True)`).
"""
from datetime import UTC, datetime
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

Role = Literal["user", "assistant", "system"]

class ChatMessage(BaseModel):
    """Сообщение внутри чата (доменная модель)."""
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "id": "660e8400-e29b-41d4-a716-446655440001",
                    "chat_id": "550e8400-e29b-41d4-a716-446655440000",
                    "role": "user",
                    "content": "Напиши hello world на Python",
                    "tokens": 12,
                    "created_at": "2026-07-04T12:00:05Z",
                }
            ]
        }
    )

    id: UUID = Field(default_factory=uuid4)
    chat_id: UUID
    role: Role
    content: str
    media_refs: dict | None = None
    tokens: int | None = None
    prompt_id: UUID | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

class Chat(BaseModel):
    """Чат — один диалог одного пользователя с ассистентом."""
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "id": "550e8400-e29b-41d4-a716-446655440000",
                    "owner_external_id": "user-123",
                    "interface": "telegram",
                    "provider": "ollama",
                    "model": "qwen2.5:3b",
                    "system_prompt": "Ты полезный ассистент.",
                    "created_at": "2026-07-04T12:00:00Z",
                }
            ]
        }
    )

    id: UUID = Field(default_factory=uuid4)
    owner_external_id: str
    interface: str
    provider: Literal["openai", "ollama", "openrouter"]
    model: str
    system_prompt: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class SystemPrompt(BaseModel):
    """Версионированный системный промпт — кандидат A/B traffic-split.

    Активные кандидаты выбираются репозиторием (active=TRUE и traffic_pct>0),
    конкретный вариант для пользователя — `prompt_selection.choose_by_split`.
    """

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    version: str
    body: str
    traffic_pct: int
