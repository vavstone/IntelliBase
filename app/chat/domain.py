from datetime import datetime, UTC
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