from typing import Annotated, Literal, Union
from pydantic import BaseModel, ConfigDict, Field, model_validator

Role = Literal["system", "user", "assistant", "tool"]


class Message(BaseModel):
    role: Role
    content: Annotated[str, Field(min_length=1, max_length=100_000)]


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0

    @classmethod
    def from_openai(cls, u) -> "Usage":
        return cls(
            prompt_tokens=getattr(u, "prompt_tokens", 0),
            completion_tokens=getattr(u, "completion_tokens", 0),
            total_tokens=getattr(u, "total_tokens", 0),
        )


class ChatRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "messages": [
                        {"role": "system", "content": "Ты полезный ассистент."},
                        {"role": "user", "content": "Напиши hello world на Python"},
                    ],
                    "model": "qwen2.5:3b",
                    "provider": "ollama",
                    "temperature": 0.0,
                    "max_tokens": 1024,
                    "stream": False,
                }
            ]
        }
    )

    messages: Annotated[list[Message], Field(min_length=1, max_length=50)]
    model: str = "qwen2.5:3b"
    provider: Literal["openai", "ollama", "openrouter"] = "ollama"
    temperature: Annotated[float, Field(ge=0.0, le=2.0)] = 0.0
    max_tokens: Annotated[int, Field(ge=1, le=16_000)] = 1024
    stream: bool = False
    user_id: str | None = None
    session_id: str | None = None

    @model_validator(mode="after")
    def _first_message_not_assistant(self) -> "ChatRequest":
        if self.messages[0].role == "assistant":
            raise ValueError("Первое сообщение не может быть от assistant")
        return self


class ChatResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "content": "Вот пример hello world на Python:\n\n```python\nprint(\"Hello, World!\")\n```",
                    "model": "qwen2.5:3b",
                    "usage": {
                        "prompt_tokens": 42,
                        "completion_tokens": 18,
                        "total_tokens": 60,
                        "estimated_cost_usd": 0.0,
                    },
                    "finish_reason": "stop",
                    "cached": False,
                    "request_id": "a1b2c3d4e5f6",
                }
            ]
        }
    )

    content: str
    model: str
    usage: Usage
    finish_reason: str | None = None
    cached: bool = False
    request_id: str | None = None

    @classmethod
    def from_openai(cls, raw) -> "ChatResponse":
        choice = raw.choices[0]
        return cls(
            content=choice.message.content or "",
            model=raw.model,
            usage=Usage.from_openai(raw.usage),
            finish_reason=choice.finish_reason,
        )


class ChatDelta(BaseModel):
    content: str | None = None
    usage: Usage | None = None


class OpenAIParams(BaseModel):
    provider: Literal["openai"]
    model: str = "gpt-4.1-nano"
    seed: int | None = None


class OllamaParams(BaseModel):
    provider: Literal["ollama"]
    model: str = "qwen2.5:3b"
    base_url: str = "http://localhost:11434/v1"


ProviderParams = Annotated[
    Union[OpenAIParams, OllamaParams],
    Field(discriminator="provider"),
]


class ChatRequestMulti(BaseModel):
    messages: list[Message]
    params: ProviderParams
    temperature: float = 0.7
    max_tokens: int = 1024