from typing import Literal
from pydantic import BaseModel, ConfigDict


class ModelInfo(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "id": "qwen2.5:3b",
                    "provider": "ollama",
                    "input_per_1m": 0.0,
                    "output_per_1m": 0.0,
                    "context_window": 32768,
                }
            ]
        }
    )

    id: str
    provider: Literal["openai", "ollama", "openrouter", "deepseek"] = "ollama"
    input_per_1m: float = 0.0
    output_per_1m: float = 0.0
    context_window: int | None = None

CATALOG: dict[str, ModelInfo] = {
    # -- Ollama --
    "qwen2.5:3b": ModelInfo(
        id="qwen2.5:3b",
        provider="ollama",
        input_per_1m=0.0,
        output_per_1m=0.0,
        context_window=32_768,
    ),
    # -- OpenAI --
    "gpt-4.1-nano": ModelInfo(
        id="gpt-4.1-nano",
        provider="openai",
        input_per_1m=0.10,
        output_per_1m=0.40,
        context_window=1_047_576,
    ),
    "gpt-4o-mini": ModelInfo(
        id="gpt-4o-mini",
        provider="openai",
        input_per_1m=0.15,
        output_per_1m=0.60,
        context_window=128_000,
    ),
    "gpt-4o": ModelInfo(
        id="gpt-4o",
        provider="openai",
        input_per_1m=2.50,
        output_per_1m=10.00,
        context_window=128_000,
    ),
    # -- DeepSeek (OpenAI-совместимый эндпоинт, работает без VPN) --
    # Цены — peak-тариф за 1M токенов; на входе берём cache miss (верхняя
    # граница: при попадании в кэш вход дешевле в 50 раз). Контекст — 1M
    # токенов, версия deepseek-v4-flash = DeepSeek-V4.1-Flash (в ответах API
    # модель называется deepseek-flash).
    "deepseek-v4-flash": ModelInfo(
        id="deepseek-v4-flash",
        provider="deepseek",
        input_per_1m=0.30,
        output_per_1m=1.20,
        context_window=1_000_000,
    ),
    "deepseek-v4-pro": ModelInfo(
        id="deepseek-v4-pro",
        provider="deepseek",
        input_per_1m=1.32,
        output_per_1m=3.96,
        context_window=1_000_000,
    ),
}