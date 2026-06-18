from typing import Literal
from pydantic import BaseModel

class ModelInfo(BaseModel):
    id: str
    provider: Literal["openai", "ollama", "anthropic"] = "openai"
    input_per_1m: float = 0.0
    output_per_1m: float = 0.0
    context_window: int | None = None

CATALOG: dict[str, ModelInfo] = {
    "gpt-4o-mini": ModelInfo(
        id="gpt-4o-mini",
        provider="openai",
        input_per_1m=0.15,
        output_per_1m=0.60,
        context_window=128_000,
    ),
    "gpt-4.1-nano": ModelInfo(
        id="gpt-4.1-nano",
        provider="openai",
        input_per_1m=0.10,
        output_per_1m=0.40,
        context_window=1_047_576 ,
    ),
    "gpt-4o": ModelInfo(
        id="gpt-4o",
        provider="openai",
        input_per_1m=2.50,
        output_per_1m=10.00,
        context_window=128_000,
    ),
}