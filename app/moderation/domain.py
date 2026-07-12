"""Доменные модели модерации."""

from dataclasses import dataclass, field


@dataclass
class ModerationResult:
    """Решение каскада модерации.

    - allowed=True, layer="passed"  — контент прошёл оба слоя
    - allowed=False, layer="regex"  — заматчился локальный blocklist
    - allowed=False, layer="openai" — OpenAI Moderation API flagged

    Спецификация задания 4.4:
    - reasons: человекочитаемые причины блокировки
    - blocked_by: конкретный паттерн/категория, сработавшая при блокировке
    """

    allowed: bool
    categories: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    blocked_by: str = ""
    layer: str = ""
    scores: dict[str, float] = field(default_factory=dict)
