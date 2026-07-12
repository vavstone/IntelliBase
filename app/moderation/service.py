"""ModerationService — двухслойный каскад модерации.

Layer 1: локальный regex-blocklist (из moderation_keywords.yaml, моментальная
проверка без сетевых вызовов). Если YAML-файл не найден — fallback на
встроенный DEFAULT_BLOCKLIST.
Layer 2: OpenAI Moderation API (omni-moderation-latest) — категории harassment,
hate, sexual, violence и т.п. Fail-open: при ошибке API пропускаем, чтобы не
ронять UX из-за upstream-проблем.

На любой блок (regex или OpenAI) сервис пишет строку в таблицу `alerts` через
`fire_alert(...)` — этот алерт потом подхватывает bot-drain и кладёт в админ-чат.
Если `session_factory` не передан (нет Postgres) — алерт просто не пишется.
"""

import asyncio
import hashlib
import logging
import re
from pathlib import Path

import structlog
import yaml
from openai import AsyncOpenAI

from app.moderation.domain import ModerationResult
from app.services.alerter import fire_alert

log = logging.getLogger(__name__)
logger = structlog.get_logger(__name__)

# Короткий таймаут для Moderation API — чтобы не держать пользователя
MODERATION_API_TIMEOUT = 5.0


DEFAULT_BLOCKLIST: list[dict[str, str]] = [
    {
        "pattern": r"(?i)\b(как\s+(сделать|собрать|изготовить|купить))\s+(бомб[уы]|взрывчатк)",
        "reason": "Запрос инструкций по изготовлению взрывчатых веществ",
        "category": "violence",
    },
]


def _load_blocklist(
    yaml_path: Path | None = None,
) -> list[dict[str, str]]:
    """Загружает блоклист из YAML. При ошибке — fallback на DEFAULT_BLOCKLIST."""
    if yaml_path is None:
        yaml_path = Path(__file__).parent / "moderation_keywords.yaml"
    try:
        if yaml_path.exists():
            with open(yaml_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict) and "patterns" in data:
                return data["patterns"]
    except Exception as exc:
        log.warning("Failed to load moderation_keywords.yaml: %s — using defaults", exc)
    return DEFAULT_BLOCKLIST


def _hash_owner(owner_id: str | None) -> str:
    """sha256[:16] от owner_external_id для PII-безопасного логирования."""
    if not owner_id:
        return "anon"
    return "sha256:" + hashlib.sha256(owner_id.encode()).hexdigest()[:16]


class ModerationService:
    def __init__(
        self,
        llm_client: AsyncOpenAI,
        use_openai_moderation: bool = True,
        blocklist: list[dict[str, str]] | None = None,
        session_factory=None,
    ):
        self.llm = llm_client
        self.use_openai = use_openai_moderation
        patterns = (
            blocklist
            if blocklist is not None
            else _load_blocklist()
        )
        self._patterns = [
            {"regex": re.compile(p["pattern"]), "reason": p.get("reason", ""), "category": p.get("category", "")}
            for p in patterns
        ]
        self.session_factory = session_factory

    async def check_input(
        self, text: str, owner_external_id: str | None = None
    ) -> ModerationResult:
        """Модерация входящего контента пользователя."""
        return await self._check(text, owner_external_id, direction="input")

    async def check_output(
        self, text: str, owner_external_id: str | None = None
    ) -> ModerationResult:
        """Модерация исходящего контента (ответа LLM)."""
        return await self._check(text, owner_external_id, direction="output")

    async def _check(
        self,
        text: str,
        owner_external_id: str | None = None,
        direction: str = "input",
    ) -> ModerationResult:
        if not text:
            return ModerationResult(allowed=True, layer="passed")

        # Layer 1: regex blocklist
        for entry in self._patterns:
            if entry["regex"].search(text):
                result = ModerationResult(
                    allowed=False,
                    categories=[entry["category"]],
                    reasons=[entry["reason"]],
                    blocked_by=entry["regex"].pattern,
                    layer="regex",
                )
                await self._raise_alert(result, owner_external_id)
                self._log_block(result, owner_external_id, direction)
                return result

        # Layer 2: OpenAI Moderation API (fail-open, timeout 5 сек)
        if self.use_openai:
            try:
                resp = await asyncio.wait_for(
                    self.llm.moderations.create(
                        model="omni-moderation-latest",
                        input=text,
                    ),
                    timeout=MODERATION_API_TIMEOUT,
                )
                r = resp.results[0]
                if r.flagged:
                    scores_dict = r.category_scores.model_dump()
                    flagged_cats = [
                        c for c, on in r.categories.model_dump().items() if on
                    ]
                    # Формируем человекочитаемые причины по сработавшим категориям
                    reasons = [
                        f"{cat} (score={scores_dict.get(cat, 0):.3f})"
                        for cat in flagged_cats
                    ]
                    result = ModerationResult(
                        allowed=False,
                        categories=flagged_cats,
                        reasons=reasons,
                        blocked_by="openai_moderation_api",
                        layer="openai",
                        scores=scores_dict,
                    )
                    await self._raise_alert(result, owner_external_id)
                    self._log_block(result, owner_external_id, direction)
                    return result
            except Exception as exc:
                log.warning("moderation API failed: %s — fail-open", exc)

        return ModerationResult(allowed=True, layer="passed")

    def _log_block(
        self,
        result: ModerationResult,
        owner_external_id: str | None,
        direction: str,
    ) -> None:
        """Структурированный лог блокировки через structlog."""
        logger.info(
            "moderation_block",
            direction=direction,
            allowed=result.allowed,
            categories=result.categories,
            reasons=result.reasons,
            blocked_by=result.blocked_by,
            layer=result.layer,
            user_hash=_hash_owner(owner_external_id),
        )

    async def _raise_alert(
        self, result: ModerationResult, owner_external_id: str | None
    ) -> None:
        if self.session_factory is None:
            return
        try:
            await fire_alert(
                self.session_factory,
                kind="moderation_block",
                payload={
                    "layer": result.layer,
                    "categories": result.categories,
                    "reasons": result.reasons,
                    "blocked_by": result.blocked_by,
                    "owner_external_id": owner_external_id,
                },
            )
        except Exception as exc:
            log.warning("fire_alert(moderation_block) failed: %s", exc)
