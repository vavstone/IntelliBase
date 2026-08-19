"""Доменные модели каталога категорий знаний (одна категория = одно ПС)."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

# Каноническая форма slug: латиница lowercase + безопасные в путях/URL/callback
# символы. Русский title в slug не допускается — перевод в латиницу вне объёма
# (см. техдолг, п.8): upload принимает slug напрямую.
_SLUG_RE = re.compile(r"[^a-z0-9_-]+")

# Fallback-категория для документов без принадлежности к ПС / пересечений.
DEFAULT_CATEGORY = "raznoe"


def normalize_slug(raw: str | None) -> str:
    """Канонизирует slug и защищает от обхода каталога.

    Нижний регистр, не-латинские/спецсимволы -> '-', обрезка краёв.
    Пустой/невалидный ввод -> DEFAULT_CATEGORY. `../../evil` -> `evil`
    (`.` и `/` не входят в разрешённый алфавит и заменяются).
    """
    if not raw:
        return DEFAULT_CATEGORY
    slug = _SLUG_RE.sub("-", raw.strip().lower()).strip("-_")
    return slug or DEFAULT_CATEGORY


class KbCategory(BaseModel):
    """Категория знаний: slug — ключ, title — русское отображение."""

    slug: str = Field(min_length=1)
    title: str = Field(min_length=1)


class CategoryCreate(BaseModel):
    """Тело POST /categories: slug обязателен, title опционален (= slug)."""

    slug: str = Field(min_length=1, max_length=128)
    title: str | None = None
