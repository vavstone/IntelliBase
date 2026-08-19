"""Репозиторий каталога категорий знаний (Postgres, таблица kb_categories).

Fail-soft: при отсутствии session_factory `list_categories` возвращает пустой
список — вызывающие роуты сами решают, отдавать ли 503.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select, text

from app.chat.repositories.pg_models import KbCategoryRow
from app.kb.domain import KbCategory


class KbCategoryRepository:
    def __init__(self, session_factory: Any) -> None:
        self.session_factory = session_factory

    async def list_categories(self) -> list[KbCategory]:
        """Категории в порядке добавления (seed идёт по таксономии)."""
        if self.session_factory is None:
            return []
        async with self.session_factory() as s:
            rows = (
                await s.execute(select(KbCategoryRow).order_by(KbCategoryRow.id))
            ).scalars().all()
        return [KbCategory(slug=r.slug, title=r.title) for r in rows]

    async def upsert_category(self, slug: str, title: str) -> KbCategory:
        """Идемпотентно создаёт категорию; существующую не перезаписывает.

        `title` применяется только при первом создании — при повторном вызове
        с уже существующим slug сохраняется исходный title (ON CONFLICT DO
        NOTHING). Возвращает фактическую запись (существующую или новую).
        """
        async with self.session_factory() as s:
            await s.execute(
                text(
                    "INSERT INTO kb_categories (slug, title) "
                    "VALUES (:slug, :title) "
                    "ON CONFLICT (slug) DO NOTHING"
                ),
                {"slug": slug, "title": title},
            )
            await s.commit()
            row = (
                await s.execute(
                    select(KbCategoryRow).where(KbCategoryRow.slug == slug)
                )
            ).scalar_one()
        return KbCategory(slug=row.slug, title=row.title)
