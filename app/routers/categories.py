"""Каталог категорий знаний (ПС): чтение списка и ручное создание.

`GET /categories` используют бот (меню `/ask`) и upload-UI для выбора темы.
`POST /categories` — запасной путь заводки новых категорий вручную (помимо
upsert при upload). Категории хранятся в Postgres (`kb_categories`); при
недоступной БД — 503 (молча возвращать пустой список нельзя: таксономия
«съедет» у бота).
"""

import logging

from fastapi import APIRouter, HTTPException

from app.deps.providers import SessionFactoryDep
from app.kb.domain import CategoryCreate, KbCategory, normalize_slug
from app.kb.repository import KbCategoryRepository

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/categories", tags=["categories"])


@router.get("", response_model=list[KbCategory], summary="Список категорий (ПС)")
async def list_categories(session_factory: SessionFactoryDep) -> list[KbCategory]:
    if session_factory is None:
        raise HTTPException(status_code=503, detail="categories require postgres")
    return await KbCategoryRepository(session_factory).list_categories()


@router.post(
    "",
    response_model=KbCategory,
    status_code=201,
    summary="Создать категорию (slug обязателен, title опционален)",
)
async def create_category(
    body: CategoryCreate, session_factory: SessionFactoryDep
) -> KbCategory:
    if session_factory is None:
        raise HTTPException(status_code=503, detail="categories require postgres")
    slug = normalize_slug(body.slug)
    title = body.title or slug
    return await KbCategoryRepository(session_factory).upsert_category(slug, title)
