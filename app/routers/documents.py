"""Офлайн-ручки индексации: загрузка документа и переиндексация корпуса.

Тяжёлая работа (парсинг + эмбеддинг) уходит в `BackgroundTasks` и не блокирует
запрос: ответ 202 Accepted отдаётся сразу со статусом «queued». Для продакшена
вместо `BackgroundTasks` — отдельный воркер на Celery/RQ/Dramatiq.
"""

import logging
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, BackgroundTasks, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.deps.providers import IngestionServiceDep, SessionFactoryDep, SettingsDep
from app.kb.domain import normalize_slug
from app.kb.repository import KbCategoryRepository

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])


class ReindexRequest(BaseModel):
    mode: Literal["full", "incremental", "files"] = "incremental"
    files: list[str] = Field(default_factory=list)


class QueuedResponse(BaseModel):
    status: Literal["queued"] = "queued"
    detail: str


@router.post(
    "/upload",
    status_code=202,
    response_model=QueuedResponse,
    summary="Загрузить документ в базу знаний",
    description="Сохраняет файл в data/kb/<category>/ и запускает индексацию в фоне.",
)
async def upload_document(
    file: UploadFile,
    background: BackgroundTasks,
    ingestion: IngestionServiceDep,
    settings: SettingsDep,
    session_factory: SessionFactoryDep,
    category: str | None = Form(None),
) -> QueuedResponse:
    if ingestion is None:
        raise HTTPException(status_code=503, detail="индексатор недоступен")
    if not file.filename:
        raise HTTPException(status_code=422, detail="имя файла обязательно")

    # Категория — slug ПС (латиница). Пусто/не задано -> raznoe.
    slug = normalize_slug(category)

    # Best-effort: регистрируем категорию в Postgres, если она новая. При
    # недоступной БД загрузка всё равно проходит — категория возьмётся из пути.
    if session_factory is not None:
        try:
            await KbCategoryRepository(session_factory).upsert_category(slug, slug)
        except Exception:
            logger.warning("не удалось зарегистрировать категорию %s", slug, exc_info=True)

    # Файл кладём в категорийную подпапку, чтобы category_from_path вернул slug,
    # а не имя файла (иначе документ не получит принадлежности к ПС).
    target_dir = Path(settings.rag_data_dir) / slug
    target_dir.mkdir(parents=True, exist_ok=True)
    # Path(...).name отбрасывает любые ../ — защита от обхода каталога.
    target = target_dir / Path(file.filename).name
    target.write_bytes(await file.read())
    background.add_task(ingestion.run_for_file, target)
    return QueuedResponse(detail=f"{target.name} принят (категория {slug}), индексация в фоне")


@router.post(
    "/reindex",
    status_code=202,
    response_model=QueuedResponse,
    summary="Переиндексировать корпус",
    description="full — вычистить и заново; incremental — по хешам docstore; files — точечно.",
)
async def reindex(
    req: ReindexRequest,
    background: BackgroundTasks,
    ingestion: IngestionServiceDep,
) -> QueuedResponse:
    if ingestion is None:
        raise HTTPException(status_code=503, detail="индексатор недоступен")
    if req.mode == "files":
        if not req.files:
            raise HTTPException(
                status_code=422, detail="mode=files: список files не должен быть пустым"
            )
        background.add_task(ingestion.ingest_files, req.files)
    elif req.mode == "full":
        background.add_task(ingestion.reindex_all)
    else:
        background.add_task(ingestion.ingest_all)
    return QueuedResponse(detail=f"режим {req.mode}, индексация в фоне")
