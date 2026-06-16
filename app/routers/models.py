from fastapi import APIRouter

from app.schemas.models import ModelInfo, CATALOG

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=list[ModelInfo])
async def list_models() -> list[ModelInfo]:
    return list(CATALOG.values())
