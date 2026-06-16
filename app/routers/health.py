from fastapi import APIRouter

router = APIRouter(tags=["health"])

@router.get("/health", summary="Liveness probe")
async def health() -> dict:
    return {"status": "ok"}