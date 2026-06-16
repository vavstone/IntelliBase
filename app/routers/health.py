import asyncio
from fastapi import APIRouter, Request, Response, status

router = APIRouter(tags=["health"])

@router.get("/health", summary="Liveness probe")
async def health() -> dict:
    return {"status": "ok"}


@router.get("/ready", summary="Readiness probe")
async def ready(request: Request, response: Response) -> dict:
    cache = getattr(request.app.state, "redis", None)
    redis_ok = False
    if cache is not None:
        try:
            await asyncio.wait_for(cache.ping(), timeout=1.5)
            redis_ok = True
        except (Exception, asyncio.TimeoutError):
            redis_ok = False

    if redis_ok:
        return {"status": "ok", "redis": "up"}

    response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {"status": "degraded", "redis": "down"}
