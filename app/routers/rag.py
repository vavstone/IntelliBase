"""RAG-endpoint: ответ по базе знаний с цитатами на источники."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.deps.providers import RAGServiceDep

router = APIRouter(prefix="/rag", tags=["rag"])


class RAGQuery(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class RAGSource(BaseModel):
    text: str
    source: str | None = None
    score: float


class RAGAnswer(BaseModel):
    answer: str
    top_score: float
    sources: list[RAGSource]


@router.post(
    "/query",
    response_model=RAGAnswer,
    summary="Ответ по базе знаний (RAG)",
    description="Ищет релевантные чанки в Qdrant и генерирует ответ строго по контексту.",
    responses={
        200: {"description": "Ответ с источниками"},
        503: {"description": "RAG-индекс недоступен"},
    },
)
async def rag_query(req: RAGQuery, rag: RAGServiceDep) -> RAGAnswer:
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG-индекс недоступен")
    result = await rag.answer(req.question)
    return RAGAnswer(**result)
