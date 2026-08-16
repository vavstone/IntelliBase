"""Тот же RAG, но руками — без LlamaIndex.

Полный путь на чистом openai + qdrant-client + собственном EmbeddingsService:
чтение файлов → наивный чанкинг (фикс. окно, без границ предложений) → эмбеддинги
(E5, префиксы query:/passage:) → upsert с плоским payload → query_points →
сборка промпта → генерация. Нужен для сравнения с app/services/rag.py (см. docs/rag.md).

Запуск отдельно:
    uv run python -m app.services.rag_baremetal
"""

import logging
from pathlib import Path

from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.core.config import Settings as AppSettings
from app.core.config import get_settings
from app.services.embeddings import EmbeddingsService
from app.services.loader_utils import stable_id

logger = logging.getLogger(__name__)

# Наивный чанкинг: фиксированное окно в символах. Взят с запасом под
# max_seq_length=512 токенов у E5 (русский ~4-5 симв. на токен).
CHUNK_CHARS = 1200

SYSTEM_PROMPT = (
    "Отвечай строго по предоставленному контексту. Если ответа в контексте нет — "
    "честно скажи, что не нашёл, и ничего не выдумывай. Отвечай по-русски, коротко."
)


def _read_text(path: Path) -> str:
    """Извлекает текст из файла по расширению (.md/.txt/.pdf/.docx)."""
    suffix = path.suffix.lower()
    if suffix in (".md", ".txt"):
        return path.read_text(encoding="utf-8")
    if suffix == ".pdf":
        from pypdf import PdfReader

        return "\n".join(p.extract_text() or "" for p in PdfReader(str(path)).pages)
    if suffix == ".docx":
        from docx import Document

        return "\n".join(p.text for p in Document(str(path)).paragraphs if p.text.strip())
    return ""


def _naive_chunk(text: str, size: int = CHUNK_CHARS) -> list[str]:
    """Режет текст на куски фиксированного размера (без учёта границ предложений)."""
    text = text.strip()
    return [text[i : i + size] for i in range(0, len(text), size)]


class BareMetalRAG:
    """RAG руками: то же поведение, что у RAGService, но без фреймворка."""

    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        qdrant_key = (
            settings.qdrant_api_key.get_secret_value()
            if settings.qdrant_api_key is not None
            else None
        )
        self._client = AsyncQdrantClient(url=settings.qdrant_url, api_key=qdrant_key)
        self._collection = settings.rag_collection_bare

        # Та же embed-модель, что и в LlamaIndex-версии (E5, префиксы внутри сервиса).
        self._embeddings = EmbeddingsService(
            provider=settings.embedding.provider,
            model=settings.embedding.model,
            batch_size=settings.embedding.batch_size,
            cache_dir=settings.embedding.cache_dir,
        )
        # LLM — та же Ollama через OpenAI-совместимый эндпоинт.
        self._llm = AsyncOpenAI(
            base_url=settings.llm.ollama_base_url,
            api_key="ollama",
        )

    def _load_chunks(self) -> list[dict]:
        """Читает корпус и возвращает список {source, index, text}."""
        out: list[dict] = []
        for path in sorted(Path(self._settings.rag_data_dir).iterdir()):
            if path.suffix.lower() not in (".md", ".txt", ".pdf", ".docx"):
                continue
            text = _read_text(path)
            for idx, chunk in enumerate(_naive_chunk(text)):
                out.append({"source": path.name, "index": idx, "text": chunk})
        return out

    async def ensure_indexed(self) -> None:
        """Идемпотентный ingestion: плоский payload {text, source}."""
        exists = await self._client.collection_exists(self._collection)
        if exists:
            info = await self._client.get_collection(self._collection)
            if info.points_count and info.points_count > 0:
                logger.info(
                    "bare-metal: коллекция %s уже наполнена (%d точек)",
                    self._collection,
                    info.points_count,
                )
                return

        if not exists:
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(
                    size=self._settings.embedding_dim, distance=Distance.COSINE
                ),
            )

        chunks = self._load_chunks()
        vectors = await self._embeddings.embed_documents([c["text"] for c in chunks])
        points = [
            PointStruct(
                id=stable_id(c["source"], c["index"]),
                vector=vec,
                payload={"text": c["text"], "source": c["source"]},
            )
            for c, vec in zip(chunks, vectors, strict=True)
        ]
        await self._client.upsert(collection_name=self._collection, points=points, wait=True)
        logger.info(
            "bare-metal: проиндексировано %d чанков из %d файлов в коллекцию %s",
            len(points),
            len({c["source"] for c in chunks}),
            self._collection,
        )

    async def answer(self, question: str) -> dict:
        q_vec = await self._embeddings.embed_query(question)
        hits = (
            await self._client.query_points(
                collection_name=self._collection,
                query=q_vec,
                limit=self._settings.rag_top_k,
                with_payload=True,
            )
        ).points

        context = "\n\n".join(f"[{h.payload['source']}] {h.payload['text']}" for h in hits)
        completion = await self._llm.chat.completions.create(
            model=self._settings.rag_llm_model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Контекст:\n{context}\n\nВопрос: {question}"},
            ],
        )

        top_score = hits[0].score if hits else 0.0
        answer_text = completion.choices[0].message.content or ""
        if top_score < self._settings.rag_score_threshold:
            answer_text = "В базе знаний нет ответа на этот вопрос."

        return {
            "answer": answer_text,
            "top_score": round(top_score, 3),
            "sources": [
                {
                    "text": h.payload["text"][:300],
                    "source": h.payload["source"],
                    "score": round(h.score, 3),
                }
                for h in hits
            ],
        }

    async def close(self) -> None:
        await self._client.close()


def _demo() -> None:
    import asyncio
    import json

    async def run() -> None:
        service = BareMetalRAG(get_settings())
        await service.ensure_indexed()
        for question in (
            "Что входит в состав КПС Тарифы «Реестр ОИС»?",
            "Какая завтра погода в Москве?",
        ):
            result = await service.answer(question)
            print(f"\nВопрос: {question}")
            print(json.dumps(result, ensure_ascii=False, indent=2))
        await service.close()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    asyncio.run(run())


if __name__ == "__main__":
    _demo()
