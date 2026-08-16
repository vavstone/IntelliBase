"""Идемпотентная загрузка JSONL-корпуса в Qdrant.

Запуск:
    uv run python scripts/load_to_qdrant.py
    uv run python scripts/load_to_qdrant.py --data data/fts_kb.jsonl --batch 256
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Добавляем корень проекта в sys.path для импорта app.*
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qdrant_client.models import PointStruct  # noqa: E402
from tqdm import tqdm  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.services.embeddings import EmbeddingsService  # noqa: E402
from app.services.loader_utils import read_jsonl, stable_id  # noqa: E402
from app.services.vector_store import VectorStore  # noqa: E402

logger = logging.getLogger("loader")
logging.basicConfig(level=logging.INFO, format="%(message)s")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Загрузка документов в Qdrant")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/fts_kb.jsonl"),
        help="Путь к JSONL с документами",
    )
    parser.add_argument("--batch", type=int, default=256, help="Размер upsert-батча")
    args = parser.parse_args()

    settings = get_settings()

    if not args.data.exists():
        raise SystemExit(
            f"Файл {args.data} не найден. Запустите `uv run python data/generate_fts_kb.py` "
            f"для генерации корпуса из data/orig_docs/."
        )

    docs = read_jsonl(args.data)
    logger.info("Загружаю %d документов из %s", len(docs), args.data)

    # EmbeddingsService с теми же настройками, что и основное приложение
    embeddings = EmbeddingsService(
        provider=settings.embedding.provider,
        model=settings.embedding.model,
        batch_size=settings.embedding.batch_size,
        max_retries=settings.embedding.max_retries,
        cache_dir=settings.embedding.cache_dir,
    )

    store = VectorStore(
        url=settings.qdrant_url,
        api_key=(
            settings.qdrant_api_key.get_secret_value()
            if settings.qdrant_api_key is not None
            else None
        ),
        collection=settings.qdrant_collection,
        dim=settings.embedding_dim,
    )

    try:
        await store.ensure_collection()
        logger.info(
            "Коллекция %s готова (dim=%d, distance=COSINE)",
            settings.qdrant_collection,
            settings.embedding_dim,
        )

        texts = [d["text"] for d in docs]

        # E5-модель: добавляем префикс "passage: " для документов
        if embeddings._is_e5():
            logger.info("E5-модель: применяю префикс 'passage:' к документам")
            vectors = []
            batch_embed = settings.embedding.batch_size
            for i in tqdm(range(0, len(texts), batch_embed), desc="embeddings"):
                chunk = texts[i : i + batch_embed]
                vectors.extend(await embeddings.embed_documents(chunk))
        else:
            vectors = []
            batch_embed = settings.embedding.batch_size
            for i in tqdm(range(0, len(texts), batch_embed), desc="embeddings"):
                chunk = texts[i : i + batch_embed]
                vectors.extend(await embeddings.embed_texts(chunk))

        if len(vectors) != len(docs):
            raise RuntimeError(
                f"Получено {len(vectors)} embeddings на {len(docs)} документов"
            )

        if vectors and len(vectors[0]) != settings.embedding_dim:
            raise RuntimeError(
                f"Embedding dim={len(vectors[0])} != EMBEDDING_DIM={settings.embedding_dim}. "
                f"Сверьте имя модели и значение EMBEDDING_DIM в .env."
            )

        points = [
            PointStruct(
                id=stable_id(doc["source"], doc["chunk_index"]),
                vector=vec,
                payload={
                    "source": doc["source"],
                    "chunk_index": doc["chunk_index"],
                    "text": doc["text"],
                    "ps": doc.get("ps", "Разное"),
                    "year": doc.get("year", "unknown"),
                    "created_at": doc["created_at"],
                },
            )
            for doc, vec in zip(docs, vectors, strict=True)
        ]

        logger.info("Заливаю %d точек батчами по %d", len(points), args.batch)
        await store.upsert(points, batch_size=args.batch)

        total = await store.count()
        logger.info("Готово. В коллекции %d точек (points_count из get_collection)", total)
    finally:
        await store.close()


if __name__ == "__main__":
    asyncio.run(main())
