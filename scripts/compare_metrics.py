"""Сравнение ранжирования cosine vs dot на пяти запросах.

Создаёт временную пару коллекций `{base}_cosine` и `{base}_dot` на одних
и тех же векторах, прогоняет запросы, выводит таблицу и сохраняет результат
в `docs/metric_comparison.json`. Временные коллекции удаляются.

Запуск:
    uv run python scripts/compare_metrics.py
    uv run python scripts/compare_metrics.py --data data/fts_kb_demo.jsonl
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qdrant_client import AsyncQdrantClient  # noqa: E402
from qdrant_client.models import Distance, PointStruct, VectorParams  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.services.embeddings import EmbeddingsService  # noqa: E402
from app.services.loader_utils import read_jsonl, stable_id  # noqa: E402

logger = logging.getLogger("compare")
logging.basicConfig(level=logging.INFO, format="%(message)s")

QUERIES: list[str] = [
    "Какие программные задачи входят в состав КПС «Постконтроль»?",
    "Как работает автоматическая проверка декларантов на попадание в таблицу 100MILLION?",
    "Какой классификатор используется для указания в таможенной декларации сведений о товарах без обязательной маркировки?",
    "Какие СУБД и языки программирования используются в КПС «Постконтроль»?",
    "Какие PL/SQL пакеты отвечают за взаимодействие КПС «Постконтроль» с системой маркировки товаров?",
]


async def _upsert_clone(
    client: AsyncQdrantClient,
    base_name: str,
    suffix: str,
    distance: Distance,
    points: list[PointStruct],
    dim: int,
) -> str:
    """Создаёт коллекцию `{base}_{suffix}` с заданной метрикой и заливает точки."""
    name = f"{base_name}_{suffix}"
    existing = {c.name for c in (await client.get_collections()).collections}
    if name in existing:
        await client.delete_collection(name)
    await client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=dim, distance=distance),
    )
    await client.upsert(collection_name=name, points=points, wait=True)
    return name


async def _top5(
    client: AsyncQdrantClient, collection: str, query_vec: list[float]
) -> list[str]:
    """top-5 id из коллекции для одного вектора."""
    result = await client.query_points(
        collection_name=collection,
        query=query_vec,
        limit=5,
        with_payload=False,
    )
    return [str(p.id) for p in result.points]


async def main() -> None:
    parser = argparse.ArgumentParser(description="Сравнение cosine vs dot на пяти запросах")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/fts_kb_demo.jsonl"),
        help="Путь к JSONL с документами (по умолчанию demo-подмножество)",
    )
    args = parser.parse_args()

    settings = get_settings()
    data_path = args.data
    if not data_path.exists():
        raise SystemExit(
            f"Файл {data_path} не найден. Запустите `uv run python data/generate_fts_kb.py`."
        )

    docs = read_jsonl(data_path)
    logger.info("Загружено %d документов из %s", len(docs), data_path)

    embeddings = EmbeddingsService(
        provider=settings.embedding.provider,
        model=settings.embedding.model,
        batch_size=settings.embedding.batch_size,
        max_retries=settings.embedding.max_retries,
        cache_dir=settings.embedding.cache_dir,
    )

    qdrant = AsyncQdrantClient(
        url=settings.qdrant_url,
        api_key=(
            settings.qdrant_api_key.get_secret_value()
            if settings.qdrant_api_key is not None
            else None
        ),
    )

    try:
        # Эмбеддинги для документов (с E5-префиксом passage: если нужно)
        logger.info("Считаю embeddings для %d документов и %d запросов", len(docs), len(QUERIES))
        if embeddings._is_e5():
            doc_vectors = await embeddings.embed_documents([d["text"] for d in docs])
            query_vectors_list = []
            for q_text in QUERIES:
                qv = await embeddings.embed_query(q_text)
                query_vectors_list.append(qv)
        else:
            doc_vectors = await embeddings.embed_texts([d["text"] for d in docs])
            query_vectors_list = await embeddings.embed_texts(QUERIES)

        points = [
            PointStruct(
                id=stable_id(d["source"], d["chunk_index"]),
                vector=v,
                payload={"source": d["source"], "chunk_index": d["chunk_index"]},
            )
            for d, v in zip(docs, doc_vectors, strict=True)
        ]

        base = settings.qdrant_collection
        cos_name = await _upsert_clone(qdrant, base, "cosine", Distance.COSINE, points, settings.embedding_dim)
        dot_name = await _upsert_clone(qdrant, base, "dot", Distance.DOT, points, settings.embedding_dim)

        logger.info("\n%-50s | %-30s | %-30s | match", "Запрос", "top-5 cosine", "top-5 dot")
        logger.info("-" * 130)
        rows: list[dict] = []
        for q, qv in zip(QUERIES, query_vectors_list, strict=True):
            cos_top = await _top5(qdrant, cos_name, qv)
            dot_top = await _top5(qdrant, dot_name, qv)
            match = cos_top == dot_top
            rows.append(
                {
                    "query": q,
                    "cosine": cos_top,
                    "dot": dot_top,
                    "match": match,
                }
            )
            logger.info(
                "%-50s | %-30s | %-30s | %s",
                q[:48],
                ",".join(c[:8] for c in cos_top),
                ",".join(d[:8] for d in dot_top),
                "✓" if match else "✗",
            )

        out = Path("docs/metric_comparison.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("\nДетали сохранены в %s", out)

        # Удаляем временные коллекции
        await qdrant.delete_collection(cos_name)
        await qdrant.delete_collection(dot_name)
        logger.info("Временные коллекции %s, %s удалены", cos_name, dot_name)
    finally:
        await qdrant.close()


if __name__ == "__main__":
    asyncio.run(main())
