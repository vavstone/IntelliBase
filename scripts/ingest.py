"""CLI индексации корпуса в Qdrant (офлайн-контур RAG).

Запуск:
    uv run python scripts/ingest.py data/kb           # инкрементально (UPSERTS)
    uv run python scripts/ingest.py data/kb --full    # вычистить и заново
    uv run python scripts/ingest.py --files data/kb/a.pdf data/kb/b.docx

Повторный запуск без изменений печатает «0 changed, N unchanged» — это проверка
того, что IngestionPipeline + DocstoreStrategy.UPSERTS + docstore на диске не
дублируют чанки между запусками.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Windows-консоль может быть в cp1251 — переключаем stdout на UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings  # noqa: E402
from app.services.ingestion import IngestionService  # noqa: E402

logger = logging.getLogger("ingest")
logging.basicConfig(level=logging.INFO, format="%(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Индексация корпуса в Qdrant")
    parser.add_argument(
        "data_dir",
        nargs="?",
        default=None,
        help="каталог корпуса (по умолчанию RAG_DATA_DIR из конфига)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="полная переиндексация: вычистить коллекцию и docstore",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=None,
        help="точечно проиндексировать перечисленные файлы",
    )
    args = parser.parse_args()

    settings = get_settings()
    if args.data_dir:
        settings = settings.model_copy(update={"rag_data_dir": Path(args.data_dir)})

    service = IngestionService(settings)
    try:
        if args.full:
            n = service.reindex_all()
            print(f"full reindex: {n} nodes")
        elif args.files:
            n = service.ingest_files(args.files)
            print(f"files ingested: {n} nodes")
        else:
            stats = service.ingest_all()
            print(
                f"{stats['changed']} changed, {stats['unchanged']} unchanged "
                f"({stats['nodes']} nodes total)"
            )
    finally:
        service.close()


if __name__ == "__main__":
    main()
