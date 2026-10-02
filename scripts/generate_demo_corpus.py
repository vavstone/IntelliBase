"""Генерация демонстрационного корпуса `data/demo_kb`.

Синтетические документы (PDF + DOCX) по категориям-ПС: их можно безопасно
хранить в репозитории и использовать для наполнения коллекции на чистом
клоне. Содержимое описано в `scripts/demo_corpus/content_*.py`.

Запуск:
    uv run python scripts/generate_demo_corpus.py            # записать корпус
    uv run python scripts/generate_demo_corpus.py --check     # только проверка
    uv run python scripts/generate_demo_corpus.py --clean     # очистить каталог

PDF собираются шрифтом DejaVu Sans (кириллица). Если он не найден — укажите
каталог со шрифтом: DEMO_CORPUS_FONT_DIR=/path/to/fonts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from demo_corpus import ALL_DOCS, find_font_dir, render


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Генерация демо-корпуса data/demo_kb")
    parser.add_argument(
        "--out",
        default="data/demo_kb",
        help="каталог корпуса (по умолчанию data/demo_kb)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="не записывать файлы — только проверить состав и доступность шрифта",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="удалить существующие файлы в каталоге перед генерацией",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out)

    docs = ALL_DOCS
    categories = sorted({d.category for d in docs})
    print(f"Документов: {len(docs)}; категорий: {len(categories)}: {', '.join(categories)}")

    try:
        font_dir = find_font_dir()
    except FileNotFoundError as e:
        print(f"ОШИБКА: {e}")
        return 2
    print(f"Шрифт для PDF: {font_dir}")

    if args.check:
        for doc in docs:
            print(f"  [{doc.category}] {doc.filename}")
        print("Проверка пройдена (файлы не записаны).")
        return 0

    if args.clean and out_dir.exists():
        removed = 0
        for path in out_dir.rglob("*"):
            if path.is_file():
                path.unlink()
                removed += 1
        print(f"Очищено файлов: {removed}")

    total = 0
    for doc in docs:
        path = render(doc, out_dir, font_dir=font_dir)
        size = path.stat().st_size
        total += size
        print(f"  {path} — {size / 1024:.1f} КБ")

    print(f"Готово: {len(docs)} документов, {total / 1024 / 1024:.2f} МБ в {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
