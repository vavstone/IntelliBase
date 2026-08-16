"""Генератор JSONL-корпуса из функциональных требований ФТС.

Обходит data/orig_docs/{год}/{ПС}/..., извлекает текст из PDF и DOCX,
разбивает на чанки и сохраняет в data/fts_kb.jsonl.

Запуск:
    uv run python data/generate_fts_kb.py
"""

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from pypdf import PdfReader
from docx import Document

logger = logging.getLogger("generate_fts")
logging.basicConfig(level=logging.INFO, format="%(message)s")

ORIG_DOCS = Path(__file__).resolve().parent / "orig_docs"
OUT = Path(__file__).resolve().parent / "fts_kb.jsonl"

# Приблизительный размер чанка в символах
CHUNK_MIN = 400
CHUNK_MAX = 1200


def extract_text_from_pdf(path: Path) -> str:
    """Извлекает текст из PDF-файла."""
    try:
        reader = PdfReader(str(path))
        parts: list[str] = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                parts.append(text)
        return "\n".join(parts)
    except Exception as e:
        logger.warning("Ошибка чтения PDF %s: %s", path.name, e)
        return ""


def extract_text_from_docx(path: Path) -> str:
    """Извлекает текст из DOCX-файла."""
    try:
        doc = Document(str(path))
        parts: list[str] = []
        for para in doc.paragraphs:
            if para.text.strip():
                parts.append(para.text.strip())
        return "\n".join(parts)
    except Exception as e:
        logger.warning("Ошибка чтения DOCX %s: %s", path.name, e)
        return ""


def clean_text(text: str) -> str:
    """Убирает лишние пробелы и мусор."""
    # Убираем заголовки с номерами страниц
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    # Схлопываем множественные пробелы
    text = re.sub(r'[ \t]+', ' ', text)
    # Схлопываем множественные переносы строк
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def chunk_text(text: str, min_size: int = CHUNK_MIN, max_size: int = CHUNK_MAX) -> list[str]:
    """Разбивает текст на чанки по границам предложений."""
    if not text or len(text) < 50:
        return []

    # Разбиваем по двойным переносам строк (абзацы)
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para)

        # Если один параграф больше max_size — режем по предложениям
        if para_len > max_size:
            # Сначала сбрасываем накопленное
            if current:
                chunks.append('\n\n'.join(current))
                current = []
                current_len = 0

            sentences = re.split(r'(?<=[.!?])\s+', para)
            sent_buf: list[str] = []
            sent_len = 0
            for sent in sentences:
                if sent_len + len(sent) > max_size and sent_buf:
                    chunks.append(' '.join(sent_buf))
                    sent_buf = []
                    sent_len = 0
                sent_buf.append(sent)
                sent_len += len(sent)
            if sent_buf:
                chunks.append(' '.join(sent_buf))
            continue

        # Обычный случай: накапливаем параграфы до достижения min_size
        if current_len + para_len > max_size and current_len >= min_size:
            chunks.append('\n\n'.join(current))
            current = []
            current_len = 0

        current.append(para)
        current_len += para_len

    if current:
        chunks.append('\n\n'.join(current))

    return chunks


def ps_from_path(rel_path: Path) -> str:
    """Извлекает ПС из пути вида ФТ_2023/Тарифы/..."""
    parts = rel_path.parts
    if len(parts) >= 2:
        return parts[1]  # Второй элемент — ПС
    return "Разное"


def year_from_path(rel_path: Path) -> str:
    """Извлекает год из пути вида ФТ_2023/..."""
    parts = rel_path.parts
    if parts and parts[0].startswith("ФТ_"):
        return parts[0][3:]  # "2023", "2024", "2025"
    return "unknown"


def source_from_path(rel_path: Path) -> str:
    """Формирует source из относительного пути."""
    return str(rel_path).replace("\\", "/")


def main() -> None:
    if not ORIG_DOCS.exists():
        raise SystemExit(f"Директория {ORIG_DOCS} не найдена. Положите документы в data/orig_docs/")

    # Собираем все поддерживаемые файлы
    pdf_files = sorted(ORIG_DOCS.rglob("*.pdf")) + sorted(ORIG_DOCS.rglob("*.PDF"))
    docx_files = sorted(ORIG_DOCS.rglob("*.docx")) + sorted(ORIG_DOCS.rglob("*.DOCX"))
    doc_files = sorted(ORIG_DOCS.rglob("*.doc"))

    all_files = pdf_files + docx_files + doc_files
    logger.info("Найдено файлов: %d PDF + %d DOCX + %d DOC = %d всего",
                len(pdf_files), len(docx_files), len(doc_files), len(all_files))

    points: list[dict] = []
    total_chunks = 0
    skipped_empty = 0

    for file_path in all_files:
        rel_path = file_path.relative_to(ORIG_DOCS)
        ps = ps_from_path(rel_path)
        year = year_from_path(rel_path)
        source = source_from_path(rel_path)

        # Извлекаем текст
        suffix = file_path.suffix.lower()
        if suffix == '.pdf':
            text = extract_text_from_pdf(file_path)
        elif suffix in ('.docx', '.doc'):
            text = extract_text_from_docx(file_path)
        else:
            continue

        if not text:
            skipped_empty += 1
            continue

        text = clean_text(text)
        if not text:
            skipped_empty += 1
            continue

        # Чанкуем
        chunks = chunk_text(text)
        if not chunks:
            skipped_empty += 1
            continue

        # Используем дату модификации файла как created_at
        mtime = file_path.stat().st_mtime
        created_at = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()

        for idx, chunk in enumerate(chunks):
            points.append({
                "source": source,
                "chunk_index": idx,
                "text": chunk,
                "ps": ps,
                "year": year,
                "created_at": created_at,
            })
            total_chunks += 1

    if not points:
        raise SystemExit("Не удалось извлечь ни одного чанка. Проверьте документы в data/orig_docs/")

    # Записываем JSONL
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for p in points:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # Статистика по ПС
    ps_counts: dict[str, int] = {}
    for p in points:
        ps_counts[p["ps"]] = ps_counts.get(p["ps"], 0) + 1

    logger.info("Записано %d чанков в %s", total_chunks, OUT)
    logger.info("Пропущено пустых файлов: %d", skipped_empty)
    logger.info("Распределение по ПС:")
    for ps_name, count in sorted(ps_counts.items(), key=lambda x: -x[1]):
        logger.info("  %s: %d чанков", ps_name, count)


if __name__ == "__main__":
    main()
