import os
import sqlite3
from typing import Optional, List
from contextlib import contextmanager

# Импорт настроек
try:
    from app.config import get_settings
    settings = get_settings()
    DATABASE_URL = settings.DATABASE_URL
except ImportError:
    # fallback для тестов или если настройки не готовы
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///data/knowledge_base.db")

def get_connection():
    """
    Фабрика соединений с БД.
    """
    # Удаляем префикс sqlite:/// если есть
    if DATABASE_URL.startswith("sqlite:///"):
        db_path = DATABASE_URL[10:]  # убираем "sqlite:///"
        return sqlite3.connect(db_path)
    else:
        # Для других СУБД (например, PostgreSQL) использовать соответствующий драйвер
        # Например: psycopg2.connect(DATABASE_URL)
        raise NotImplementedError(f"Unsupported database URL: {DATABASE_URL}")

@contextmanager
def get_db_connection():
    """Контекстный менеджер для автоматического закрытия соединения."""
    conn = get_connection()
    try:
        yield conn
    finally:
        conn.close()

def search_documents(
    tags: Optional[List[str]] = None,
    title: Optional[str] = None,
    doc_type: Optional[str] = None,
    doc_number: Optional[str] = None,
    doc_date_start: Optional[str] = None,
    doc_date_end: Optional[str] = None,
    created_at_start: Optional[str] = None,
    created_at_end: Optional[str] = None,
    updated_at_start: Optional[str] = None,
    updated_at_end: Optional[str] = None,
) -> str:
    """
    Поиск документов в БД knowledge_base.db по различным критериям.
    Возвращает строку с описанием найденных документов (top 10 по created_at).
    Все параметры опциональны.
    """
    query = """
        SELECT doc_number, doc_date, title, url, created_at
        FROM documents
        WHERE is_deleted = 0
    """
    params = []

    # Поиск по тегам (регистронезависимый, частичное совпадение)
    if tags:
        tag_conditions = []
        for tag in tags:
            tag_conditions.append("LOWER(tags) LIKE LOWER(?)")
            params.append(f"%{tag}%")
        query += " AND (" + " OR ".join(tag_conditions) + ")"

    if title:
        query += " AND LOWER(title) LIKE LOWER(?)"
        params.append(f"%{title}%")

    if doc_type:
        query += " AND LOWER(doc_type) = LOWER(?)"
        params.append(doc_type)

    if doc_number:
        query += " AND LOWER(doc_number) LIKE LOWER(?)"
        params.append(f"%{doc_number}%")

    if doc_date_start:
        query += " AND doc_date >= ?"
        params.append(doc_date_start)
    if doc_date_end:
        query += " AND doc_date <= ?"
        params.append(doc_date_end)

    if created_at_start:
        query += " AND created_at >= ?"
        params.append(created_at_start)
    if created_at_end:
        query += " AND created_at <= ?"
        params.append(created_at_end)

    if updated_at_start:
        query += " AND updated_at >= ?"
        params.append(updated_at_start)
    if updated_at_end:
        query += " AND updated_at <= ?"
        params.append(updated_at_end)

    query += " ORDER BY created_at DESC LIMIT 10"

    with get_db_connection() as conn:
        cursor = conn.execute(query, params)
        rows = cursor.fetchall()

    if not rows:
        return "Документы не найдены."

    result_lines = []
    for row in rows:
        doc_number_val, doc_date_val, title_val, url_val, _ = row
        number_date_part = ""
        if doc_number_val or doc_date_val:
            parts = []
            if doc_number_val:
                parts.append(doc_number_val)
            if doc_date_val:
                parts.append(doc_date_val)
            number_date_part = f"<{' '.join(parts)}> "
        url_part = f" {url_val}" if url_val else ""
        result_lines.append(f"{number_date_part}{title_val}{url_part}")

    return "\n".join(result_lines)