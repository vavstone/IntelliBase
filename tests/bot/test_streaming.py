"""Тесты format_sources — подпись источников RAG-ответа в Telegram."""

from bot.services.streaming import format_sources


def test_format_sources_empty() -> None:
    assert format_sources([]) == ""


def test_format_sources_lists_source_with_page() -> None:
    out = format_sources([{"id": 1, "file_name": "заявка.pdf", "page": 3}])
    assert "Источники:" in out
    assert "[1] заявка.pdf, стр. 3" in out


def test_format_sources_no_page_omits_page() -> None:
    out = format_sources([{"id": 1, "file_name": "заявка.md", "page": None}])
    assert "[1] заявка.md" in out
    assert "стр." not in out


def test_format_sources_caps_at_5() -> None:
    sources = [{"id": i, "file_name": f"f{i}.pdf", "page": None} for i in range(1, 8)]
    out = format_sources(sources)
    assert "[5] f5.pdf" in out
    assert "[6] f6.pdf" not in out
    assert "[7] f7.pdf" not in out


def test_format_sources_merges_chunks_of_same_page() -> None:
    """Два фрагмента одной страницы — одна строка, оба номера сохранены.

    Регрессия демо: retrieve вернул два куска с одной страницы, и подпись
    показывала один и тот же файл двумя одинаковыми строками.
    """
    out = format_sources([
        {"id": 1, "file_name": "Регламент.pdf", "page": 1},
        {"id": 2, "file_name": "Регламент.pdf", "page": 1},
    ])
    assert out.count("Регламент.pdf") == 1
    assert "[1][2] Регламент.pdf, стр. 1" in out


def test_format_sources_keeps_different_pages_apart() -> None:
    """Один файл, разные страницы — разные строки: страница часть ссылки."""
    out = format_sources([
        {"id": 1, "file_name": "Регламент.pdf", "page": 1},
        {"id": 2, "file_name": "Регламент.pdf", "page": 7},
    ])
    assert "[1] Регламент.pdf, стр. 1" in out
    assert "[2] Регламент.pdf, стр. 7" in out


def test_format_sources_documents_without_page_merge() -> None:
    """У docx страницы нет — фрагменты схлопываются по имени файла."""
    out = format_sources([
        {"id": 1, "file_name": "ТТ.docx", "page": None},
        {"id": 2, "file_name": "ТТ.docx", "page": None},
    ])
    assert out.count("ТТ.docx") == 1
    assert "[1][2] ТТ.docx" in out


def test_format_sources_merges_before_capping() -> None:
    """Пять строк — это пять разных документов, а не пять фрагментов.

    Иначе дубли съедали лимит: две страницы одного отчёта вытесняли чужой
    документ из подписи.
    """
    sources = [
        {"id": 1, "file_name": "один.pdf", "page": 1},
        {"id": 2, "file_name": "один.pdf", "page": 1},
        {"id": 3, "file_name": "два.pdf", "page": None},
        {"id": 4, "file_name": "три.pdf", "page": None},
        {"id": 5, "file_name": "четыре.pdf", "page": None},
        {"id": 6, "file_name": "пять.pdf", "page": None},
        {"id": 7, "file_name": "шесть.pdf", "page": None},
    ]
    out = format_sources(sources)
    assert "[1][2] один.pdf, стр. 1" in out
    assert "пять.pdf" in out          # 5-й документ ещё виден
    assert "шесть.pdf" not in out     # 6-й отсечён по лимиту строк
