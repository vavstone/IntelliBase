"""strip_citations: маркеры источников не должны попадать в метрики содержания.

Модуль без зависимости от ragas — тест идёт в обычном прогоне pytest.
"""

from app.eval.citations import strip_citations


def test_strips_bare_marker() -> None:
    assert strip_citations("Приказ № 233 [1].") == "Приказ № 233."


def test_strips_expanded_marker_with_filename() -> None:
    text = "Размер пакета — 500 МБ [1 — Регламент обмена.pdf]."
    assert strip_citations(text) == "Размер пакета — 500 МБ."


def test_strips_multiple_markers_and_dashes() -> None:
    text = "Факт [1 — a.pdf] и ещё [2 – b.docx] и [3 - c.pdf]."
    assert strip_citations(text) == "Факт и ещё и."


def test_keeps_numbers_in_brackets() -> None:
    """[2026] — не маркер источника (индекс ≤ top_k), не трогаем."""
    assert strip_citations("Отчёт [2026] года") == "Отчёт [2026] года"


def test_plain_text_unchanged() -> None:
    assert strip_citations("Ответ без ссылок.") == "Ответ без ссылок."
