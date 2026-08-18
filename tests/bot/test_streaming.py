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
