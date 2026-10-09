"""Тесты бота: подпись источников (format_sources) и рендер стрима.

Рендер-тесты идут через edit_text-путь (`_stream_via_edit_text`): у фейкового
Message нет `bot.send_message_draft`, и stream_to_chat сам переключается на
fallback — то есть проверяется реальная ветка кода, а не только сборка текста.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from bot.services.streaming import (
    TELEGRAM_LIMIT,
    clip_for_telegram,
    format_sources,
    stream_to_chat,
)


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


# ── подмена буфера при output-модерации ─────────────────────────────────


class _FakeSent:
    def __init__(self) -> None:
        self.edits: list[str] = []

    async def edit_text(self, text, **kwargs):  # noqa: ANN001
        self.edits.append(text)


class _FakeMessage:
    """Мини-заглушка aiogram Message без `bot` — стриминг уходит в edit_text-путь."""

    def __init__(self) -> None:
        self.chat = SimpleNamespace(id=1)
        self.sent = _FakeSent()
        self.answer = AsyncMock(return_value=self.sent)


async def _events(items):
    for item in items:
        yield item


@pytest.mark.asyncio
async def test_moderation_notice_replaces_streamed_text() -> None:
    """Заблокированный ответ: финальный текст заменяется заглушкой из события.

    Токены к моменту проверки уже ушли в чат (и в драфт), поэтому единственная
    точка подмены — финальное сообщение, собранное из буфера.
    """
    message = _FakeMessage()
    await stream_to_chat(
        message,
        _events(
            [
                {"type": "token", "delta": "заблокированный текст"},
                {
                    "type": "moderation_notice",
                    "categories": ["test"],
                    "reasons": ["тест"],
                    "replacement": "Ответ скрыт модерацией.",
                },
                {"type": "message_saved", "message_id": "m1"},
            ]
        ),
    )
    # Точку MarkdownV2 экранирует (`\.`) — проверяем основу без спецсимволов.
    assert "Ответ скрыт модерацией" in message.sent.edits[-1]
    assert "заблокированный текст" not in message.sent.edits[-1]


@pytest.mark.asyncio
async def test_stream_without_moderation_keeps_text() -> None:
    message = _FakeMessage()
    await stream_to_chat(
        message, _events([{"type": "token", "delta": "обычный ответ"}])
    )
    assert "обычный ответ" in message.sent.edits[-1]


# ── обрезка под лимит Telegram ──────────────────────────────────────────


def test_clip_keeps_short_text_untouched() -> None:
    assert clip_for_telegram("короткий ответ") == "короткий ответ"


def test_clip_trims_long_text_with_ellipsis() -> None:
    """Страховка от ответа >4096: Telegram отверг бы такое сообщение целиком."""
    long_text = "а" * (TELEGRAM_LIMIT + 500)
    clipped = clip_for_telegram(long_text)
    assert len(clipped) == TELEGRAM_LIMIT
    assert clipped.endswith("…")


@pytest.mark.asyncio
async def test_long_streamed_answer_is_clipped() -> None:
    """Длинный ответ доходит до Telegram уже обрезанным (чатовый путь)."""
    message = _FakeMessage()
    await stream_to_chat(
        message, _events([{"type": "token", "delta": "б" * (TELEGRAM_LIMIT + 300)}])
    )
    assert len(message.sent.edits[-1]) <= TELEGRAM_LIMIT
