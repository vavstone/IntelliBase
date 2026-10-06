"""Юнит-тесты доставки агентского черновика в Telegram (app/services/notifier.py).

`notify_user` (реальный HTTP к боту) подменяется моком: проверяется контракт —
что уходит в бота и что недоставка не превращается в «отправлено».
"""

from unittest.mock import AsyncMock

import pytest

import app.services.notifier as notifier_module
from app.services.notifier import deliver_to_bot

BOT_URL = "http://bot:9000"
TOKEN = "internal-token"


@pytest.mark.asyncio
async def test_delivers_draft_to_bot(monkeypatch) -> None:
    notify = AsyncMock()
    monkeypatch.setattr(notifier_module, "notify_user", notify)

    result = await deliver_to_bot(
        {"chat_id": "73061220", "text": "Отчёт готов"}, BOT_URL, TOKEN
    )

    notify.assert_awaited_once_with(73061220, "Отчёт готов", BOT_URL, TOKEN)
    assert result == "сообщение отправлено в чат 73061220"


@pytest.mark.asyncio
async def test_failure_raises_instead_of_reporting_success(monkeypatch) -> None:
    notify = AsyncMock(side_effect=RuntimeError("503"))
    monkeypatch.setattr(notifier_module, "notify_user", notify)

    with pytest.raises(RuntimeError, match="бот недоступен"):
        await deliver_to_bot({"chat_id": "73061220", "text": "x"}, BOT_URL, TOKEN)


@pytest.mark.asyncio
async def test_non_numeric_chat_id_is_a_delivery_error(monkeypatch) -> None:
    """chat_id приходит от LLM: нечисловой не должен доходить до HTTP-вызова."""
    notify = AsyncMock()
    monkeypatch.setattr(notifier_module, "notify_user", notify)

    with pytest.raises(RuntimeError, match="бот недоступен"):
        await deliver_to_bot({"chat_id": "ivanov", "text": "x"}, BOT_URL, TOKEN)
    notify.assert_not_awaited()


@pytest.mark.asyncio
async def test_empty_chat_id_is_a_delivery_error(monkeypatch) -> None:
    notify = AsyncMock()
    monkeypatch.setattr(notifier_module, "notify_user", notify)

    with pytest.raises(RuntimeError, match="бот недоступен"):
        await deliver_to_bot({"text": "x"}, BOT_URL, TOKEN)
    notify.assert_not_awaited()


# ── отправка документа вложением ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_delivers_document_as_multipart(monkeypatch, tmp_path) -> None:
    """Байты файла уходят боту телом запроса: корпус смонтирован в app,
    а в контейнер бота — нет, поэтому путь передать нельзя."""
    doc = tmp_path / "ПС Тарифы. Технические требования. Версия 2.4.docx"
    doc.write_bytes(b"PK\x03\x04 fake docx")

    captured: dict = {}

    class _Response:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

    class _Client:
        def __init__(self, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, url, **kwargs):
            captured["url"] = url
            captured.update(kwargs)
            return _Response()

    monkeypatch.setattr(notifier_module.httpx, "AsyncClient", _Client)

    result = await notifier_module.deliver_document_to_bot(
        {"chat_id": "111222333", "path": str(doc), "caption": "Выжимка"},
        BOT_URL,
        TOKEN,
    )

    assert captured["url"] == f"{BOT_URL}/notify/document"
    assert captured["data"] == {"chat_id": "111222333", "caption": "Выжимка"}
    assert captured["files"]["file"][0] == doc.name
    assert captured["files"]["file"][1] == b"PK\x03\x04 fake docx"
    assert captured["headers"]["X-Internal-Token"] == TOKEN
    assert "ПС Тарифы" in result


@pytest.mark.asyncio
async def test_missing_file_is_a_delivery_error(monkeypatch, tmp_path) -> None:
    """Файла нет на диске — честная ошибка, а не «отправлено»."""
    with pytest.raises(RuntimeError, match="файл недоступен"):
        await notifier_module.deliver_document_to_bot(
            {"chat_id": "111222333", "path": str(tmp_path / "нет.docx")},
            BOT_URL,
            TOKEN,
        )
