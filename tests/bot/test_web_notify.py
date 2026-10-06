"""Тесты HTTP-канала app → бот: `POST /notify` и `POST /notify/document`.

Bot подменён моком: проверяется контракт ручек (токен, передача файла вложением,
понятные коды на ошибки Telegram), а не обмен с Telegram.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from aiogram.exceptions import TelegramBadRequest, TelegramForbiddenError
from fastapi.testclient import TestClient

from bot.web import MAX_DOCUMENT_BYTES, build_api

TOKEN = "internal-secret"
HEADERS = {"X-Internal-Token": TOKEN}


def _client(bot: MagicMock | AsyncMock) -> TestClient:
    bot.send_message = AsyncMock()
    bot.send_document = AsyncMock()
    return TestClient(build_api(bot, TOKEN))


def _forbidden() -> TelegramForbiddenError:
    return TelegramForbiddenError(
        method=MagicMock(), message="bot was blocked by the user"
    )


def test_notify_requires_token() -> None:
    client = _client(MagicMock())
    resp = client.post("/notify", json={"chat_id": 1, "text": "привет"})
    assert resp.status_code == 422 or resp.status_code == 401  # заголовок обязателен


def test_notify_rejects_wrong_token() -> None:
    client = _client(MagicMock())
    resp = client.post(
        "/notify",
        json={"chat_id": 1, "text": "привет"},
        headers={"X-Internal-Token": "wrong-token"},
    )
    assert resp.status_code == 401


def test_notify_sends_text() -> None:
    bot = MagicMock()
    client = _client(bot)
    resp = client.post(
        "/notify", json={"chat_id": 111, "text": "отчёт"}, headers=HEADERS
    )
    assert resp.status_code == 200
    bot.send_message.assert_awaited_once_with(chat_id=111, text="отчёт")


def test_notify_document_sends_attachment() -> None:
    bot = MagicMock()
    client = _client(bot)
    resp = client.post(
        "/notify/document",
        data={"chat_id": "111", "caption": "Выжимка"},
        files={"file": ("ПС Тарифы. Версия 2.4.docx", b"PK\x03\x04docx",
                        "application/octet-stream")},
        headers=HEADERS,
    )

    assert resp.status_code == 200
    assert resp.json()["file"] == "ПС Тарифы. Версия 2.4.docx"
    kwargs = bot.send_document.call_args.kwargs
    assert kwargs["chat_id"] == 111
    assert kwargs["caption"] == "Выжимка"
    assert kwargs["document"].filename == "ПС Тарифы. Версия 2.4.docx"


def test_notify_document_without_caption_sends_none() -> None:
    """Пустая подпись → Telegram не должен получать пустую строку."""
    bot = MagicMock()
    client = _client(bot)
    client.post(
        "/notify/document",
        data={"chat_id": "111", "caption": ""},
        files={"file": ("a.txt", b"x", "text/plain")},
        headers=HEADERS,
    )
    assert bot.send_document.call_args.kwargs["caption"] is None


def test_notify_document_rejects_too_large_file() -> None:
    bot = MagicMock()
    client = _client(bot)
    resp = client.post(
        "/notify/document",
        data={"chat_id": "111", "caption": ""},
        files={"file": ("big.pdf", b"x" * (MAX_DOCUMENT_BYTES + 1), "application/pdf")},
        headers=HEADERS,
    )
    assert resp.status_code == 413
    bot.send_document.assert_not_awaited()


def test_notify_document_blocked_by_user_returns_400() -> None:
    bot = MagicMock()
    bot.send_document = AsyncMock(side_effect=_forbidden())
    client = TestClient(build_api(bot, TOKEN))
    resp = client.post(
        "/notify/document",
        data={"chat_id": "111", "caption": ""},
        files={"file": ("a.txt", b"x", "text/plain")},
        headers=HEADERS,
    )
    assert resp.status_code == 400


@pytest.mark.parametrize("endpoint", ["/notify", "/notify/document"])
def test_telegram_api_error_returns_502(endpoint: str) -> None:
    bot = MagicMock()
    bot.send_message = AsyncMock(
        side_effect=TelegramBadRequest(method=MagicMock(), message="bad request")
    )
    bot.send_document = AsyncMock(
        side_effect=TelegramBadRequest(method=MagicMock(), message="bad request")
    )
    client = TestClient(build_api(bot, TOKEN))
    if endpoint == "/notify":
        resp = client.post(endpoint, json={"chat_id": 1, "text": "x"}, headers=HEADERS)
    else:
        resp = client.post(
            endpoint,
            data={"chat_id": "1", "caption": ""},
            files={"file": ("a.txt", b"x", "text/plain")},
            headers=HEADERS,
        )
    assert resp.status_code == 502
