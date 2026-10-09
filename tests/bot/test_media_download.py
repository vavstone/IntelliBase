"""Тесты скачивания медиа из Telegram (bot/handlers/media.py).

Живой прогон 09.10: `TelegramNetworkError: ServerDisconnectedError` на
скачивании документа — без повтора сбой превращался в «Что-то пошло не так»
от глобального обработчика. Здесь проверяются ретрай и понятный ответ.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiogram.exceptions import TelegramNetworkError

from bot.handlers.media import _download_to_bytes, _safe_download


def _network_error() -> TelegramNetworkError:
    return TelegramNetworkError(method=None, message="Server disconnected")


def _bot(fail_times: int) -> SimpleNamespace:
    """Фейковый bot: первые `fail_times` вызовов get_file падают сетью."""
    bot = SimpleNamespace()
    file = SimpleNamespace(file_path="voice/file.ogg")
    side = [_network_error()] * fail_times + [file]
    bot.get_file = AsyncMock(side_effect=side)

    async def _download(file_path, destination):  # noqa: ANN001
        destination.write(b"audio-bytes")

    bot.download_file = AsyncMock(side_effect=_download)
    return bot


@pytest.mark.asyncio
async def test_download_retries_after_network_error(mocker) -> None:
    """Одиночный сетевой сбой — повтор, данные доходят."""
    mocker.patch("bot.handlers.media.asyncio.sleep", AsyncMock())
    bot = _bot(fail_times=1)

    data = await _download_to_bytes(bot, "fid")

    assert data == b"audio-bytes"
    assert bot.get_file.await_count == 2  # сбой + успешная попытка


@pytest.mark.asyncio
async def test_download_raises_after_all_attempts(mocker) -> None:
    mocker.patch("bot.handlers.media.asyncio.sleep", AsyncMock())
    bot = _bot(fail_times=2)

    with pytest.raises(TelegramNetworkError):
        await _download_to_bytes(bot, "fid")

    assert bot.get_file.await_count == 2


@pytest.mark.asyncio
async def test_safe_download_answers_user_on_failure(mocker) -> None:
    """После повторов пользователь получает понятное сообщение, не молчание."""
    mocker.patch("bot.handlers.media.asyncio.sleep", AsyncMock())
    message = SimpleNamespace(bot=_bot(fail_times=2), answer=AsyncMock())

    result = await _safe_download(message, "fid")

    assert result is None
    message.answer.assert_awaited_once()
    assert "скачать" in message.answer.await_args.args[0]


@pytest.mark.asyncio
async def test_safe_download_returns_data_without_errors(mocker) -> None:
    mocker.patch("bot.handlers.media.asyncio.sleep", AsyncMock())
    message = SimpleNamespace(bot=_bot(fail_times=0), answer=AsyncMock())

    assert await _safe_download(message, "fid") == b"audio-bytes"
    message.answer.assert_not_awaited()
