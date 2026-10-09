"""
Tests for app.chat.media — audio/ogg → Whisper transcription → text-part.

Uses `unittest.mock.AsyncMock` to stub `client.audio.transcriptions.create`
so no network calls are made.
"""

from io import BytesIO
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import UploadFile


# ── helpers ──────────────────────────────────────────────────────────────

def _fake_upload_file(
    content_type: str, data: bytes, filename: str = "test"
) -> UploadFile:
    """Create a real FastAPI UploadFile wrapping in-memory bytes."""
    return UploadFile(
        filename=filename,
        file=BytesIO(data),
        headers={"content-type": content_type},
    )


def _make_mock_llm(transcript_text: str) -> MagicMock:
    """Build a mock AsyncOpenAI whose audio.transcriptions.create returns
    an object with `.text` attribute."""
    # The result of transcriptions.create
    transcript_result = MagicMock()
    transcript_result.text = transcript_text

    # transcriptions.create is an AsyncMock returning transcript_result
    transcriptions_create = AsyncMock(return_value=transcript_result)

    # transcriptions
    transcriptions = MagicMock()
    transcriptions.create = transcriptions_create

    # audio
    audio = MagicMock()
    audio.transcriptions = transcriptions

    # top-level client
    llm_client = MagicMock()
    llm_client.audio = audio

    return llm_client


# ── audio/ogg test ──────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_media_to_part_audio_ogg_returns_text_with_prefix():
    """audio/ogg → Whisper stub → text-part with [пользователь сказал голосом]:."""
    from app.chat.media import media_to_part

    fake_audio = b"\x00" * 100  # dummy ogg content
    media = _fake_upload_file("audio/ogg", fake_audio, "voice.ogg")
    llm_client = _make_mock_llm("привет мир")

    # Аудио идёт через transcription_client (OpenAI-клиент), а не через клиент
    # провайдера чата: у Ollama/DeepSeek нет /audio/transcriptions.
    result = await media_to_part(media, llm_client, llm_client)

    assert result["type"] == "text"
    assert result["text"] == "[пользователь сказал голосом]:\nпривет мир"

    # Verify the mock was called with expected model
    llm_client.audio.transcriptions.create.assert_awaited_once()
    call_kwargs = llm_client.audio.transcriptions.create.call_args.kwargs
    assert call_kwargs["model"] == "whisper-1"
    assert call_kwargs["file"] is not None


@pytest.mark.anyio
async def test_media_to_part_audio_ogg_missing_content_type():
    """audio/ogg as application/ogg (Telegram voice) also routed to Whisper."""
    from app.chat.media import media_to_part

    fake_audio = b"\x00" * 50
    media = _fake_upload_file("application/ogg", fake_audio, "voice.ogg")
    llm_client = _make_mock_llm("тест")

    result = await media_to_part(media, llm_client, llm_client)

    assert result["type"] == "text"
    assert result["text"] == "[пользователь сказал голосом]:\nтест"


@pytest.mark.anyio
async def test_audio_without_openai_client_gets_clear_error():
    """Без OpenAI-клиента голосовое не летит чужому провайдеру (404), а даёт понятный отказ."""
    from app.chat.media import media_to_part

    media = _fake_upload_file("audio/ogg", b"\x00" * 50, "voice.ogg")
    llm_client = _make_mock_llm("не должен вызываться")

    with pytest.raises(ValueError, match="OpenAI"):
        await media_to_part(media, llm_client)  # transcription_client не передан

    llm_client.audio.transcriptions.create.assert_not_awaited()


# ── имя файла для Whisper ───────────────────────────────────────────────


def test_audio_filename_replaces_bot_placeholder():
    """Бот шлёт вложение как file.bin — Whisper по расширению не поймёт формат.

    Живой прогон 09.10: Telegram-голосовое (ogg) получало от OpenAI
    «400 Invalid file format», потому что имя было file.bin.
    """
    from app.chat.media import _audio_filename

    assert _audio_filename("file.bin", "audio/ogg") == "voice.ogg"
    assert _audio_filename("file.bin", "audio/mpeg") == "voice.mp3"
    assert _audio_filename("file.bin", "audio/x-m4a") == "voice.m4a"
    # Незнакомый MIME — безопасный дефолт вместо .bin.
    assert _audio_filename("file.bin", "application/octet-stream") == "voice.ogg"


def test_audio_filename_keeps_real_audio_names():
    """Имя с аудио-расширением не трогаем — это прямой вызов (тесты, API)."""
    from app.chat.media import _audio_filename

    assert _audio_filename("recording.m4a", "audio/ogg") == "recording.m4a"
    assert _audio_filename("notes.WAV", "audio/ogg") == "notes.WAV"


@pytest.mark.anyio
async def test_media_to_part_bot_voice_uses_ogg_name():
    """End-to-end: голосовое от бота (file.bin) уходит в Whisper как voice.ogg."""
    from app.chat.media import media_to_part

    media = _fake_upload_file("audio/ogg", b"\x00" * 50, "file.bin")
    llm_client = _make_mock_llm("привет")

    await media_to_part(media, llm_client, llm_client)

    call_kwargs = llm_client.audio.transcriptions.create.call_args.kwargs
    assert call_kwargs["file"].name == "voice.ogg"


@pytest.mark.anyio
async def test_whisper_transcribe_uses_filename():
    """whisper_transcribe passes BytesIO with .name set from filename."""
    from app.chat.media import whisper_transcribe

    audio_data = b"\x01\x02\x03"
    llm_client = _make_mock_llm("ok")

    result = await whisper_transcribe(audio_data, "recording.ogg", llm_client)

    assert result == "ok"

    # Check that file passed to create has correct .name
    call_args = llm_client.audio.transcriptions.create.call_args
    file_arg = call_args.kwargs["file"]
    assert file_arg.name == "recording.ogg"
    assert file_arg.read() == audio_data
