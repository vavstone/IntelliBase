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

    result = await media_to_part(media, llm_client)

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

    result = await media_to_part(media, llm_client)

    assert result["type"] == "text"
    assert result["text"] == "[пользователь сказал голосом]:\nтест"


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
