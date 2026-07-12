"""
Tests for app.chat.media — PDF, PNG, JPEG → content-part conversion.

No network LLM calls needed — image parts use local base64 encoding,
PDF extraction uses local pypdf.
"""

import base64
from io import BytesIO
from unittest.mock import MagicMock

import pytest
from fastapi import UploadFile
from pypdf import PdfWriter


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


def _minimal_pdf_bytes() -> bytes:
    """Generate a minimal valid PDF (1 blank page) via pypdf."""
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    buf = BytesIO()
    writer.write(buf)
    return buf.getvalue()


# ── test images generated via Pillow ────────────────────────────────────


def _make_png_bytes() -> bytes:
    """Generate a 1×1 red PNG."""
    from PIL import Image

    img = Image.new("RGB", (1, 1), color=(255, 0, 0))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_jpeg_bytes() -> bytes:
    """Generate a 1×1 blue JPEG."""
    from PIL import Image

    img = Image.new("RGB", (1, 1), color=(0, 0, 255))
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# ── PDF tests ───────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_media_to_part_pdf_returns_text_part():
    """PDF → text-part with [документ PDF]: prefix."""
    from app.chat.media import media_to_part

    pdf_data = _minimal_pdf_bytes()
    media = _fake_upload_file("application/pdf", pdf_data, "doc.pdf")
    llm_client = MagicMock()  # not used for PDF

    result = await media_to_part(media, llm_client)

    assert result["type"] == "text"
    assert result["text"].startswith("[документ PDF]:\n")
    # The blank page has no text, so content after prefix is empty or minimal
    assert isinstance(result["text"], str)


@pytest.mark.anyio
async def test_extract_pdf_text_reads_pages():
    """extract_pdf_text parses a real (blank) PDF without crashing."""
    from app.chat.media import extract_pdf_text

    pdf_data = _minimal_pdf_bytes()
    text = extract_pdf_text(pdf_data)
    # blank page returns empty string — that's valid behaviour
    assert text == ""


@pytest.mark.anyio
async def test_extract_pdf_text_scan_detection():
    """Multi-page PDF with little text triggers scan stub."""
    from app.chat.media import extract_pdf_text

    writer = PdfWriter()
    for _ in range(5):
        writer.add_blank_page(width=612, height=792)
    buf = BytesIO()
    writer.write(buf)
    pdf_data = buf.getvalue()

    text = extract_pdf_text(pdf_data)
    # ≥5 pages with <100 chars → scan stub
    assert "скан" in text.lower() or "ocr" in text.lower()


# ── image tests ─────────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_media_to_part_png_returns_image_url_with_data_uri():
    """PNG → image_url-part with correct data: URI."""
    from app.chat.media import media_to_part

    png_data = _make_png_bytes()
    media = _fake_upload_file("image/png", png_data, "img.png")
    llm_client = MagicMock()

    result = await media_to_part(media, llm_client)

    assert result["type"] == "image_url"
    url = result["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")

    # Round-trip: decode the base64 and compare
    prefix = "data:image/png;base64,"
    decoded = base64.b64decode(url[len(prefix):])
    assert decoded == png_data


@pytest.mark.anyio
async def test_media_to_part_jpeg_returns_image_url_with_data_uri():
    """JPEG → image_url-part with correct data: URI."""
    from app.chat.media import media_to_part

    jpeg_data = _make_jpeg_bytes()
    media = _fake_upload_file("image/jpeg", jpeg_data, "img.jpeg")
    llm_client = MagicMock()

    result = await media_to_part(media, llm_client)

    assert result["type"] == "image_url"
    url = result["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,")

    prefix = "data:image/jpeg;base64,"
    decoded = base64.b64decode(url[len(prefix):])
    assert decoded == jpeg_data


@pytest.mark.anyio
async def test_media_to_part_unsupported_type_raises():
    """Unsupported MIME type raises ValueError."""
    from app.chat.media import media_to_part

    media = _fake_upload_file("application/zip", b"dummy", "file.zip")
    llm_client = MagicMock()

    with pytest.raises(ValueError, match="Unsupported media type"):
        await media_to_part(media, llm_client)
