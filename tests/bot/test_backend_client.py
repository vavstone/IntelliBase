"""
Tests for BackendClient using httpx.MockTransport.

Verifies:
  - send_message correctly parses SSE frames (data: ...\\n\\n)
  - clear_messages sends DELETE to the right URL
  - get_or_create_chat returns a UUID
"""

import json
from uuid import UUID

import httpx
import pytest

from bot.services.backend_client import BackendClient


# ── helpers ──────────────────────────────────────────────────────────────

def _client(response_text: str, status_code: int = 200) -> BackendClient:
    """Create BackendClient with a MockTransport that returns `response_text`."""
    transport = httpx.MockTransport(
        lambda req: httpx.Response(status_code, text=response_text)
    )
    http = httpx.AsyncClient(transport=transport, base_url="http://test.local")
    return BackendClient(http)


def _sse_frame(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


# ── send_message SSE parsing ─────────────────────────────────────────────

@pytest.mark.anyio
async def test_send_message_parses_tokens_and_message_saved():
    """send_message yields token deltas and message_saved events."""
    frames = (
        _sse_frame({"type": "token", "delta": "Hello"})
        + _sse_frame({"type": "token", "delta": " world"})
        + _sse_frame({"type": "message_saved", "message_id": "uuid-123"})
        + _sse_frame({"type": "done"})
    )
    backend = _client(frames)

    events = [
        e async for e in backend.send_message(
            UUID("00000000-0000-0000-0000-000000000001"),
            "hi",
            owner_external_id="u1",
        )
    ]

    assert events == [
        {"type": "token", "delta": "Hello"},
        {"type": "token", "delta": " world"},
        {"type": "message_saved", "message_id": "uuid-123"},
    ]


@pytest.mark.anyio
async def test_send_message_stops_on_done():
    """send_message stops yielding when it sees type:done."""
    frames = (
        _sse_frame({"type": "token", "delta": "x"})
        + _sse_frame({"type": "done"})
        + _sse_frame({"type": "token", "delta": "SHOULD_NOT_APPEAR"})
    )
    backend = _client(frames)

    events = [
        e async for e in backend.send_message(
            UUID("00000000-0000-0000-0000-000000000002"),
            "hi",
            owner_external_id="u1",
        )
    ]

    deltas = [e["delta"] for e in events if e["type"] == "token"]
    assert deltas == ["x"]


@pytest.mark.anyio
async def test_send_message_handles_empty_stream():
    """Empty SSE stream — no events yielded."""
    backend = _client(_sse_frame({"type": "done"}))

    events = [
        e async for e in backend.send_message(
            UUID("00000000-0000-0000-0000-000000000003"),
            "hi",
            owner_external_id="u1",
        )
    ]

    assert events == []


@pytest.mark.anyio
async def test_send_message_ignores_garbage_lines():
    """Lines not starting with 'data: ' are skipped."""
    frames = (
        "some garbage\n"
        "\n"
        + _sse_frame({"type": "token", "delta": "ok"})
        + _sse_frame({"type": "done"})
    )
    backend = _client(frames)

    events = [
        e async for e in backend.send_message(
            UUID("00000000-0000-0000-0000-000000000004"),
            "hi",
            owner_external_id="u1",
        )
    ]

    assert events == [{"type": "token", "delta": "ok"}]


# ── clear_messages ───────────────────────────────────────────────────────

def _capturing_transport(response=None):
    """MockTransport that captures the last request into a mutable list."""
    captured = []
    if response is None:
        response = httpx.Response(200, json={"status": "ok"})

    def handler(req):
        captured.append(req)
        return response

    transport = httpx.MockTransport(handler)
    return transport, captured


@pytest.mark.anyio
async def test_clear_messages_sends_delete_with_owner_header():
    """clear_messages sends DELETE with X-Owner-External-Id header."""
    transport, captured = _capturing_transport()
    http = httpx.AsyncClient(transport=transport, base_url="http://test.local")
    backend = BackendClient(http)

    chat_id = UUID("a0000000-0000-0000-0000-000000000001")
    await backend.clear_messages(chat_id, owner_external_id="user-42")

    req = captured[0]
    assert req.method == "DELETE"
    assert req.url.path == f"/chats/{chat_id}/messages"
    assert req.headers["X-Owner-External-Id"] == "user-42"


@pytest.mark.anyio
async def test_clear_messages_without_owner_omits_header():
    """No owner = no X-Owner-External-Id header."""
    transport, captured = _capturing_transport()
    http = httpx.AsyncClient(transport=transport, base_url="http://test.local")
    backend = BackendClient(http)

    chat_id = UUID("b0000000-0000-0000-0000-000000000001")
    await backend.clear_messages(chat_id)

    req = captured[0]
    assert "x-owner-external-id" not in req.headers


# ── get_or_create_chat ───────────────────────────────────────────────────

@pytest.mark.anyio
async def test_get_or_create_chat_returns_uuid():
    """get_or_create_chat returns a valid UUID from JSON response."""
    chat_id = "550e8400-e29b-41d4-a716-446655440000"
    transport = httpx.MockTransport(
        lambda req: httpx.Response(200, json={"chat_id": chat_id})
    )
    http = httpx.AsyncClient(transport=transport, base_url="http://test.local")
    backend = BackendClient(http)

    result = await backend.get_or_create_chat("user-1", "telegram")

    assert result == UUID(chat_id)


@pytest.mark.anyio
async def test_get_or_create_chat_sends_correct_body():
    """get_or_create_chat POSTs owner_external_id and interface."""
    transport, captured = _capturing_transport(
        httpx.Response(200, json={"chat_id": "00000000-0000-0000-0000-000000000001"})
    )
    http = httpx.AsyncClient(transport=transport, base_url="http://test.local")
    backend = BackendClient(http)

    await backend.get_or_create_chat("user-99", "telegram")

    req = captured[0]
    body = json.loads(req.content)
    assert req.method == "POST"
    assert req.url.path == "/chats"
    assert body["owner_external_id"] == "user-99"
    assert body["interface"] == "telegram"
    assert req.headers["X-Owner-External-Id"] == "user-99"


@pytest.mark.anyio
async def test_get_or_create_chat_raises_on_http_error():
    """HTTP errors are propagated."""
    transport = httpx.MockTransport(
        lambda req: httpx.Response(500, json={"detail": "boom"})
    )
    http = httpx.AsyncClient(transport=transport, base_url="http://test.local")
    backend = BackendClient(http)

    with pytest.raises(httpx.HTTPStatusError):
        await backend.get_or_create_chat("u", "telegram")
