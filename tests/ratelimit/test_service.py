"""Tests for rate-limit: atomic counter + enforce_rate_limit dependency."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI, Depends
from fastapi.testclient import TestClient


# ── increment_and_check ──────────────────────────────────────────────────

@pytest.mark.anyio
async def test_increment_first_request_allowed():
    """First request in a bucket is always allowed."""
    mock_session = AsyncMock()
    mock_session.execute = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalar.return_value = 1
    mock_session.execute.return_value = mock_result

    from app.ratelimit.service import increment_and_check
    allowed, count = await increment_and_check(
        mock_session, "user1", "message", limit=15,
    )

    assert allowed is True
    assert count == 1
    mock_session.commit.assert_awaited_once()


@pytest.mark.anyio
async def test_increment_exceeds_limit():
    """When count > limit → blocked."""
    mock_session = AsyncMock()
    mock_session.execute = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalar.return_value = 16
    mock_session.execute.return_value = mock_result

    from app.ratelimit.service import increment_and_check
    allowed, count = await increment_and_check(
        mock_session, "user1", "message", limit=15,
    )

    assert allowed is False
    assert count == 16


@pytest.mark.anyio
async def test_increment_at_limit():
    """Count == limit → still allowed."""
    mock_session = AsyncMock()
    mock_session.execute = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalar.return_value = 15
    mock_session.execute.return_value = mock_result

    from app.ratelimit.service import increment_and_check
    allowed, count = await increment_and_check(
        mock_session, "user1", "message", limit=15,
    )

    assert allowed is True
    assert count == 15


# ── enforce_rate_limit dependency ────────────────────────────────────────

@pytest.mark.anyio
async def test_enforce_rate_limit_no_header_passes():
    """Without X-Owner-External-Id header → silently passes."""
    from app.ratelimit.dependencies import enforce_rate_limit

    dep = enforce_rate_limit("message", 15)
    app = FastAPI()

    @app.get("/test", dependencies=[Depends(dep)])
    async def test_route():
        return {"ok": True}

    client = TestClient(app)
    resp = client.get("/test")
    assert resp.status_code == 200


@pytest.mark.anyio
async def test_enforce_rate_limit_no_session_factory_passes():
    """Without session_factory → silently passes."""
    from app.ratelimit.dependencies import enforce_rate_limit

    dep = enforce_rate_limit("message", 15)
    app = FastAPI()

    @app.get("/test", dependencies=[Depends(dep)])
    async def test_route():
        return {"ok": True}

    client = TestClient(app)
    resp = client.get("/test", headers={"X-Owner-External-Id": "user1"})
    assert resp.status_code == 200


@pytest.mark.anyio
async def test_enforce_rate_limit_within_budget():
    """Within limit → 200."""
    from app.ratelimit.dependencies import enforce_rate_limit

    dep = enforce_rate_limit("message", 5)

    patched_increment = AsyncMock(return_value=(True, 3))
    app = FastAPI()
    app.state.session_factory = MagicMock()

    with patch(
        "app.ratelimit.dependencies.increment_and_check",
        patched_increment,
    ):
        @app.get("/test", dependencies=[Depends(dep)])
        async def test_route():
            return {"ok": True}

        client = TestClient(app)
        resp = client.get("/test", headers={"X-Owner-External-Id": "user1"})
        assert resp.status_code == 200


@pytest.mark.anyio
async def test_enforce_rate_limit_exceeded():
    """Over limit → 429 with Retry-After header."""
    from app.ratelimit.dependencies import enforce_rate_limit

    dep = enforce_rate_limit("message", 5)

    patched_increment = AsyncMock(return_value=(False, 6))
    app = FastAPI()
    app.state.session_factory = MagicMock()

    with patch(
        "app.ratelimit.dependencies.increment_and_check",
        patched_increment,
    ):
        @app.get("/test", dependencies=[Depends(dep)])
        async def test_route():
            return {"ok": True}

        client = TestClient(app)
        resp = client.get("/test", headers={"X-Owner-External-Id": "user1"})

    assert resp.status_code == 429
    assert "Retry-After" in resp.headers
    detail = resp.json()["detail"]
    assert detail["code"] == "rate_limit"
