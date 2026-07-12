"""Tests for admin API endpoints: /stats, /users, /broadcast, /export, /handoff."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.admin.schemas import StatsOut, UserOut


# ── helpers ──────────────────────────────────────────────────────────────

ADMIN_TOKEN = "test-admin-secret"


def _build_app(*, with_session: bool = True) -> FastAPI:
    """Create test FastAPI app with admin router and required state."""
    from app.admin.routes import router as admin_router

    app = FastAPI()
    app.include_router(admin_router)
    if with_session:
        app.state.session_factory = MagicMock()
    return app


def _mock_settings():
    """Return a mock Settings object."""
    s = MagicMock()
    s.admin_token = SecretStr(ADMIN_TOKEN)
    s.bot_url = "http://bot:9000"
    s.internal_token = SecretStr("internal-secret")
    return s


def _auth_headers() -> dict[str, str]:
    return {"X-Admin-Token": ADMIN_TOKEN}


# ── require_admin guard ──────────────────────────────────────────────────

@pytest.mark.anyio
async def test_stats_no_token_returns_403():
    """Without X-Admin-Token → 403."""
    app = _build_app()
    with patch("app.admin.routes.get_settings", return_value=_mock_settings()):
        client = TestClient(app)
        resp = client.get("/chats/admin/stats")
        assert resp.status_code == 403


@pytest.mark.anyio
async def test_stats_wrong_token_returns_403():
    """Wrong token → 403."""
    app = _build_app()
    with patch("app.admin.routes.get_settings", return_value=_mock_settings()):
        client = TestClient(app)
        resp = client.get(
            "/chats/admin/stats",
            headers={"X-Admin-Token": "wrong-token"},
        )
        assert resp.status_code == 403


# ── GET /stats ───────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_stats_valid_token_returns_200():
    """Valid admin token → 200 with stats JSON."""
    app = _build_app()
    mock_repo = MagicMock()
    mock_repo.compute_stats = AsyncMock(return_value=StatsOut(
        total_messages=100, active_users=10,
        avg_latency_ms=250.5, moderation_block_rate=0.02, feedback_ratio=0.85,
    ))

    with (
        patch("app.admin.routes.get_settings", return_value=_mock_settings()),
        patch("app.admin.routes.AdminRepository", return_value=mock_repo),
    ):
        client = TestClient(app)
        resp = client.get("/chats/admin/stats", headers=_auth_headers())

    assert resp.status_code == 200
    data = resp.json()
    assert data["total_messages"] == 100
    assert data["active_users"] == 10
    assert data["avg_latency_ms"] == 250.5
    assert data["moderation_block_rate"] == 0.02
    assert data["feedback_ratio"] == 0.85


@pytest.mark.anyio
async def test_stats_custom_window():
    """window_hours query parameter is passed through."""
    app = _build_app()
    mock_repo = MagicMock()
    mock_repo.compute_stats = AsyncMock(return_value=StatsOut(
        total_messages=0, active_users=0,
    ))

    with (
        patch("app.admin.routes.get_settings", return_value=_mock_settings()),
        patch("app.admin.routes.AdminRepository", return_value=mock_repo),
    ):
        client = TestClient(app)
        resp = client.get("/chats/admin/stats?window_hours=48", headers=_auth_headers())
        assert resp.status_code == 200
        mock_repo.compute_stats.assert_awaited_once_with(window_hours=48)


# ── GET /users ───────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_users_returns_list():
    """GET /users returns list of UserOut."""
    app = _build_app()
    mock_repo = MagicMock()
    mock_repo.list_users = AsyncMock(return_value=[
        UserOut(owner_external_id="123", interface="telegram",
                last_seen_at="2026-07-12T12:00:00+00:00"),
        UserOut(owner_external_id="456", interface="telegram",
                last_seen_at="2026-07-12T11:00:00+00:00"),
    ])

    with (
        patch("app.admin.routes.get_settings", return_value=_mock_settings()),
        patch("app.admin.routes.AdminRepository", return_value=mock_repo),
    ):
        client = TestClient(app)
        resp = client.get("/chats/admin/users?limit=10", headers=_auth_headers())

    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 2
    assert data[0]["owner_external_id"] == "123"


@pytest.mark.anyio
async def test_users_empty():
    """No users → empty list."""
    app = _build_app()
    mock_repo = MagicMock()
    mock_repo.list_users = AsyncMock(return_value=[])

    with (
        patch("app.admin.routes.get_settings", return_value=_mock_settings()),
        patch("app.admin.routes.AdminRepository", return_value=mock_repo),
    ):
        client = TestClient(app)
        resp = client.get("/chats/admin/users", headers=_auth_headers())

    assert resp.status_code == 200
    assert resp.json() == []


# ── POST /broadcast ──────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_broadcast_with_interface_filter():
    """Broadcast with interface_filter → queued."""
    app = _build_app()

    with (
        patch("app.admin.routes.get_settings", return_value=_mock_settings()),
        patch("app.admin.routes.enqueue_broadcast") as mock_enqueue,
    ):
        mock_enqueue.return_value = {"id": 5, "status": "pending", "total": 3}

        client = TestClient(app)
        resp = client.post(
            "/chats/admin/broadcast",
            json={"text": "hello", "interface_filter": "telegram"},
            headers=_auth_headers(),
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["sent"] == 0
    assert "queued" in data.get("detail", "")


@pytest.mark.anyio
async def test_broadcast_with_owner_ids():
    """Broadcast with explicit owner_ids → synchronous send."""
    app = _build_app()

    with (
        patch("app.admin.routes.get_settings", return_value=_mock_settings()),
        patch("app.admin.routes.broadcast_sync") as mock_sync,
    ):
        mock_sync.return_value = {"sent": 2, "failed": 0}

        client = TestClient(app)
        resp = client.post(
            "/chats/admin/broadcast",
            json={"text": "hello", "owner_ids": [111, 222]},
            headers=_auth_headers(),
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["sent"] == 2


@pytest.mark.anyio
async def test_broadcast_no_recipients_returns_400():
    """Neither owner_ids nor interface_filter → 400."""
    app = _build_app()

    with patch("app.admin.routes.get_settings", return_value=_mock_settings()):
        client = TestClient(app)
        resp = client.post(
            "/chats/admin/broadcast",
            json={"text": "hello"},
            headers=_auth_headers(),
        )

    assert resp.status_code == 400


# ── POST /handoff ────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_handoff_paused_for_human():
    """Handoff with paused_for_human status → 200 + alert + notify."""
    app = _build_app()

    with (
        patch("app.admin.routes.get_settings", return_value=_mock_settings()),
        patch("app.admin.routes.set_handoff_status_by_owner") as mock_handoff,
        patch("app.admin.routes.fire_alert") as mock_alert,
        patch("app.admin.routes.notify_user") as mock_notify,
    ):
        mock_handoff.return_value = 1

        client = TestClient(app)
        resp = client.post(
            "/chats/admin/handoff",
            json={
                "owner_external_id": "123",
                "interface": "telegram",
                "status": "paused_for_human",
            },
            headers=_auth_headers(),
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    mock_alert.assert_awaited_once()
    mock_notify.assert_awaited_once()
