"""Тесты HTTP-заглушки бота — запуск без BOT_TOKEN (см. bot/web.py).

Нужна для сценария «чистый клон»: в .env.example токен пустой, и контейнер
бота не должен уходить в рестарт-луп — /health отвечает, /notify отдаёт 503.
"""

from fastapi.testclient import TestClient

from bot.web import build_disabled_api

REASON = "BOT_TOKEN не задан"


def _client() -> TestClient:
    return TestClient(build_disabled_api(REASON, "secret"))


def test_health_ok_without_token():
    """/health отвечает 200 — healthcheck compose проходит."""
    resp = _client().get("/health")

    assert resp.status_code == 200
    assert resp.json()["bot"] == "disabled"


def test_notify_rejects_foreign_token():
    """Чужой X-Internal-Token отбивается до проверки режима."""
    resp = _client().post(
        "/notify",
        json={"chat_id": 1, "text": "привет"},
        headers={"X-Internal-Token": "wrong"},
    )

    assert resp.status_code == 401


def test_notify_returns_503_when_disabled():
    """Свой токен, но бот выключен — честный 503 с причиной."""
    resp = _client().post(
        "/notify",
        json={"chat_id": 1, "text": "привет"},
        headers={"X-Internal-Token": "secret"},
    )

    assert resp.status_code == 503
    assert REASON in resp.json()["detail"]
