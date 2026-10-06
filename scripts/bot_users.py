"""Разрешённые пользователи бота: список, выдача и отзыв доступа.

Ходит в админ-API бэкенда (`/chats/admin/bot-users`), а не в БД напрямую:
только этот путь пишет автора изменения (`created_by`), проверяет длину полей и
не требует держать на хосте пароль Postgres. Бот закрыт: пока пользователя нет
в списке, он не получит ответа ни на одну команду (кроме подсказки со своим id).

Запуск (стек поднят):
    uv run python scripts/bot_users.py list
    uv run python scripts/bot_users.py add 123456789 "Иванов Пётр, руководитель"
    uv run python scripts/bot_users.py remove 123456789
    uv run python scripts/bot_users.py check 123456789

Через Makefile:
    make users ARGS="list"
    make users ARGS='add 123456789 "Иванов Пётр"'

Адрес бэкенда берётся из `BACKEND_URL` (в `.env` — http://127.0.0.1:8000),
токен — из `ADMIN_TOKEN`. Проверка доступа (`check`) — тот же вызов, что делает
бот на каждом апдейте: `GET /access/{chat_id}` с `X-Internal-Token`.

Если сервис лежит и доступ нужно выдать «руками», прямой SQL (значения подставить
свои; пароль — POSTGRES_PASSWORD из .env):

    docker compose exec db psql -U chat -d intellibase -c "
      INSERT INTO bot_users (chat_id, title, created_by)
      VALUES ('123456789', 'Иванов Пётр', 'manual')
      ON CONFLICT (chat_id) DO UPDATE
        SET title = EXCLUDED.title, is_active = TRUE;"
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import httpx

TIMEOUT = 20.0


def load_env(path: Path) -> None:
    """Подхватывает `.env`, не затирая уже заданное окружение (как в smoke.py)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip().strip('"').strip("'")
        value = value.split(" #", 1)[0].strip()
        os.environ.setdefault(key.strip(), value)


def _client() -> tuple[httpx.Client, str]:
    """Клиент к бэкенду + админ-токен. Понятная ошибка, если токена нет."""
    base = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
    token = os.environ.get("ADMIN_TOKEN", "")
    if not token:
        sys.exit(
            "ADMIN_TOKEN не задан: заполните его в .env "
            "(или экспортируйте переменную) — без него админ-API недоступен"
        )
    return httpx.Client(base_url=base, timeout=TIMEOUT), token


def _request(client: httpx.Client, token: str, method: str, url: str, **kwargs):
    """Запрос к админ-API с человекочитаемой диагностикой на ошибках."""
    try:
        r = client.request(method, url, headers={"X-Admin-Token": token}, **kwargs)
    except httpx.RequestError as exc:
        sys.exit(f"Бэкенд недоступен ({client.base_url}): {exc}")
    if r.status_code == 403:
        sys.exit("403: ADMIN_TOKEN не совпадает с настройками бэкенда")
    if r.status_code >= 400:
        sys.exit(f"{r.status_code}: {r.text[:300]}")
    return r


def cmd_list(client: httpx.Client, token: str, args: argparse.Namespace) -> None:
    params = {} if args.all else {"active_only": "true"}
    items = _request(client, token, "GET", "/chats/admin/bot-users", params=params).json()
    if not items:
        print("Список пуст — бот не отвечает никому, кроме BOT_ALLOWED_CHAT_IDS.")
        return
    print(f"{'chat_id':<14} {'активен':<8} имя")
    for u in items:
        mark = "да" if u["is_active"] else "нет"
        print(f"{u['chat_id']:<14} {mark:<8} {u['title']}")
    if args.all:
        inactive = [u["chat_id"] for u in items if not u["is_active"]]
        if inactive:
            print(f"\nОтозван доступ ({len(inactive)}): {', '.join(inactive)}")


def cmd_add(client: httpx.Client, token: str, args: argparse.Namespace) -> None:
    r = _request(
        client,
        token,
        "POST",
        "/chats/admin/bot-users",
        json={"chat_id": args.chat_id, "title": args.title},
    )
    u = r.json()
    print(f"Доступ выдан: {u['chat_id']} — {u['title']} (автор: {u['created_by']})")
    print("Пользователь может писать боту; агент может ему отправлять.")


def cmd_remove(client: httpx.Client, token: str, args: argparse.Namespace) -> None:
    r = _request(
        client, token, "DELETE", f"/chats/admin/bot-users/{args.chat_id}"
    )
    if r.json().get("deactivated"):
        print(f"Доступ отозван: {args.chat_id}")
    else:
        print(f"{args.chat_id}: активной записи не было (уже отозван или не добавлен)")


def cmd_check(client: httpx.Client, token: str, args: argparse.Namespace) -> None:
    """Что ответит бэкенд боту на проверке доступа (без Telegram)."""
    internal = os.environ.get("INTERNAL_TOKEN", "")
    base = str(client.base_url)
    try:
        r = httpx.get(
            f"{base}/access/{args.chat_id}",
            headers={"X-Internal-Token": internal},
            timeout=TIMEOUT,
        )
    except httpx.RequestError as exc:
        sys.exit(f"Бэкенд недоступен ({base}): {exc}")
    if r.status_code == 403:
        sys.exit("403: INTERNAL_TOKEN не совпадает с настройками бэкенда")
    if r.status_code >= 400:
        sys.exit(f"{r.status_code}: {r.text[:300]}")
    allowed = r.json().get("allowed")
    print(f"{args.chat_id}: {'доступ есть' if allowed else 'доступа нет'}")
    if not allowed:
        print("Бот ответит этому пользователю отказом и покажет его id.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Разрешённые пользователи бота (таблица bot_users через админ-API)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="показать список доступа")
    p_list.add_argument(
        "--all", action="store_true", help="включая записи с отозванным доступом"
    )
    p_list.set_defaults(func=cmd_list)

    p_add = sub.add_parser("add", help="выдать доступ (upsert по chat_id)")
    p_add.add_argument("chat_id", help="Telegram chat_id пользователя")
    p_add.add_argument(
        "title", help="имя для адресной книги: «Иванов Пётр, руководитель отдела»"
    )
    p_add.set_defaults(func=cmd_add)

    p_rm = sub.add_parser("remove", help="отозвать доступ (мягко)")
    p_rm.add_argument("chat_id")
    p_rm.set_defaults(func=cmd_remove)

    p_check = sub.add_parser("check", help="проверить, что ответит бот (GET /access)")
    p_check.add_argument("chat_id")
    p_check.set_defaults(func=cmd_check)

    return parser


def main() -> None:
    load_env(Path(__file__).resolve().parent.parent / ".env")
    args = build_parser().parse_args()
    client, token = _client()
    try:
        args.func(client, token, args)
    finally:
        client.close()


if __name__ == "__main__":
    main()
