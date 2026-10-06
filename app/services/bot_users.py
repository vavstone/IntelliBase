"""Разрешённые пользователи бота: одна таблица на два вопроса.

Таблица `bot_users` (chat_id + имя) отвечает сразу на два вопроса:

1. **Кому бот отвечает** — `is_allowed` для гейта на входе (бот спрашивает
   бэкенд через `GET /access/{chat_id}`, см. `app/routers/access.py`).
2. **Кому агент может отправлять** — `resolve_allowed`: по имени работает
   «отправь Иванову» (инструмент `find_recipient`), имя же видит человек в
   превью перед подтверждением.

Список ведёт админ через `/chats/admin/bot-users`, а не разработчик правкой
`.env`. Bootstrap (`BOT_ALLOWED_CHAT_IDS`) нужен только для чистого клона: пока
таблица пуста, бот закрыт для всех, и первый администратор должен откуда-то
взяться. Bootstrap-ids действуют **всегда** (их не надо дублировать в БД) — это
ещё и аварийный доступ, если список в БД испорчен.

Поведение при недоступной БД — fail-closed по таблице: возвращаем пустое
множество, а не «разрешаем всё». Чат-инициатор запроса разрешён на уровне графа
и от БД не зависит, поэтому демо-сценарий «отправь мне» переживает недоступность
Postgres.
"""

import logging

from sqlalchemy import text

log = logging.getLogger(__name__)

MAX_USERS_PER_QUERY = 5


def parse_chat_ids(raw: str) -> set[str]:
    """`BOT_ALLOWED_CHAT_IDS="1, 2"` → {"1", "2"}. Пустая строка → пустое множество."""
    return {part.strip() for part in (raw or "").split(",") if part.strip()}


def resolve_allowed(rows: dict[str, str], bootstrap: str) -> dict[str, str]:
    """Разрешённые пользователи: записи БД + bootstrap из настроек.

    Значение — человекочитаемое имя; у bootstrap-ids его нет (пустая строка),
    в интерфейсе вместо имени покажется только chat_id.
    """
    allowed = dict(rows)
    for chat_id in parse_chat_ids(bootstrap):
        allowed.setdefault(chat_id, "")
    return allowed


async def load_active_users(session_factory) -> dict[str, str]:
    """{chat_id: title} активных пользователей. Ошибка БД → пустой словарь."""
    if session_factory is None:
        return {}
    try:
        async with session_factory() as s:
            rows = (
                await s.execute(
                    text(
                        """
                        SELECT chat_id, title FROM bot_users
                        WHERE is_active IS TRUE
                        ORDER BY id
                        """
                    )
                )
            ).all()
    except Exception as exc:  # noqa: BLE001 — гейт важнее диагностики
        log.warning("Список пользователей недоступен (%s) — доступ только по bootstrap", exc)
        return {}
    return {str(r.chat_id): r.title for r in rows}


async def is_allowed(session_factory, chat_id: str, bootstrap: str = "") -> bool:
    """Есть ли у пользователя доступ к боту (для `GET /access/{chat_id}`)."""
    needle = str(chat_id).strip()
    if not needle:
        return False
    if needle in parse_chat_ids(bootstrap):
        return True
    return needle in await load_active_users(session_factory)


async def search_users(session_factory, query: str) -> list[dict]:
    """Поиск пользователя по имени — для инструмента `find_recipient`.

    Ищет только среди активных: инструмент не должен показывать модели тех,
    кому отправка всё равно не разрешена.
    """
    needle = (query or "").strip()
    if session_factory is None or not needle:
        return []
    async with session_factory() as s:
        rows = (
            await s.execute(
                text(
                    """
                    SELECT chat_id, title FROM bot_users
                    WHERE is_active IS TRUE AND title ILIKE :needle
                    ORDER BY id
                    LIMIT :limit
                    """
                ),
                {"needle": f"%{needle}%", "limit": MAX_USERS_PER_QUERY},
            )
        ).all()
    return [{"chat_id": str(r.chat_id), "title": r.title} for r in rows]


async def list_users(session_factory, *, active_only: bool = False) -> list[dict]:
    """Записи списка для админ-API (новые — первыми)."""
    if session_factory is None:
        return []
    where = "WHERE is_active IS TRUE" if active_only else ""
    async with session_factory() as s:
        rows = (
            await s.execute(
                text(
                    f"""
                    SELECT chat_id, title, is_active, created_at, created_by
                    FROM bot_users {where}
                    ORDER BY id DESC
                    """
                )
            )
        ).all()
    return [
        {
            "chat_id": str(r.chat_id),
            "title": r.title,
            "is_active": bool(r.is_active),
            "created_at": r.created_at,
            "created_by": r.created_by,
        }
        for r in rows
    ]


async def add_user(
    session_factory, chat_id: str, title: str, created_by: str | None = None
) -> dict:
    """Выдать доступ (upsert по chat_id).

    Повторное добавление с новым именем реактивирует запись: админ, вносящий
    человека заново, ожидает, что доступ заработает.
    """
    async with session_factory() as s:
        row = (
            await s.execute(
                text(
                    """
                    INSERT INTO bot_users (chat_id, title, created_by)
                    VALUES (:chat_id, :title, :created_by)
                    ON CONFLICT (chat_id) DO UPDATE
                        SET title = EXCLUDED.title,
                            is_active = TRUE,
                            created_by = EXCLUDED.created_by
                    RETURNING chat_id, title, is_active, created_at, created_by
                    """
                ),
                {"chat_id": chat_id, "title": title, "created_by": created_by},
            )
        ).one()
        await s.commit()
    return {
        "chat_id": str(row.chat_id),
        "title": row.title,
        "is_active": bool(row.is_active),
        "created_at": row.created_at,
        "created_by": row.created_by,
    }


async def deactivate_user(session_factory, chat_id: str) -> bool:
    """Отозвать доступ (мягко). False — записи не было или доступ уже отозван."""
    async with session_factory() as s:
        result = await s.execute(
            text(
                """
                UPDATE bot_users SET is_active = FALSE
                WHERE chat_id = :chat_id AND is_active IS TRUE
                """
            ),
            {"chat_id": chat_id},
        )
        await s.commit()
    return bool(result.rowcount)
