"""Broadcaster: асинхронная очередь массовых рассылок.

Pattern:
1. `enqueue_broadcast` — создаёт запись в таблице broadcasts (status=pending).
2. Фоновая таска `broadcast_worker` — забирает pending-записи, отправляет
   через bot /notify с тротлингом, обновляет статус (sending → done / partial_fail).
3. Graceful resume: при старте проверяет sending-записи (упали mid-flight) →
   переводит в pending для повторной обработки.

THROTTLE ~25 msg/sec — безопасно ниже Telegram-лимита 30/sec на бота.
"""

import asyncio
import json
import logging
from datetime import UTC, datetime

import httpx
from sqlalchemy import text as sa_text

log = logging.getLogger(__name__)

THROTTLE = 0.04  # ~25 msg/sec


async def enqueue_broadcast(
    session_factory,
    text: str,
    interface_filter: str = "telegram",
) -> dict:
    """Создаёт запись broadcast в очереди. Возвращает {id, status, total}."""
    if session_factory is None:
        raise RuntimeError("broadcast requires postgres")
    async with session_factory() as s:
        # Считаем получателей
        result = await s.execute(
            sa_text(
                """
                SELECT COUNT(DISTINCT owner_external_id)
                FROM chats WHERE interface = :i
                """
            ),
            {"i": interface_filter},
        )
        total = result.scalar() or 0
        if total == 0:
            return {"id": None, "status": "done", "total": 0}

        result = await s.execute(
            sa_text(
                """
                INSERT INTO broadcasts (text, interface_filter, status, sent, failed, total, created_at)
                VALUES (:t, :i, 'pending', 0, 0, :total, NOW())
                RETURNING id
                """
            ),
            {"t": text, "i": interface_filter, "total": total},
        )
        bid = result.scalar()
        await s.commit()
    return {"id": bid, "status": "pending", "total": total}


async def broadcast_worker(
    session_factory,
    bot_url: str,
    internal_token: str,
    poll_interval: float = 5.0,
) -> None:
    """Фоновая таска: разгребает очередь broadcasts.

    Периодически опрашивает таблицу, отправляет, обновляет статус.
    При старте переводит sending → pending (graceful resume после падения).
    """
    if session_factory is None:
        return

    # Graceful resume: sending → pending
    try:
        async with session_factory() as s:
            await s.execute(
                sa_text(
                    "UPDATE broadcasts SET status='pending' WHERE status='sending'"
                )
            )
            await s.commit()
    except Exception as exc:
        log.warning("broadcast resume failed: %s", exc)

    http = httpx.AsyncClient(timeout=5.0)

    while True:
        try:
            async with session_factory() as s:
                # Берём первый pending broadcast
                row = (
                    await s.execute(
                        sa_text(
                            """
                            SELECT id, text, interface_filter
                            FROM broadcasts
                            WHERE status = 'pending'
                            ORDER BY created_at ASC
                            LIMIT 1
                            """
                        )
                    )
                ).first()
                if row is None:
                    await asyncio.sleep(poll_interval)
                    continue

                bid, text_val, iface = row.id, row.text, row.interface_filter

                # Помечаем как sending
                await s.execute(
                    sa_text("UPDATE broadcasts SET status='sending' WHERE id=:id"),
                    {"id": bid},
                )
                await s.commit()

                # Получаем список получателей
                owner_rows = (
                    await s.execute(
                        sa_text(
                            """
                            SELECT DISTINCT owner_external_id
                            FROM chats WHERE interface = :i
                            """
                        ),
                        {"i": iface},
                    )
                ).all()
                owner_ids = []
                for r in owner_rows:
                    try:
                        owner_ids.append(int(r.owner_external_id))
                    except (TypeError, ValueError):
                        continue

                # Отправляем
                sent = failed = 0
                for owner_id in owner_ids:
                    try:
                        r = await http.post(
                            f"{bot_url}/notify",
                            json={"chat_id": owner_id, "text": text_val},
                            headers={"X-Internal-Token": internal_token},
                        )
                        r.raise_for_status()
                        sent += 1
                    except httpx.HTTPError as e:
                        failed += 1
                        log.warning(
                            "broadcast[%s]: failed for %s: %s", bid, owner_id, e
                        )
                    await asyncio.sleep(THROTTLE)

                # Финализируем
                final_status = "done" if failed == 0 else "partial_fail"
                await s.execute(
                    sa_text(
                        """
                        UPDATE broadcasts
                        SET status=:st, sent=:sent, failed=:failed, finished_at=NOW()
                        WHERE id=:id
                        """
                    ),
                    {
                        "st": final_status,
                        "sent": sent,
                        "failed": failed,
                        "id": bid,
                    },
                )
                await s.commit()
                log.info(
                    "broadcast[%s] finished: status=%s sent=%d failed=%d",
                    bid, final_status, sent, failed,
                )
        except Exception as exc:
            log.warning("broadcast worker: %s", exc)
            await asyncio.sleep(poll_interval)


async def broadcast_sync(
    text: str,
    owner_ids: list[int],
    bot_url: str,
    internal_token: str,
) -> dict:
    """Синхронная рассылка по явному списку (для обратной совместимости)."""
    sent = failed = 0
    async with httpx.AsyncClient(timeout=5.0) as c:
        for owner_id in owner_ids:
            try:
                r = await c.post(
                    f"{bot_url}/notify",
                    json={"chat_id": owner_id, "text": text},
                    headers={"X-Internal-Token": internal_token},
                )
                r.raise_for_status()
                sent += 1
            except httpx.HTTPError as e:
                failed += 1
                log.warning("broadcast: failed for %s: %s", owner_id, e)
            await asyncio.sleep(THROTTLE)
    return {"sent": sent, "failed": failed}
