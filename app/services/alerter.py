"""Alerter: упрощённая БД-очередь алертов + мониторинг порогов.

Pattern: fire_alert пишет строку в alerts (jsonb payload), бот периодически
drain'ит pending → шлёт в админ-чат → ack. Это даёт at-least-once delivery
без внешнего message broker'а.

Threshold monitoring: фоновая таска мониторит агрегаты за последний час и
генерирует алерты при превышении порогов:
- moderation_block_rate > 5%
- 5xx error rate > 30%
- 429 rate limit rate > 20%

Для PoC хватает; в проде стоит подумать про partitioning по created_at
и cleanup acked-строк cron'ом.
"""

import asyncio
import json
import logging
from datetime import UTC, datetime, timedelta

from sqlalchemy import text

log = logging.getLogger(__name__)

# Пороги для мониторинга
MODERATION_BLOCK_RATE_THRESHOLD = 0.05   # 5%
ERROR_5XX_RATE_THRESHOLD = 0.30          # 30%
RATE_LIMIT_429_RATE_THRESHOLD = 0.20     # 20%

MONITOR_INTERVAL = 60.0  # Проверка раз в минуту
MONITOR_WINDOW_HOURS = 1


async def fire_alert(session_factory, kind: str, payload: dict) -> None:
    if session_factory is None:
        return
    async with session_factory() as s:
        await s.execute(
            text(
                """
                INSERT INTO alerts (kind, payload, created_at)
                VALUES (:k, CAST(:p AS jsonb), NOW())
                """
            ),
            {"k": kind, "p": json.dumps(payload)},
        )
        await s.commit()


async def fetch_pending_alerts(session_factory) -> list[dict]:
    if session_factory is None:
        return []
    async with session_factory() as s:
        rows = (
            await s.execute(
                text(
                    """
                    SELECT id, kind, payload FROM alerts
                    WHERE acked_at IS NULL
                    ORDER BY created_at ASC LIMIT 50
                    """
                )
            )
        ).all()
    return [
        {"id": r.id, "kind": r.kind, "payload": r.payload} for r in rows
    ]


async def ack_alert(session_factory, alert_id: int) -> None:
    if session_factory is None:
        return
    async with session_factory() as s:
        await s.execute(
            text("UPDATE alerts SET acked_at = NOW() WHERE id = :id"),
            {"id": alert_id},
        )
        await s.commit()


async def threshold_monitor(
    session_factory,
    poll_interval: float = MONITOR_INTERVAL,
) -> None:
    """Фоновая таска: периодически проверяет пороги и генерирует алерты.

    Использует таблицу request_metrics для расчёта агрегатов
    за последний час.
    """
    if session_factory is None:
        return

    # Предыдущие значения для избежания повторных алертов
    last_alerts: dict[str, float] = {}

    while True:
        try:
            since = datetime.now(UTC) - timedelta(hours=MONITOR_WINDOW_HOURS)
            async with session_factory() as s:
                metrics = (
                    await s.execute(
                        text(
                            """
                            SELECT
                                COUNT(*) AS total,
                                COUNT(*) FILTER (
                                    WHERE detail_code = 'moderation_blocked'
                                ) AS blocked,
                                COUNT(*) FILTER (
                                    WHERE status_code >= 500
                                ) AS errors_5xx,
                                COUNT(*) FILTER (
                                    WHERE detail_code = 'rate_limit'
                                ) AS rate_limited
                            FROM request_metrics
                            WHERE created_at >= :since
                            """
                        ),
                        {"since": since},
                    )
                ).first()

            total = max(metrics.total or 0, 1)  # избегаем деления на 0
            block_rate = (metrics.blocked or 0) / total
            error_rate = (metrics.errors_5xx or 0) / total
            rl_rate = (metrics.rate_limited or 0) / total

            now_ts = datetime.now(UTC).timestamp()

            # Проверка порогов с debounce (не чаще раза в 5 минут на тип)
            debounce = 300
            if block_rate > MODERATION_BLOCK_RATE_THRESHOLD:
                last = last_alerts.get("moderation_block_rate", 0)
                if now_ts - last > debounce:
                    await fire_alert(
                        session_factory,
                        kind="threshold_breach",
                        payload={
                            "metric": "moderation_block_rate",
                            "value": round(block_rate, 4),
                            "threshold": MODERATION_BLOCK_RATE_THRESHOLD,
                            "window_hours": MONITOR_WINDOW_HOURS,
                            "total_requests": total,
                        },
                    )
                    last_alerts["moderation_block_rate"] = now_ts

            if error_rate > ERROR_5XX_RATE_THRESHOLD:
                last = last_alerts.get("error_5xx_rate", 0)
                if now_ts - last > debounce:
                    await fire_alert(
                        session_factory,
                        kind="threshold_breach",
                        payload={
                            "metric": "error_5xx_rate",
                            "value": round(error_rate, 4),
                            "threshold": ERROR_5XX_RATE_THRESHOLD,
                            "window_hours": MONITOR_WINDOW_HOURS,
                            "total_requests": total,
                        },
                    )
                    last_alerts["error_5xx_rate"] = now_ts

            if rl_rate > RATE_LIMIT_429_RATE_THRESHOLD:
                last = last_alerts.get("rate_limit_429_rate", 0)
                if now_ts - last > debounce:
                    await fire_alert(
                        session_factory,
                        kind="threshold_breach",
                        payload={
                            "metric": "rate_limit_429_rate",
                            "value": round(rl_rate, 4),
                            "threshold": RATE_LIMIT_429_RATE_THRESHOLD,
                            "window_hours": MONITOR_WINDOW_HOURS,
                            "total_requests": total,
                        },
                    )
                    last_alerts["rate_limit_429_rate"] = now_ts
        except Exception as exc:
            log.warning("threshold_monitor: %s", exc)
        await asyncio.sleep(poll_interval)
