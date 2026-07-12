"""AdminRepository: агрегации по таблицам chat_messages / chats / message_feedback.

PII-маскирование применяется только к экспорту, не к сторонним read-API.
Внутренние логи/UI продолжают видеть исходный контент (это полезно для
дебага). Если включить маскировку в /list_messages — пользователь увидит
свои же сообщения с [EMAIL] и не поймёт, что произошло.
"""

from datetime import UTC, datetime, timedelta

from sqlalchemy import text

from app.admin.schemas import ExportItem, ExportResult, StatsOut, UserOut
from app.observability.pii import redact_pii


class AdminRepository:
    def __init__(self, session_factory):
        self.session_factory = session_factory

    async def compute_stats(self, window_hours: int = 24) -> StatsOut:
        """Активность за окно времени. `deleted_at` намеренно НЕ фильтруем:
        soft-delete нужен LLM-контексту (после /clear прошлые сообщения не
        подтягиваются в prompt), но факт активности юзера в окне остался —
        статистика должна это видеть.

        avg_latency_ms и moderation_block_rate считаются по request_metrics
        (заполняется observability-middleware).
        """
        if self.session_factory is None:
            return StatsOut(total_messages=0, active_users=0)
        since = datetime.now(UTC) - timedelta(hours=window_hours)
        async with self.session_factory() as s:
            total = await s.scalar(
                text(
                    "SELECT COUNT(*) FROM chat_messages WHERE created_at >= :since"
                ),
                {"since": since},
            )
            active = await s.scalar(
                text(
                    """
                    SELECT COUNT(DISTINCT c.owner_external_id)
                    FROM chats c
                    JOIN chat_messages cm ON cm.chat_id = c.id
                    WHERE cm.created_at >= :since
                    """
                ),
                {"since": since},
            )
            # Метрики из request_metrics
            metrics_row = (
                await s.execute(
                    text(
                        """
                        SELECT
                            COALESCE(AVG(duration_ms), 0) AS avg_latency_ms,
                            COUNT(*) AS total_requests,
                            COUNT(*) FILTER (
                                WHERE status_code = 403
                                  AND detail_code = 'moderation_blocked'
                            ) AS blocked_requests
                        FROM request_metrics
                        WHERE created_at >= :since
                        """
                    ),
                    {"since": since},
                )
            ).first()
            avg_latency = (metrics_row.avg_latency_ms if metrics_row else 0) or 0.0
            total_reqs = (metrics_row.total_requests if metrics_row else 0) or 1
            blocked = (metrics_row.blocked_requests if metrics_row else 0) or 0
            block_rate = blocked / total_reqs if total_reqs > 0 else 0.0

            fb = await s.execute(
                text(
                    """
                    SELECT
                        COUNT(*) FILTER (WHERE value='up') AS up,
                        COUNT(*) FILTER (WHERE value='down') AS down
                    FROM message_feedback
                    WHERE created_at >= :since
                    """
                ),
                {"since": since},
            )
            row = fb.first()
            up = (row.up if row else 0) or 0
            down = (row.down if row else 0) or 0
            ratio = up / (up + down) if (up + down) > 0 else 0.0
        return StatsOut(
            total_messages=total or 0,
            active_users=active or 0,
            avg_latency_ms=round(avg_latency, 2),
            moderation_block_rate=round(block_rate, 4),
            feedback_ratio=round(ratio, 4),
        )

    async def list_owner_ids_by_interface(self, interface: str) -> list[int]:
        """Возвращает уникальные owner_external_id для рассылок.

        Для telegram owner_external_id — это str(message.chat.id), поэтому
        корректно интерпретируется как int. Owner'ы, чьи id не приводятся
        к int (другой интерфейс с не-числовым id) — пропускаются.
        """
        if self.session_factory is None:
            return []
        async with self.session_factory() as s:
            rows = (
                await s.execute(
                    text(
                        """
                        SELECT DISTINCT owner_external_id
                        FROM chats
                        WHERE interface = :i
                        """
                    ),
                    {"i": interface},
                )
            ).all()
        out: list[int] = []
        for r in rows:
            try:
                out.append(int(r.owner_external_id))
            except (TypeError, ValueError):
                continue
        return out

    async def list_users(self, limit: int = 50) -> list[UserOut]:
        """Список пользователей, отсортированный по last_seen_at (свежие сначала).

        last_seen_at определяется как MAX(created_at) среди сообщений чата
        данного пользователя.
        """
        if self.session_factory is None:
            return []
        async with self.session_factory() as s:
            rows = (
                await s.execute(
                    text(
                        """
                        SELECT
                            c.owner_external_id,
                            c.interface,
                            MAX(cm.created_at) AS last_seen_at
                        FROM chats c
                        JOIN chat_messages cm ON cm.chat_id = c.id
                        GROUP BY c.owner_external_id, c.interface
                        ORDER BY last_seen_at DESC
                        LIMIT :limit
                        """
                    ),
                    {"limit": limit},
                )
            ).all()
        return [
            UserOut(
                owner_external_id=r.owner_external_id,
                interface=r.interface,
                last_seen_at=r.last_seen_at.isoformat() if r.last_seen_at else "",
            )
            for r in rows
        ]

    async def export_messages(
        self, after: datetime | None, limit: int
    ) -> ExportResult:
        if self.session_factory is None:
            return ExportResult(items=[], next_after=None)
        async with self.session_factory() as s:
            stmt = text(
                """
                SELECT id, chat_id, role, content, created_at
                FROM chat_messages
                WHERE deleted_at IS NULL
                  AND (CAST(:after AS TIMESTAMPTZ) IS NULL
                       OR created_at > CAST(:after AS TIMESTAMPTZ))
                ORDER BY created_at ASC
                LIMIT :limit
                """
            )
            rows = (
                await s.execute(stmt, {"after": after, "limit": limit})
            ).all()
        items = [
            ExportItem(
                id=str(r.id),
                chat_id=str(r.chat_id),
                role=r.role,
                content=redact_pii(r.content),
                created_at=r.created_at.isoformat(),
            )
            for r in rows
        ]
        return ExportResult(
            items=items,
            next_after=rows[-1].created_at if rows else None,
        )
