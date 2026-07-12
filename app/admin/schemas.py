"""Pydantic-схемы admin-API."""

from datetime import datetime

from pydantic import BaseModel


class StatsOut(BaseModel):
    """Агрегаты за окно времени.

    avg_latency_ms и moderation_block_rate считаются по таблице request_metrics,
    которая заполняется observability-middleware на каждом запросе.
    """

    total_messages: int
    active_users: int
    avg_latency_ms: float = 0.0
    moderation_block_rate: float = 0.0
    feedback_ratio: float = 0.0


class UserOut(BaseModel):
    """Пользователь сервиса — owner_external_id + метаданные."""

    owner_external_id: str
    interface: str
    last_seen_at: str  # ISO-формат datetime


class BroadcastIn(BaseModel):
    """Адресаты задаются ровно одним способом:

    - явный `owner_ids: list[int]` — рассылка по списку Telegram chat_id;
    - `interface_filter: "telegram"` — backend сам подтянет всех owner_external_id
      из таблицы `chats` по этому интерфейсу и поставит broadcast в очередь.

    Хотя бы одно из полей должно быть задано, иначе route вернёт 400.
    """

    text: str
    owner_ids: list[int] | None = None
    interface_filter: str | None = None


class BroadcastResult(BaseModel):
    sent: int
    failed: int
    detail: str | None = None


class ExportItem(BaseModel):
    id: str
    chat_id: str
    role: str
    content: str
    created_at: str


class ExportResult(BaseModel):
    items: list[ExportItem]
    next_after: datetime | None = None


class HandoffIn(BaseModel):
    owner_external_id: str
    interface: str = "telegram"
    status: str  # 'active' | 'paused_for_human' | 'resolved'


class AlertOut(BaseModel):
    id: int
    kind: str
    payload: dict
