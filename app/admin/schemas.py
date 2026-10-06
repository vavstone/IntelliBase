"""Pydantic-схемы admin-API."""

from datetime import datetime

from pydantic import BaseModel, Field


class StatsOut(BaseModel):
    """Агрегаты за окно времени.

    total_requests, avg_latency_ms, p95_latency_ms и moderation_block_rate считаются
    по таблице request_metrics, которая заполняется observability-middleware на
    каждом запросе (служебные /health и /ready исключены — иначе пробы
    healthcheck'ов перекашивают распределение). cache_* — счётчики кэша LLM из
    Redis за всё время работы сервиса (подмешивает роут /stats).

    RAG-дельта (Б5.5): refusal_rate — доля отказов «не нашёл», negative_feedback_rate
    — доля отрицательных оценок, knowledge_gaps — топ вопросов без уверенного ответа.
    """

    total_messages: int
    active_users: int
    total_requests: int = 0
    avg_latency_ms: float = 0.0
    # p95 по request_metrics.duration_ms — «хвост» задержек для шага метрик в демо.
    p95_latency_ms: float = 0.0
    # Попадания/промахи кэша LLM (Redis, счётчики за всё время работы).
    cache_hits: int = 0
    cache_misses: int = 0
    cache_hit_rate: float = 0.0
    moderation_block_rate: float = 0.0
    feedback_ratio: float = 0.0
    refusal_rate: float = 0.0
    negative_feedback_rate: float = 0.0
    knowledge_gaps: list[str] = Field(default_factory=list)


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


class BotUserIn(BaseModel):
    """Разрешённый пользователь бота: кому бот отвечает и кому можно отправлять.

    `title` — человекочитаемое имя, по которому получателя находит инструмент
    («отправь Иванову») и которое видит человек в превью перед подтверждением.
    """

    chat_id: str = Field(min_length=1, max_length=32)
    title: str = Field(min_length=1, max_length=255)


class BotUserOut(BaseModel):
    chat_id: str
    title: str
    is_active: bool = True
    created_at: datetime | None = None
    created_by: str | None = None
