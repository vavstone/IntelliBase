"""Метрики RAGAS 0.4 и сборка строки оценки.

RAGAS 0.4 — collections-API: каждая метрика это объект с корутиной `ascore(...)`
(аргументы только именованные, у разных метрик разный набор полей), результат —
`MetricResult` со `.value`. Судья создаётся через `llm_factory`, эмбеддинги для
`AnswerRelevancy` — через `ragas.embeddings.HuggingFaceEmbeddings` (локальная E5,
совпадает с прод-эмбеддингами — облако не нужно). Кастомная категориальная
метрика — декоратором `@discrete_metric` (в 0.4 `AspectCritic` убран).

Судья по умолчанию — DeepSeek (`eval_judge_provider="deepseek"`), OpenAI-совместимый
эндпоинт без VPN; отделён от production-LLM в /rag/query (роли разные).

Модуль импортирует ragas на верхнем уровне — тяжёлая группа `eval`
(`uv sync --extra eval`). В проде не используется.
"""

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from pydantic import BaseModel
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.llms import llm_factory
from ragas.metrics import discrete_metric
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

from app.core.config import Settings

# DeepSeek (deepseek-v4-flash) — reasoning-модель: по умолчанию на каждый вызов
# генерирует chain-of-thought (reasoning_tokens ~3–4x от полезного ответа). Для
# extract/summary/judge-задач это чистый оверхед — отключаем через extra_body.
DEEPSEEK_NO_THINKING = {"extra_body": {"thinking": {"type": "disabled"}}}


class TokenCounter:
    """Суммирует input/output-токены LLM-вызовов (для оценки стоимости прогона)."""

    def __init__(self) -> None:
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def add(self, usage: Any) -> None:
        if usage is None:
            return
        self.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
        self.completion_tokens += getattr(usage, "completion_tokens", 0) or 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def _counting_client(client: AsyncOpenAI, counter: TokenCounter) -> AsyncOpenAI:
    """Оборачивает chat.completions.create, чтобы считать токены по usage ответа.

    openai-клиент хранит `chat` как cached_property (тот же объект при каждом
    доступе), поэтому подмена `chat.completions.create` сохраняется.
    """
    create = client.chat.completions.create

    async def _wrapped(*args: Any, **kwargs: Any) -> Any:
        response = await create(*args, **kwargs)
        counter.add(getattr(response, "usage", None))
        return response

    client.chat.completions.create = _wrapped  # type: ignore[method-assign]
    return client


@dataclass
class RagasMetrics:
    """Четыре collections-метрики RAGAS, собранных на одном судье.

    FactualCorrectness намеренно не включена: критерий Б5.6 требует 5 метрик
    (4 collections + has_citation), а это дополнительный судья-вызов на строку.
    """

    faithfulness: Faithfulness
    answer_relevancy: AnswerRelevancy
    context_precision: ContextPrecision
    context_recall: ContextRecall


def build_judge(settings: Settings) -> Any:
    """Судья (LLM) для метрик. DeepSeek по умолчанию, поддерживает openai/anthropic.

    DeepSeek и OpenAI — OpenAI-совместимые клиенты через `llm_factory(..., provider="openai")`.
    На возвращённый судья вешается атрибут `token_counter` (TokenCounter) — по нему
    run_eval считает реальный расход токенов за прогон.
    """
    counter = TokenCounter()
    provider = settings.eval_judge_provider
    if provider == "deepseek":
        client = _counting_client(
            AsyncOpenAI(
                base_url=settings.llm.deepseek_base_url,
                api_key=settings.llm.deepseek_api_key.get_secret_value(),
            ),
            counter,
        )
        judge = llm_factory(
            settings.eval_judge_model,
            provider="openai",
            client=client,
            **DEEPSEEK_NO_THINKING,
        )
    elif provider == "openai":
        client = _counting_client(
            AsyncOpenAI(
                base_url=settings.llm.openai_base_url,
                api_key=settings.llm.openai_api_key.get_secret_value(),
            ),
            counter,
        )
        judge = llm_factory(settings.eval_judge_model, provider="openai", client=client)
    elif provider == "anthropic":
        api_key = (
            settings.anthropic_api_key.get_secret_value()
            if settings.anthropic_api_key is not None
            else None
        )
        judge = llm_factory(
            settings.eval_judge_model,
            provider="anthropic",
            client=AsyncAnthropic(api_key=api_key),
        )
    else:
        raise ValueError(f"Неизвестный провайдер судьи: {provider}")

    try:
        judge.token_counter = counter  # type: ignore[attr-defined]
    except Exception:
        pass
    return judge


def build_embeddings(settings: Settings) -> HuggingFaceEmbeddings:
    """Локальные эмбеддинги для AnswerRelevancy (та же модель, что и в RAG).

    E5-префиксы `query:`/`passage:` здесь не добавляются: AnswerRelevancy сравнивает
    косинус между вопросом и восстановленными по ответу вопросами — обе стороны
    кодируются одинаково, поэтому метрика остаётся корректной.
    """
    return HuggingFaceEmbeddings(model=settings.embedding.model)


def build_metrics(judge: Any, embeddings: HuggingFaceEmbeddings) -> RagasMetrics:
    """Четыре метрики генерации/контекста на общем судье."""
    return RagasMetrics(
        faithfulness=Faithfulness(llm=judge),
        answer_relevancy=AnswerRelevancy(llm=judge, embeddings=embeddings),
        context_precision=ContextPrecision(llm=judge),
        context_recall=ContextRecall(llm=judge),
    )


class CitationVerdict(BaseModel):
    has_citation: Literal["yes", "no"]


CITATION_PROMPT = (
    "Содержит ли ответ ссылку на источник: маркер вида '[1]'/'[doc_id]', имя "
    "файла, или фразу 'согласно ...', 'в источнике X указано'?\n\n"
    "Ответ: {response}"
)


def make_has_citation(judge: Any):
    """Кастомная метрика «ответ содержит цитату» через @discrete_metric.

    Возвращает метрику, замкнутую на переданном судье, — так её удобно
    тестировать с фейковым судьёй и переиспользовать в run_eval.
    """

    @discrete_metric(name="has_citation", allowed_values=["yes", "no"])
    async def has_citation(response: str) -> str:
        verdict = await judge.agenerate(
            CITATION_PROMPT.format(response=response), response_model=CitationVerdict
        )
        return verdict.has_citation

    return has_citation


async def eval_row(rag: Any, row: dict, metrics: RagasMetrics, has_citation: Any) -> dict:
    """Пять метрик + latency по одной строке golden dataset.

    `rag` — RAGService (или совместимый), у которого есть `evaluate_inputs`.
    Аргументы метрик именованные: у каждой свой набор полей.

    Метрики, которым не хватает входных полей (пустой reference / пустые contexts —
    например при отказе score-guard), помечаются None вместо вызова судьи: это
    защита от падения на неполных строках golden dataset.
    """
    t0 = perf_counter()
    result = await rag.evaluate_inputs(row.get("user_input", ""))
    latency_ms = round((perf_counter() - t0) * 1000.0, 1)
    answer = result.get("answer", "")
    contexts = result.get("retrieved_contexts") or []
    q = row.get("user_input", "")
    ref = (row.get("reference") or "").strip()

    return {
        "user_input": q,
        "faithfulness": (
            (
                await metrics.faithfulness.ascore(
                    user_input=q, response=answer, retrieved_contexts=contexts
                )
            ).value
            if contexts
            else None
        ),
        "answer_relevancy": (
            await metrics.answer_relevancy.ascore(user_input=q, response=answer)
        ).value,
        "context_precision": (
            (
                await metrics.context_precision.ascore(
                    user_input=q, reference=ref, retrieved_contexts=contexts
                )
            ).value
            if ref and contexts
            else None
        ),
        "context_recall": (
            (
                await metrics.context_recall.ascore(
                    user_input=q, retrieved_contexts=contexts, reference=ref
                )
            ).value
            if ref and contexts
            else None
        ),
        "has_citation": (await has_citation.ascore(response=answer)).value,
        "latency_ms": latency_ms,
    }
