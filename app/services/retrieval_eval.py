"""Retrieval-метрики по golden dataset (ДЗ 5.4).

Считаются на уровне документа (doc_id = имя файла), а не чанка: ретривер
возвращает ранжированный список doc_id, который сравнивается с
relevant_doc_ids из tests/eval/retrieval_dataset.json.

Метрики:
- Hit Rate@5  — доля запросов, где хотя бы один релевантный документ попал
  в top-5;
- MRR@10     — среднее 1/rank первого релевантного документа в top-10;
- Recall@10  — средняя доля релевантных документов, найденных в top-10.

Единая обёртка evaluate_retrieval() возвращает словарь со всеми тремя числами.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence


def hit_rate_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int = 5) -> float:
    """1.0, если хотя бы один релевантный документ есть в top-k, иначе 0.0."""
    if not relevant:
        return 0.0
    return 1.0 if (set(retrieved[:k]) & set(relevant)) else 0.0


def mrr_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int = 10) -> float:
    """1/rank первого релевантного документа в top-k (0.0, если нет)."""
    rel = set(relevant)
    for rank, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in rel:
            return 1.0 / rank
    return 0.0


def recall_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int = 10) -> float:
    """Доля релевантных документов, попавших в top-k."""
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & set(relevant)) / len(set(relevant))


def evaluate_retrieval(
    retrieve_fn: Callable[[str], list[str]],
    golden_set: Sequence[dict],
    top_k: int = 10,
) -> dict:
    """Прогоняет golden dataset и возвращает словарь с тремя метриками.

    retrieve_fn(question) -> list[str] — ранжированный список doc_id (имён
    файлов) без дубликатов в порядке релевантности.

    Возвращает {hit_rate_at_5, mrr_at_10, recall_at_10, n_questions, per_question}.
    """
    hit5 = 0.0
    mrr10 = 0.0
    recall10 = 0.0
    per_question: list[dict] = []

    for item in golden_set:
        question = item["question"]
        relevant = item["relevant_doc_ids"]
        retrieved = retrieve_fn(question) or []

        h5 = hit_rate_at_k(retrieved, relevant, 5)
        m10 = mrr_at_k(retrieved, relevant, 10)
        r10 = recall_at_k(retrieved, relevant, 10)
        hit5 += h5
        mrr10 += m10
        recall10 += r10
        per_question.append(
            {
                "question": question,
                "relevant_doc_ids": relevant,
                "retrieved_doc_ids": retrieved[:top_k],
                "hit_at_5": h5,
                "mrr_at_10": m10,
                "recall_at_10": r10,
            }
        )

    n = len(golden_set) or 1
    return {
        "hit_rate_at_5": round(hit5 / n, 4),
        "mrr_at_10": round(mrr10 / n, 4),
        "recall_at_10": round(recall10 / n, 4),
        "n_questions": len(golden_set),
        "per_question": per_question,
    }
