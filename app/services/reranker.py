"""Re-ranker: cross-encoder поверх bi-encoder поиска (ДЗ 5.4).

BAAI/bge-reranker-v2-m3 — мультиязычный (в т.ч. русский) cross-encoder,
который точнее косинусной близости пересортировывает кандидатов, потому что
видит запрос и чанк в одном проходе (attention между словами). Модель
скачивается с HuggingFace при первом запуске (~2.2 ГБ) и работает локально.

Роль в пайплайне: bi-encoder достаёт top-K кандидатов (recall), cross-encoder
пересортировывает их и оставляет top-N (precision).
"""

from __future__ import annotations

from typing import Any

from sentence_transformers import CrossEncoder

DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"


class Reranker:
    """Обёртка над CrossEncoder.

    Вход — список кандидатов (dict с ключом "text"), выход — пересортированный
    top-N с добавленным ключом "rerank_score".
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, top_n: int | None = None) -> None:
        self.model_name = model_name
        self.top_n = top_n
        self._model: CrossEncoder = CrossEncoder(model_name)

    def scores(self, query: str, texts: list[str]) -> list[float]:
        """Сырые оценки релевантности пары (query, text) для каждого текста."""
        if not texts:
            return []
        raw = self._model.predict([(query, text) for text in texts])
        return [float(score) for score in raw]

    def rerank(self, query: str, candidates: list[dict], top_n: int | None = None) -> list[dict]:
        """Пересортировывает кандидатов по cross-encoder и обрезает до top_n.

        Аргументы:
            query — текст запроса;
            candidates — список dict с ключом "text" (прочие ключи сохраняются);
            top_n — сколько оставить (по умолчанию self.top_n; None — вернуть все).

        Возвращает копии кандидатов, отсортированные по убыванию "rerank_score".
        """
        if not candidates:
            return []
        texts = [str(c["text"]) for c in candidates]
        scores = self.scores(query, texts)
        ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)

        result: list[dict] = []
        for candidate, score in ranked:
            item = dict(candidate)
            item["rerank_score"] = round(score, 4)
            result.append(item)

        limit = top_n if top_n is not None else self.top_n
        return result[:limit] if limit is not None else result


def top_n_doc_ids(query: str, candidates: list[dict], reranker: Reranker, top_n: int) -> list[str]:
    """Удобство для eval: возвращает doc_id после переранжирования, в порядке.

    candidate = {"text": ..., "doc_id": <имя файла>, ...}.
    """
    ranked = reranker.rerank(query, candidates, top_n=top_n)
    return [c["doc_id"] for c in ranked]
