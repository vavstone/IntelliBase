"""Юнит-тесты для RAG-блока 5.5.

Покрывают чистую логику без живой инфраструктуры:
- build_sources / parse_citations / numbered_context / build_filters;
- _naive_chunk и _read_text (bare-metal, из Б5.3);
- эндпоинт POST /rag/query (фейковый RAG-сервис + новый контракт sources).

Живой end-to-end прогон (Qdrant + Ollama + E5) — в dev_tasks/verify_5_5.py.
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
from llama_index.core.schema import NodeWithScore, TextNode

from app.services.rag import (
    build_filters,
    build_sources,
    numbered_context,
    parse_citations,
)
from app.services.rag_baremetal import _naive_chunk, _read_text


def _node(text: str, source: str | None = None, page: int | None = None, score: float = 0.5) -> NodeWithScore:
    meta: dict = {}
    if source is not None:
        meta["source"] = source
    if page is not None:
        meta["page"] = page
    return NodeWithScore(node=TextNode(text=text, metadata=meta), score=score)


# ── build_sources ────────────────────────────────────────────────────────

def test_build_sources_maps_id_file_page_score_snippet() -> None:
    src = build_sources([_node("фрагмент источника", source="a.pdf", page=3, score=0.8765)])
    assert src[0]["id"] == 1
    assert src[0]["file_name"] == "a.pdf"
    assert src[0]["page"] == 3
    assert src[0]["score"] == 0.876
    assert src[0]["snippet"] == "фрагмент источника"


def test_build_sources_no_page_and_unknown_file() -> None:
    src = build_sources([_node("x", source=None, page=None)])
    assert src[0]["file_name"] == "unknown"
    assert src[0]["page"] is None


def test_build_sources_numbers_sequentially() -> None:
    src = build_sources([_node("a"), _node("b"), _node("c")])
    assert [s["id"] for s in src] == [1, 2, 3]


# ── parse_citations ──────────────────────────────────────────────────────

def test_parse_citations_expands_brackets() -> None:
    text = "Возврат 14 дней [1], подробнее [2]."
    sources = [{"id": 1, "file_name": "a.pdf"}, {"id": 2, "file_name": "b.pdf"}]
    out = parse_citations(text, sources)
    assert "[1 — a.pdf]" in out
    assert "[2 — b.pdf]" in out


def test_parse_citations_keeps_unknown_ids() -> None:
    assert parse_citations("[9] нет источника", []) == "[9] нет источника"


# ── numbered_context ─────────────────────────────────────────────────────

def test_numbered_context_orders_nodes() -> None:
    ctx = numbered_context([_node("первый"), _node("второй")])
    assert "[1] первый" in ctx
    assert "[2] второй" in ctx


# ── build_filters ────────────────────────────────────────────────────────

def test_build_filters_visibility_and_categories() -> None:
    f = build_filters(visibility="internal", categories=["Тарифы"])
    assert f is not None
    assert len(f.filters) == 2


def test_build_filters_none_when_empty() -> None:
    assert build_filters(visibility=None, categories=None) is None


# ── bare-metal helpers (Б5.3) ────────────────────────────────────────────

def test_naive_chunk_splits_fixed_window() -> None:
    assert _naive_chunk("0123456789", size=4) == ["0123", "4567", "89"]


def test_naive_chunk_empty_text() -> None:
    assert _naive_chunk("   ") == []


def test_read_text_markdown(tmp_path: Path) -> None:
    f = tmp_path / "doc.md"
    f.write_text("# Заголовок\n\nТекст", encoding="utf-8")
    assert _read_text(f) == "# Заголовок\n\nТекст"


def test_read_text_unsupported_extension(tmp_path: Path) -> None:
    f = tmp_path / "data.xls"
    f.write_text("nope", encoding="utf-8")
    assert _read_text(f) == ""


# ── endpoint POST /rag/query (новый контракт) ────────────────────────────

class _FakeRAGService:
    def __init__(self, result: dict) -> None:
        self._result = result

    async def answer(self, question: str) -> dict:
        return self._result


_HAPPY = {
    "answer": "Возврат оформляется 14 дней [1].",
    "top_score": 0.57,
    "sources": [
        {"id": 1, "file_name": "a.pdf", "page": 2, "score": 0.57, "snippet": "Возврат..."},
    ],
    "confident": True,
}
_REFUSAL = {
    "answer": "В базе знаний я не нашёл ответа на этот вопрос.",
    "top_score": 0.12,
    "sources": [],
    "confident": False,
}


def _build_app(rag_service=None) -> FastAPI:
    from app.routers.rag import router

    app = FastAPI()
    app.include_router(router)
    app.state.rag_service = rag_service
    # Без PG: log_rag_query — no-op при session_factory=None.
    app.state.session_factory = None
    return app


def test_rag_query_returns_confident_answer_with_sources() -> None:
    client = TestClient(_build_app(_FakeRAGService(_HAPPY)))
    resp = client.post("/rag/query", json={"question": "как вернуть деньги?"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "Возврат оформляется 14 дней [1]."
    assert data["top_score"] == 0.57
    assert data["confident"] is True
    assert data["sources"][0] == {
        "id": 1, "file_name": "a.pdf", "page": 2, "score": 0.57, "snippet": "Возврат...",
    }


def test_rag_query_refusal_is_not_confident() -> None:
    client = TestClient(_build_app(_FakeRAGService(_REFUSAL)))
    resp = client.post("/rag/query", json={"question": "чего нет в базе"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["confident"] is False
    assert data["sources"] == []


def test_rag_query_503_when_service_unavailable() -> None:
    client = TestClient(_build_app(rag_service=None))
    resp = client.post("/rag/query", json={"question": "что это?"})
    assert resp.status_code == 503


def test_rag_query_422_on_empty_question() -> None:
    client = TestClient(_build_app(_FakeRAGService(_HAPPY)))
    resp = client.post("/rag/query", json={"question": ""})
    assert resp.status_code == 422
