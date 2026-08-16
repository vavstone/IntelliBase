"""Юнит-тесты для RAG-блока 5.3.

Покрывают чистую логику без обращения к живой инфраструктуре:
- format_result (контракт answer()/fallback по порогу);
- _naive_chunk и _read_text (bare-metal);
- эндпоинт POST /rag/query (мокированным RAG-сервисом).

Живой end-to-end прогон (Qdrant + Ollama + E5) — в dev_tasks/verify_5_3.py.
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.services.rag import format_result
from app.services.rag_baremetal import _naive_chunk, _read_text


# ── fakes ────────────────────────────────────────────────────────────────

class _FakeNode:
    def __init__(self, text: str, source: str | None, score: float) -> None:
        self.text = text
        self.metadata = {"file_name": source}
        self.score = score


class _FakeResponse:
    def __init__(self, nodes: list, answer: str = "сгенерированный ответ") -> None:
        self.source_nodes = nodes
        self._answer = answer

    def __str__(self) -> str:
        return self._answer


def _nodes(*specs) -> list:
    return [_FakeNode(text, source, score) for text, source, score in specs]


# ── format_result ────────────────────────────────────────────────────────

def test_format_result_maps_sources_and_truncates_text() -> None:
    """sources: text обрезается до 300, берутся file_name и округлённый score."""
    long_text = "а" * 500
    result = format_result(
        _FakeResponse(_nodes((long_text, "doc.md", 0.8765))), threshold=0.5
    )
    assert result["answer"] == "сгенерированный ответ"
    assert result["top_score"] == 0.876
    assert len(result["sources"]) == 1
    s = result["sources"][0]
    assert s["text"] == long_text[:300]
    assert s["source"] == "doc.md"
    assert s["score"] == 0.876


def test_format_result_top_score_is_max_across_nodes() -> None:
    """top_score — максимум по всем нодам."""
    result = format_result(
        _FakeResponse(_nodes(("a", "x.md", 0.3), ("b", "y.md", 0.9), ("c", "z.md", 0.5))),
        threshold=0.5,
    )
    assert result["top_score"] == 0.9


def test_format_result_fallback_below_threshold() -> None:
    """top-1 ниже порога — честный fallback вместо сгенерированного текста."""
    result = format_result(
        _FakeResponse(_nodes(("шум", "offtopic.md", 0.2)), answer="выдуманный ответ"),
        threshold=0.5,
    )
    assert result["answer"] == "В базе знаний нет ответа на этот вопрос."


def test_format_result_empty_nodes() -> None:
    """Пустой список нод — top_score 0 и пустые sources, fallback."""
    result = format_result(_FakeResponse([]), threshold=0.5)
    assert result["top_score"] == 0.0
    assert result["sources"] == []
    assert result["answer"] == "В базе знаний нет ответа на этот вопрос."


# ── bare-metal helpers ───────────────────────────────────────────────────

def test_naive_chunk_splits_fixed_window() -> None:
    chunks = _naive_chunk("0123456789", size=4)
    assert chunks == ["0123", "4567", "89"]


def test_naive_chunk_empty_text() -> None:
    assert _naive_chunk("   ") == []


def test_read_text_markdown(tmp_path: Path) -> None:
    f = tmp_path / "doc.md"
    f.write_text("# Заголовок\n\nТекст документа", encoding="utf-8")
    assert _read_text(f) == "# Заголовок\n\nТекст документа"


def test_read_text_unsupported_extension(tmp_path: Path) -> None:
    f = tmp_path / "data.xls"
    f.write_text("nope", encoding="utf-8")
    assert _read_text(f) == ""


# ── endpoint POST /rag/query ─────────────────────────────────────────────

class _FakeRAGService:
    async def answer(self, question: str) -> dict:
        return {
            "answer": "ответ по контексту",
            "top_score": 0.9,
            "sources": [
                {"text": "фрагмент", "source": "doc.md", "score": 0.9},
            ],
        }


def _build_app(rag_service=None) -> FastAPI:
    from app.routers.rag import router

    app = FastAPI()
    app.include_router(router)
    app.state.rag_service = rag_service
    return app


def test_rag_query_returns_answer_and_sources() -> None:
    client = TestClient(_build_app(_FakeRAGService()))
    resp = client.post("/rag/query", json={"question": "что это?"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "ответ по контексту"
    assert data["top_score"] == 0.9
    assert len(data["sources"]) == 1
    assert data["sources"][0]["source"] == "doc.md"


def test_rag_query_503_when_service_unavailable() -> None:
    client = TestClient(_build_app(rag_service=None))
    resp = client.post("/rag/query", json={"question": "что это?"})
    assert resp.status_code == 503


def test_rag_query_422_on_empty_question() -> None:
    client = TestClient(_build_app(_FakeRAGService()))
    resp = client.post("/rag/query", json={"question": ""})
    assert resp.status_code == 422
