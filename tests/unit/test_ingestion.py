"""Юнит-тесты чистых функций офлайн-контура индексации (без Qdrant/embedding)."""

from pathlib import Path
from types import SimpleNamespace

from llama_index.core.schema import Document

from app.services import ingestion
from app.services.ingestion import (
    EXCLUDED_EMBED_KEYS,
    _doc_id,
    category_from_path,
    clean,
    doc_type_from_path,
    enrich,
    file_metadata,
    version_from_filename,
)


def test_clean_strips_footer_and_joins_hyphenation() -> None:
    raw = "Регламент возврата Стр. 12 из 47 авто-\nмобиль доступен https://x.io/a тут"
    out = clean(raw)
    assert "Стр. 12 из 47" not in out
    assert "автомобиль" in out
    assert "https://" not in out


def test_clean_collapses_blank_lines() -> None:
    assert clean("a\n\n\n\n\nb") == "a\n\nb"


def test_category_from_path_uses_top_folder() -> None:
    assert category_from_path("data/kb/finance/2025/policy.pdf") == "finance"
    assert category_from_path("knowledge_base/support/faq.md") == "support"
    assert category_from_path("data/finance/policy.pdf") == "finance"


def test_category_from_path_slug_folder() -> None:
    # slug-папки ПС (новая таксономия) проходят как есть.
    assert category_from_path("data/kb/malahit/file.pdf") == "malahit"
    assert category_from_path("data/kb/tarify/2025/заявка.docx") == "tarify"


def test_category_from_path_root_file_falls_back_to_raznoe() -> None:
    # Корневой файл (data/kb/<file>) не должен возвращать имя файла как категорию.
    assert category_from_path("data/kb/file.pdf") == "raznoe"
    assert category_from_path("data/kb/ГУИТ_ДТ.docx") == "raznoe"


def test_category_from_path_defaults_to_raznoe() -> None:
    assert category_from_path("/tmp/loose_file.pdf") == "raznoe"
    assert category_from_path("data") == "raznoe"
    assert category_from_path("data/kb") == "raznoe"


def test_category_from_path_with_corpus_root() -> None:
    """С корнем корпуса категория — первый сегмент ОТНОСИТЕЛЬНО корня.

    Живой прогон 09.10: у демо-корпуса (`data/demo_kb/<категория>/...`) без
    корня якорь «data» делал категорией «demo_kb» — фильтр RAG по ПС ломался,
    `/ask` с любой темой отвечал отказом.
    """
    assert (
        category_from_path("data/demo_kb/malahit/doc.pdf", corpus_root="data/demo_kb")
        == "malahit"
    )
    assert (
        category_from_path("data/kb/tarify/2025/doc.docx", corpus_root="data/kb")
        == "tarify"
    )
    # Файл в корне корпуса — категории нет.
    assert category_from_path("data/demo_kb/doc.pdf", corpus_root="data/demo_kb") == "raznoe"
    # Путь вне корня — fallback на эвристику по якорям.
    assert category_from_path("/tmp/loose.pdf", corpus_root="data/demo_kb") == "raznoe"


def test_doc_type_from_path() -> None:
    assert doc_type_from_path("a/b/policy.PDF") == "pdf"
    assert doc_type_from_path("note.md") == "md"
    assert doc_type_from_path("no_ext") == "unknown"


def test_version_from_filename() -> None:
    assert version_from_filename("policy_2025_v3.pdf") == "2025_v3"
    assert version_from_filename("plain_doc.pdf") == "unversioned"


def test_file_metadata_has_filter_fields(tmp_path) -> None:
    p = tmp_path / "onboarding_2025_v2.docx"
    p.write_text("x", encoding="utf-8")
    meta = file_metadata(str(p))
    assert meta["doc_type"] == "docx"
    assert meta["version"] == "2025_v2"
    assert meta["visibility"] == "internal"
    assert meta["source"] == "onboarding_2025_v2.docx"
    assert "last_modified" in meta


def test_enrich_cleans_text_and_excludes_technical_keys() -> None:
    docs = [Document(text="Тариф Стр. 3 из 9 описан тут", metadata={"category": "Тарифы"})]
    out = enrich(docs)
    assert "Стр. 3 из 9" not in out[0].text
    assert out[0].excluded_embed_metadata_keys == EXCLUDED_EMBED_KEYS
    assert out[0].excluded_llm_metadata_keys == EXCLUDED_EMBED_KEYS


def test_excluded_keys_cover_noise_fields() -> None:
    # Технические поля не должны попадать в эмбеддинг; category — остаётся.
    assert "page" in EXCLUDED_EMBED_KEYS
    assert "source" in EXCLUDED_EMBED_KEYS
    assert "version" in EXCLUDED_EMBED_KEYS
    assert "pdf_type" in EXCLUDED_EMBED_KEYS
    assert "extraction_tool" in EXCLUDED_EMBED_KEYS
    assert "category" not in EXCLUDED_EMBED_KEYS


def test_doc_id_is_deterministic_and_unique() -> None:
    """Стабильный doc_id — залог идемпотентности UPSERTS."""
    p = Path("data/kb/Тарифы/заявка.pdf")
    # одинаковый путь+страница → одинаковый id (между запусками)
    assert _doc_id(p, 3) == _doc_id(p, 3)
    # разные страницы → разные id
    assert _doc_id(p, 1) != _doc_id(p, 2)
    # без страницы (DOCX/MD/HTML) — стабилен и не равен страничному
    assert _doc_id(p, None) == _doc_id(p, None)
    assert _doc_id(p, None) != _doc_id(p, 1)


# --- PDF-движок (pdf-inspector) -----------------------------------------------


class _FakePage:
    """Заглушка `PageMarkdown` из pdf-inspector."""

    def __init__(self, page: int, markdown: str, needs_ocr: bool = False) -> None:
        self.page = page
        self.markdown = markdown
        self.needs_ocr = needs_ocr
        self.ocr_reason = None


class _FakeInspector:
    """Заглушка модуля pdf_inspector — только используемые функции."""

    def __init__(self, pages: list[_FakePage], pdf_type: str = "text_based") -> None:
        self._pages = pages
        self._pdf_type = pdf_type

    def detect_pdf(self, path: str) -> SimpleNamespace:
        return SimpleNamespace(
            pdf_type=self._pdf_type, confidence=1.0, page_count=len(self._pages)
        )

    def extract_pages_markdown(self, path: str) -> SimpleNamespace:
        return SimpleNamespace(pages=self._pages)


def test_pages_from_inspector_builds_documents_per_page(monkeypatch) -> None:
    """Страницы → Document'ы с 1-indexed `page` и стабильным doc_id."""
    path = Path("data/kb/malahit/3354.pdf")
    fake = _FakeInspector([_FakePage(0, "# Титул"), _FakePage(1, "Основной текст")])
    monkeypatch.setattr(ingestion, "_load_pdf_inspector", lambda: fake)

    docs = ingestion.pages_from_inspector(path, {"category": "malahit"})

    assert docs is not None
    assert [d.metadata["page"] for d in docs] == [1, 2]
    assert docs[0].metadata["category"] == "malahit"
    assert docs[0].metadata["pdf_type"] == "text_based"
    assert docs[0].metadata["extraction_tool"] == "pdf-inspector"
    # doc_id тот же, что у legacy-движка: откат флагом не ломает UPSERTS
    assert docs[0].doc_id == _doc_id(path, 1)
    assert docs[1].text.startswith("Основной текст")


def test_pages_from_inspector_skips_pages_without_text_layer(monkeypatch) -> None:
    """Страницы под OCR в индекс не идут — пустые ноды шумят в векторном поиске."""
    fake = _FakeInspector(
        [_FakePage(0, "текст"), _FakePage(1, "", needs_ocr=True), _FakePage(2, "   ")],
        pdf_type="mixed",
    )
    monkeypatch.setattr(ingestion, "_load_pdf_inspector", lambda: fake)

    docs = ingestion.pages_from_inspector(Path("scan.pdf"), {})

    assert [d.metadata["page"] for d in docs] == [1]


def test_pages_from_inspector_returns_none_without_package(monkeypatch) -> None:
    """Пакета нет — вызывающий откатывается на PyMuPDFReader."""
    monkeypatch.setattr(ingestion, "_load_pdf_inspector", lambda: None)

    assert ingestion.pages_from_inspector(Path("scan.pdf"), {}) is None
