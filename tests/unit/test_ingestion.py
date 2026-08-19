"""Юнит-тесты чистых функций офлайн-контура индексации (без Qdrant/embedding)."""

from llama_index.core.schema import Document

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
    assert "category" not in EXCLUDED_EMBED_KEYS


def test_doc_id_is_deterministic_and_unique() -> None:
    """Стабильный doc_id — залог идемпотентности UPSERTS."""
    from pathlib import Path

    p = Path("data/kb/Тарифы/заявка.pdf")
    # одинаковый путь+страница → одинаковый id (между запусками)
    assert _doc_id(p, 3) == _doc_id(p, 3)
    # разные страницы → разные id
    assert _doc_id(p, 1) != _doc_id(p, 2)
    # без страницы (DOCX/MD/HTML) — стабилен и не равен страничному
    assert _doc_id(p, None) == _doc_id(p, None)
    assert _doc_id(p, None) != _doc_id(p, 1)
