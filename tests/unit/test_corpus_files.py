"""Юнит-тесты работы с файлами корпуса (app/services/corpus_files.py).

Проверяем две вещи: что отправить можно только документ из корпуса (никаких
путей и `..` — модель называет файл, путь собирает сервер) и что каталог
документов для «какие ФТ есть в базе» строится детерминированно.
"""

from pathlib import Path

import pytest

from app.services.corpus_files import (
    MAX_DOCUMENT_BYTES,
    list_corpus_documents,
    resolve_corpus_file,
)

DOC_NAME = "ПС Тарифы. Технические требования. Версия 2.4.docx"


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    (tmp_path / "tarify").mkdir()
    (tmp_path / "tarify" / DOC_NAME).write_text("требования", encoding="utf-8")
    (tmp_path / "tarify" / "Схема.drawio").write_text("не документ", encoding="utf-8")
    (tmp_path / "crsved").mkdir()
    (tmp_path / "crsved" / "Регламент ЦРСВЭД.pdf").write_text("регламент", encoding="utf-8")
    (tmp_path / "В корне.txt").write_text("файл в корне", encoding="utf-8")
    return tmp_path


def test_finds_document_by_name(corpus: Path) -> None:
    found = resolve_corpus_file(DOC_NAME, corpus)
    assert found is not None and found.name == DOC_NAME


def test_name_is_case_insensitive(corpus: Path) -> None:
    assert resolve_corpus_file(DOC_NAME.lower(), corpus) is not None


def test_path_is_ignored_only_name_is_used(corpus: Path) -> None:
    """Путь отбрасывается: ищем по имени файла внутри корпуса."""
    assert resolve_corpus_file(f"../../etc/{DOC_NAME}", corpus) is not None


def test_traversal_outside_corpus_is_refused(corpus: Path, tmp_path: Path) -> None:
    secret = tmp_path.parent / "secret.env"
    secret.write_text("TOKEN=1", encoding="utf-8")
    assert resolve_corpus_file("../secret.env", corpus) is None
    assert resolve_corpus_file(str(secret), corpus) is None


def test_unknown_file_is_refused(corpus: Path) -> None:
    assert resolve_corpus_file("ФТ на Тарифы-1.docx", corpus) is None


def test_non_document_suffix_is_refused(corpus: Path) -> None:
    assert resolve_corpus_file("Схема.drawio", corpus) is None


def test_oversized_file_is_refused(corpus: Path, monkeypatch) -> None:
    monkeypatch.setattr("app.services.corpus_files.MAX_DOCUMENT_BYTES", 4)
    assert resolve_corpus_file(DOC_NAME, corpus) is None
    assert MAX_DOCUMENT_BYTES == 50 * 1024 * 1024  # исходный лимит не тронут


def test_list_documents_groups_by_category(corpus: Path) -> None:
    docs = list_corpus_documents(corpus)
    names = {d["file_name"] for d in docs}
    assert DOC_NAME in names
    assert "Регламент ЦРСВЭД.pdf" in names
    assert "Схема.drawio" not in names  # не документ
    categories = {d["category"] for d in docs}
    assert categories == {"tarify", "crsved", "raznoe"}  # файл в корне — raznoe


def test_list_documents_filters_by_query(corpus: Path) -> None:
    assert [d["file_name"] for d in list_corpus_documents(corpus, "тарифы")] == [DOC_NAME]
    assert list_corpus_documents(corpus, "тарифы") == list_corpus_documents(corpus, "ТАРИФЫ")
    # Фильтр работает и по категории.
    assert len(list_corpus_documents(corpus, "crsved")) == 1
    assert list_corpus_documents(corpus, "нет-такого") == []


def test_list_documents_respects_limit(corpus: Path) -> None:
    assert len(list_corpus_documents(corpus, limit=1)) == 1


def test_list_documents_of_missing_root_is_empty(tmp_path: Path) -> None:
    assert list_corpus_documents(tmp_path / "нет-такой-папки") == []


# ── имя без расширения (как его обычно пишет человек) ────────────────────


def test_finds_document_by_name_without_suffix(corpus: Path) -> None:
    """«ПС Тарифы. Технические требования. Версия 2.4» → файл с .docx."""
    found = resolve_corpus_file(DOC_NAME.removesuffix(".docx"), corpus)
    assert found is not None and found.name == DOC_NAME


def test_ambiguous_stem_is_refused(tmp_path: Path) -> None:
    """Одинаковый ствол у pdf и docx — угадывать нельзя, лучше отказ."""
    (tmp_path / "tarify").mkdir()
    (tmp_path / "tarify" / "Требования.pdf").write_text("a", encoding="utf-8")
    (tmp_path / "tarify" / "Требования.docx").write_text("b", encoding="utf-8")
    assert resolve_corpus_file("Требования", tmp_path) is None
    # С расширением — по-прежнему однозначно.
    assert resolve_corpus_file("Требования.pdf", tmp_path) is not None


def test_non_document_is_not_found_even_by_stem(corpus: Path) -> None:
    assert resolve_corpus_file("Схема", corpus) is None
