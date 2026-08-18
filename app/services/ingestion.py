"""Офлайн-контур RAG: парсинг корпуса, обогащение метаданными, индексация.

Индексация вынесена из онлайн-запроса: `IngestionPipeline` c
`DocstoreStrategy.UPSERTS` и docstore, который сохраняется на диск. Повторный
прогон обновляет только изменённые документы (дедуп по `doc.id_` + `doc.hash`),
а не переэмбеддит весь корпус. Запись идёт в ту же коллекцию Qdrant, из которой
читает `RAGService` (онлайн-контур).

Парсинг маршрутизируется по расширению на специализированные ридеры:
`PyMuPDFReader` (PDF, по странице — даёт `page` для цитат), `DocxReader`,
`HTMLTagReader`, `MarkdownReader`. Метаданные из путей (`category`, `version`)
и файла (`last_modified`) обогащают ноды для фильтрации и цитирования.

Чистые функции (`clean`, `category_from_path`, `file_metadata`, ...) не зависят
от внешних сервисов и покрыты юнит-тестами; класс `IngestionService` ходит в
Qdrant и embedding-модель и проверяется на живых сервисах через scripts/ingest.py.

Внимание к идемпотентности: в метаданные НЕ кладём поле, меняющееся от запуска
к запуску (типа `indexed_at=date.today()`), иначе `doc.hash` меняется ежедневно и
UPSERTS переиндексирует весь корпус. `last_modified` берём из stat-файла — он
стабилен, пока файл не менялся.
"""

from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from llama_index.core import Document
from llama_index.core.ingestion import DocstoreStrategy, IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file import (
    DocxReader,
    HTMLTagReader,
    MarkdownReader,
    PyMuPDFReader,
)
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from app.core.config import Settings as AppSettings

logger = logging.getLogger(__name__)

SUPPORTED_EXTS = (".pdf", ".docx", ".html", ".md")

# E5-префиксы обязательны для intfloat/multilingual-e5-large (иначе качество падает).
QUERY_INSTRUCTION = "query: "
TEXT_INSTRUCTION = "passage: "

# Технические поля не несут смысла для эмбеддинга — они нужны только как фильтры
# и для отображения источника. Если их не исключить, LlamaIndex подмешает их в
# текст ноды и зашумит вектор. `category` намеренно оставляем в эмбеддинге: это
# осмысленная семантическая метка, помогающая поиску.
EXCLUDED_EMBED_KEYS = [
    "file_path",
    "file_name",
    "source",
    "file_type",
    "file_size",
    "total_pages",
    "page",
    "creation_date",
    "last_modified_date",
    "last_modified",
    "doc_type",
    "version",
    "visibility",
]

# Якоря корпуса: категория = папка сразу после одного из них.
# Порядок важен: 'kb'/'knowledge_base' должны матчиться раньше 'data',
# иначе путь 'data/kb/<cat>/...' вернёт 'kb' вместо '<cat>'.
_CATEGORY_ANCHORS = ("knowledge_base", "kb", "data")

# Пространство имён для стабильного doc_id. LlamaIndex по умолчанию даёт
# Document случайный id_ (uuid4), поэтому UPSERTS не смог бы сопоставить один
# и тот же документ между запусками. Мы явно выставляем детерминированный id
# из пути (+ страницы) — он стабилен, пока файл на месте, и не меняется от
# `date.today()`/счётчиков.
_DOC_NAMESPACE = uuid.UUID("c0ffee00-0000-0000-0000-c0ffee000001")


def _doc_id(path: Path, page: int | None) -> str:
    """Детерминированный id документа: uuid5 от пути (+ страницы)."""
    key = f"{path}::{page}" if page is not None else str(path)
    return str(uuid.uuid5(_DOC_NAMESPACE, key))


def clean(text: str) -> str:
    """Снимает типичный шум PDF-экспорта перед чанкингом.

    Колонтитулы и номера страниц попадают в каждый чанк и зашумляют эмбеддинг;
    перенос слова по дефису на конце строки рвёт токен («авто-\\nмобиль»).
    """
    text = re.sub(r"Стр\.\s*\d+\s*из\s*\d+", "", text)
    text = re.sub(r"-\n(\w)", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"https?://\S+", "", text)
    return text.strip()


def category_from_path(path: str) -> str:
    """`data/kb/finance/2025/policy.pdf` -> `finance` (папка верхнего уровня корпуса)."""
    parts = Path(path).parts
    for anchor in _CATEGORY_ANCHORS:
        if anchor in parts:
            idx = parts.index(anchor)
            return parts[idx + 1] if len(parts) > idx + 1 else "general"
    return "general"


def doc_type_from_path(path: str) -> str:
    """Тип документа из расширения файла: `pdf`, `docx`, `md`, ..."""
    return Path(path).suffix.lstrip(".").lower() or "unknown"


def version_from_filename(path: str) -> str:
    """`policy_2025_v3.pdf` -> `2025_v3`; если версии нет — `unversioned`."""
    match = re.search(r"(20\d{2}(?:[_-]v?\d+)?)", Path(path).stem)
    return match.group(1) if match else "unversioned"


def file_metadata(path: str) -> dict[str, str]:
    """Метаданные на этапе загрузки: источник, категория, тип, версия, дата.

    `last_modified` берём из stat-файла — стабильно между запусками, в отличие
    от `indexed_at=date.today()`, который сломал бы идемпотентность UPSERTS.
    """
    p = Path(path)
    try:
        mtime = p.stat().st_mtime
    except OSError:
        mtime = 0.0
    return {
        "source": p.name,
        "category": category_from_path(path),
        "doc_type": doc_type_from_path(path),
        "version": version_from_filename(path),
        "visibility": "internal",
        "last_modified": datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat(),
    }


def enrich(documents: list[Document]) -> list[Document]:
    """Чистит текст и помечает технические поля исключёнными из эмбеддинга.

    Метаданные уже проставлены в `_read_file`; здесь — финальная нормализация
    документа перед чанкингом.
    """
    for doc in documents:
        # Document.text — read-only property поверх text_resource, пишем через set_content.
        doc.set_content(clean(doc.text))
        doc.excluded_embed_metadata_keys = EXCLUDED_EMBED_KEYS
        doc.excluded_llm_metadata_keys = EXCLUDED_EMBED_KEYS
    return documents


def build_embed_model(model_name: str) -> HuggingFaceEmbedding:
    """Embed-модель та же, что и в онлайн-контуре (E5, префиксы, нормализация)."""
    return HuggingFaceEmbedding(
        model_name=model_name,
        query_instruction=QUERY_INSTRUCTION,
        text_instruction=TEXT_INSTRUCTION,
        normalize=True,
    )


class IngestionService:
    """Индексатор корпуса: один экземпляр на процесс, переиспользует пайплайн.

    Docstore сохраняется на диск (`rag_docstore_path`), что делает UPSERTS
    идемпотентным между запусками: неизменённые документы пропускаются.
    """

    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        qdrant_key = (
            settings.qdrant_api_key.get_secret_value()
            if settings.qdrant_api_key is not None
            else None
        )
        self._data_dir = Path(settings.rag_data_dir)
        self._docstore_path = Path(settings.rag_docstore_path)

        self._client = QdrantClient(url=settings.qdrant_url, api_key=qdrant_key)
        self._vector_store = QdrantVectorStore(
            client=self._client,
            collection_name=settings.rag_collection,
        )
        self._embed_model = build_embed_model(settings.embedding.model)
        self._docstore = self._load_docstore()
        self._pipeline = self._build_pipeline()

    @property
    def docstore_path(self) -> Path:
        """Путь к сохранённому docstore — состояние инкрементальной индексации."""
        return self._docstore_path

    def _build_pipeline(self) -> IngestionPipeline:
        return IngestionPipeline(
            transformations=[
                SentenceSplitter(
                    chunk_size=self._settings.rag_chunk_size,
                    chunk_overlap=self._settings.rag_chunk_overlap,
                ),
                self._embed_model,
            ],
            docstore=self._docstore,
            vector_store=self._vector_store,
            docstore_strategy=DocstoreStrategy.UPSERTS,
        )

    def _load_docstore(self) -> SimpleDocumentStore:
        if self._docstore_path.exists():
            return SimpleDocumentStore.from_persist_path(str(self._docstore_path))
        return SimpleDocumentStore()

    def _persist_docstore(self) -> None:
        self._docstore_path.parent.mkdir(parents=True, exist_ok=True)
        self._docstore.persist(str(self._docstore_path))

    def is_collection_empty(self) -> bool:
        """Коллекции нет или она пуста — значит нужна первичная индексация."""
        if not self._client.collection_exists(self._settings.rag_collection):
            return True
        return self._client.count(self._settings.rag_collection).count == 0

    def _collect_files(self, input_files: list[Path] | None = None) -> list[Path]:
        if input_files:
            return [Path(p) for p in input_files]
        return [
            p
            for p in sorted(self._data_dir.rglob("*"))
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
        ]

    def _read_file(self, path: Path) -> list[Document]:
        """Загружает один файл нужным ридером, проставляет метаданные."""
        suffix = path.suffix.lower()
        base = file_metadata(str(path))

        if suffix == ".pdf":
            # PyMuPDFReader отдаёт один Document на страницу; в metadata['source']
            # у него лежит номер страницы (строка). Сохраняем его в 'page'.
            docs = PyMuPDFReader().load(file_path=str(path))
            for i, doc in enumerate(docs, start=1):
                raw = str(doc.metadata.get("source", ""))
                page = int(raw) if raw.isdigit() else i
                doc.metadata = {**base, "page": page}
                doc.doc_id = _doc_id(path, page)  # стабильный id для UPSERTS
            return docs

        if suffix == ".docx":
            docs = DocxReader().load_data(file=path)
        elif suffix == ".html":
            docs = HTMLTagReader().load_data(file=path)
        elif suffix == ".md":
            docs = MarkdownReader().load_data(file=str(path))
        else:
            return []

        for idx, doc in enumerate(docs):
            doc.metadata = dict(base)
            # Ридер может вернуть несколько Document на файл (секции по заголовкам),
            # поэтому id уникален по индексу секции — иначе все секции получили бы
            # один doc_id и UPSERTS не смог бы их различить.
            doc.doc_id = _doc_id(path, idx)
        return docs

    def _read(self, input_files: list[Path] | None = None) -> list[Document]:
        """Читает корпус (или перечисленные файлы) с изоляцией упавших.

        Упавший файл логируется и переименовывается в `.failed` — не попадает
        в индекс и виден в логах.
        """
        documents: list[Document] = []
        for path in self._collect_files(input_files):
            try:
                documents.extend(self._read_file(path))
            except Exception:
                logger.exception("ingestion: не удалось прочитать file=%s", path.name)
                try:
                    path.rename(path.with_suffix(path.suffix + ".failed"))
                except OSError:
                    pass
        return enrich(documents)

    def _changed_unchanged(self, documents: list[Document]) -> tuple[int, int]:
        """Сколько документов изменены/новы против уже проиндексированных.

        Повторяет логику `_handle_upserts` IngestionPipeline: документ считается
        неизменённым, если в docstore по `doc.id_` лежит тот же `doc.hash`.
        """
        changed = unchanged = 0
        for doc in documents:
            existing = self._docstore.get_document_hash(doc.doc_id)
            if existing is None or existing != doc.hash:
                changed += 1
            else:
                unchanged += 1
        return changed, unchanged

    def ingest_all(self) -> dict:
        """Полная (инкрементальная) переиндексация корпуса.

        UPSERTS пропустит неизменённое. Возвращает сводку
        {changed, unchanged, nodes} для отчёта и самопроверки.
        """
        documents = self._read()
        changed, unchanged = self._changed_unchanged(documents)
        nodes = self._pipeline.run(documents=documents, show_progress=True)
        self._persist_docstore()
        logger.info(
            "ingestion: корпус проиндексирован документов=%d нод=%d "
            "changed=%d unchanged=%d",
            len(documents),
            len(nodes),
            changed,
            unchanged,
        )
        return {"changed": changed, "unchanged": unchanged, "nodes": len(nodes)}

    def ingest_files(self, paths: list[str]) -> int:
        """Точечная индексация перечисленных файлов (webhook документооборота)."""
        files = [Path(p) for p in paths if Path(p).exists()]
        if not files:
            return 0
        documents = self._read(input_files=files)
        nodes = self._pipeline.run(documents=documents, show_progress=False)
        self._persist_docstore()
        logger.info(
            "ingestion: точечно проиндексировано файлов=%d нод=%d", len(files), len(nodes)
        )
        return len(nodes)

    def reindex_all(self) -> int:
        """Полная переиндексация: вычищаем коллекцию и docstore, строим заново.

        Нужна после смены схемы метаданных или модели эмбеддингов, когда
        инкрементальный UPSERTS по хешам уже не отражает реальное состояние.
        """
        if self._client.collection_exists(self._settings.rag_collection):
            self._client.delete_collection(self._settings.rag_collection)
        self._docstore = SimpleDocumentStore()
        self._docstore_path.unlink(missing_ok=True)
        self._pipeline = self._build_pipeline()
        logger.info(
            "ingestion: полная переиндексация коллекции %s", self._settings.rag_collection
        )
        return self.ingest_all()["nodes"]

    def run_for_file(self, path: Path) -> int:
        """Индексация одного загруженного файла. Упавший файл изолируется в `.failed`."""
        try:
            count = self.ingest_files([str(path)])
            logger.info("ingestion: файл проиндексирован file=%s нод=%d", path.name, count)
            return count
        except Exception:
            logger.exception("ingestion: файл не проиндексирован file=%s", path.name)
            try:
                path.rename(path.with_suffix(path.suffix + ".failed"))
            except OSError:
                pass
            return 0

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            logger.debug("ошибка при закрытии Qdrant-клиента ingestion", exc_info=True)
