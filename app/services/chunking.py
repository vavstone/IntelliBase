"""Три стратегии чанкинга для RAG-эксперимента (ДЗ 5.4).

Один корпус → три набора нод, которые индексируются в отдельные коллекции
Qdrant (docs_fixed / docs_recursive / docs_semantic) и сравниваются на
golden dataset по retrieval-метрикам (см. app/services/retrieval_eval.py).

Стратегии:
- fixed_size — TokenTextSplitter, жёсткая нарезка по токенам без учёта границ
  предложений (baseline);
- recursive   — SentenceSplitter с paragraph_separator="\\n\\n" и
  chunking_tokenizer_fn под русские предложения (аналог
  RecursiveCharacterTextSplitter из LangChain);
- semantic    — SemanticSplitterNodeParser, режет по смене темы через
  эмбеддинги соседних предложений (buffer_size=1, percentile=95).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Sequence

from llama_index.core import Document, SimpleDirectoryReader
from llama_index.core.node_parser import (
    SemanticSplitterNodeParser,
    SentenceSplitter,
    TokenTextSplitter,
)
from llama_index.core.schema import BaseNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# E5-префиксы обязательны для intfloat/multilingual-e5-large (иначе качество падает).
QUERY_INSTRUCTION = "query: "
TEXT_INSTRUCTION = "passage: "

# Граница предложения: пунктуация . ! ? … + пробел + заглавная/цифра/скобка/кавычка.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+(?=[А-ЯЁA-Z0-9«\"'(])")


def _russian_sentences(text: str) -> list[str]:
    """Разбивает русский текст на предложения (без nltk/spacy).

    Простая эвристика по границам предложений — достаточна для официальных
    документов домена ФТС. Аббревиатуры с точкой («т.е.», «пр.») не распознаются
    как конец предложения только в редких случаях; для baseline-эксперимента
    точности хватает.
    """
    parts = _SENTENCE_SPLIT_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def build_embed_model(model_name: str) -> HuggingFaceEmbedding:
    """Embed-модель, та же что и в индексе (E5, префиксы, нормализация)."""
    return HuggingFaceEmbedding(
        model_name=model_name,
        query_instruction=QUERY_INSTRUCTION,
        text_instruction=TEXT_INSTRUCTION,
        normalize=True,
    )


def load_documents(data_dir: str | Path, recursive: bool = True) -> list[Document]:
    """Читает корпус (PDF/DOCX/MD) через SimpleDirectoryReader."""
    return SimpleDirectoryReader(
        input_dir=str(data_dir),
        recursive=recursive,
    ).load_data()


def fixed_size(
    documents: Sequence[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[BaseNode]:
    """Baseline: строгая нарезка по токенам, без учёта границ предложений."""
    splitter = TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator=" ",
    )
    return splitter.get_nodes_from_documents(list(documents))


def recursive(
    documents: Sequence[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    tokenizer_fn: Callable[[str], list[str]] | None = None,
) -> list[BaseNode]:
    """Рекурсивный сплиттер по русским предложениям (абзац → предложение)."""
    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        paragraph_separator="\n\n",
        chunking_tokenizer_fn=tokenizer_fn or _russian_sentences,
    )
    return splitter.get_nodes_from_documents(list(documents))


def semantic(
    documents: Sequence[Document],
    embed_model: HuggingFaceEmbedding,
    buffer_size: int = 1,
    breakpoint_percentile_threshold: int = 95,
) -> list[BaseNode]:
    """Семантический сплиттер: режет там, где косинусное расстояние между
    соседними предложениями резко растёт (смена темы).

    embed_model — та же модель, что и в индексе. Стоимость: по одному
    embedding-вызову на каждое предложение — на больших корпусах заметна.
    """
    splitter = SemanticSplitterNodeParser(
        buffer_size=buffer_size,
        breakpoint_percentile_threshold=breakpoint_percentile_threshold,
        embed_model=embed_model,
    )
    return splitter.get_nodes_from_documents(list(documents))


def chunk_stats(nodes: Sequence[BaseNode], n_documents: int) -> dict:
    """Статистика нарезки: всего чанков, среднее чанков/документ, средняя длина."""
    total = len(nodes)
    avg_len = (
        sum(len(n.get_content()) for n in nodes) / total if total else 0.0
    )
    return {
        "total_chunks": total,
        "avg_chunks_per_doc": round(total / n_documents, 2) if n_documents else 0.0,
        "avg_chunk_len_chars": round(avg_len, 1),
    }
