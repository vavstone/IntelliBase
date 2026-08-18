"""RAG на LlamaIndex: широкий retrieval, опциональный реранкинг, ответ с цитатами.

Онлайн-контур построен по принципу «retrieve → guard → synthesize»:

1. Ретривер достаёт `rag_top_k` кандидатов из Qdrant.
2. Опциональный реранкер (config-флаг, тяжёлая зависимость) пересортировывает
   и оставляет `rag_rerank_top_n`; без него — обрезка dense-топа до того же N.
3. Код-гард: если лучший score ниже порога — отдаём честный отказ, не дёргая
   LLM (быстрее и дешевле галлюцинации).
4. Иначе синтез ответа по пронумерованному контексту с цитатами [1], [2].

Индексацию делает офлайн-контур (app/services/ingestion.py); этот сервис только
ПОДКЛЮЧАЕТСЯ к готовой коллекции через `from_vector_store`.

Запуск отдельно:
    uv run python -m app.services.rag
"""

import logging
import re

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.base.llms.types import LLMMetadata, MessageRole
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI as _OpenAI
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import AsyncQdrantClient, QdrantClient

from app.core.config import Settings as AppSettings
from app.core.config import get_settings

logger = logging.getLogger(__name__)

# E5-префиксы обязательны для intfloat/multilingual-e5-large.
QUERY_INSTRUCTION = "query: "
TEXT_INSTRUCTION = "passage: "

REFUSAL_TEXT = "В базе знаний я не нашёл ответа на этот вопрос."

CITATION_QA_TEMPLATE = (
    "Ниже — пронумерованные источники из базы знаний.\n"
    "---------------------\n{context_str}\n---------------------\n"
    "Ответь на вопрос, опираясь ТОЛЬКО на источники. Каждый факт сопровождай "
    "номером источника в квадратных скобках, например [1] или [2]. Если ответа "
    "в источниках нет — честно напиши, что не нашёл его в базе знаний, и ничего "
    "не выдумывай. Отвечай по-русски, коротко и по делу.\n"
    "Вопрос: {query_str}\n"
    "Ответ: "
)


class OllamaLLM(_OpenAI):
    """OpenAI-совместимый LLM, указывающий на локальную Ollama.

    Базовый llama_index.llms.openai.OpenAI определяет context_window по имени
    модели через openai_modelname_to_contextsize и падает на имени локальной
    модели (не OpenAI). Переопределяем metadata явно, всё остальное
    (async-вызовы через api_base) наследуется.
    """

    def __init__(self, *args, context_window: int = 8192, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._context_window = context_window

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self._context_window,
            num_output=self.max_tokens or -1,
            is_chat_model=True,
            is_function_calling_model=False,
            model_name=self.model,
            system_role=MessageRole.SYSTEM,
        )


def numbered_context(nodes: list[NodeWithScore]) -> str:
    """Пронумерованный контекст для цитирования: '[1] текст', '[2] текст', ..."""
    return "\n\n".join(f"[{i}] {sn.get_content()}" for i, sn in enumerate(nodes, start=1))


def build_sources(source_nodes: list[NodeWithScore]) -> list[dict]:
    """Нумерованные цитаты [1..N]: id, file_name, page, score, snippet."""
    sources = []
    for i, sn in enumerate(source_nodes, start=1):
        meta = sn.metadata or {}
        sources.append(
            {
                "id": i,
                "file_name": meta.get("source") or meta.get("file_name") or "unknown",
                "page": meta.get("page"),
                "score": round(sn.score or 0.0, 3),
                "snippet": sn.get_content()[:200].strip(),
            }
        )
    return sources


def parse_citations(text: str, sources: list[dict]) -> str:
    """Разворачивает [1] в [1 — file.pdf], чтобы источник был виден в тексте."""
    by_id = {s["id"]: s for s in sources}

    def replace(match: re.Match) -> str:
        source = by_id.get(int(match.group(1)))
        return f"[{match.group(1)} — {source['file_name']}]" if source else match.group(0)

    return re.sub(r"\[(\d+)\]", replace, text)


def build_filters(
    *, visibility: str | None = "internal", categories: list[str] | None = None
) -> MetadataFilters | None:
    """Фильтр доступа до поиска: документы вне видимости даже не достаются.

    Применяется на уровне векторного хранилища, а не после ретрива — иначе
    кусок недоступного документа может попасть в контекст LLM.
    """
    filters = []
    if visibility:
        filters.append(
            MetadataFilter(key="visibility", value=visibility, operator=FilterOperator.EQ)
        )
    if categories:
        filters.append(
            MetadataFilter(key="category", value=categories, operator=FilterOperator.IN)
        )
    return MetadataFilters(filters=filters) if filters else None


class RAGService:
    """Один экземпляр на процесс: ретривер и движок собираются один раз на старте."""

    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        qdrant_key = (
            settings.qdrant_api_key.get_secret_value()
            if settings.qdrant_api_key is not None
            else None
        )

        Settings.embed_model = HuggingFaceEmbedding(
            model_name=settings.embedding.model,
            query_instruction=QUERY_INSTRUCTION,
            text_instruction=TEXT_INSTRUCTION,
            normalize=True,
        )
        Settings.llm = OllamaLLM(
            model=settings.rag_llm_model,
            temperature=0.0,
            api_key="ollama",
            api_base=settings.llm.ollama_base_url,
            timeout=settings.rag_llm_timeout,
            context_window=settings.rag_llm_context_window,
        )

        self._client = QdrantClient(url=settings.qdrant_url, api_key=qdrant_key)
        self._aclient = AsyncQdrantClient(url=settings.qdrant_url, api_key=qdrant_key)
        self._index: VectorStoreIndex | None = None
        self._retriever = None
        self._postprocessors: list = []

    def _vector_store(self) -> QdrantVectorStore:
        return QdrantVectorStore(
            collection_name=self._settings.rag_collection,
            client=self._client,
            aclient=self._aclient,
        )

    def _collection_ready(self) -> bool:
        """Коллекция существует и непуста — значит офлайн-контур уже отработал."""
        if not self._client.collection_exists(self._settings.rag_collection):
            return False
        return self._client.count(self._settings.rag_collection).count > 0

    def _build_reranker(self):
        """Реранкер — опциональная тяжёлая зависимость (sentence-transformers + torch).
        Импорт и загрузка модели только когда rag_rerank_enabled=True."""
        from llama_index.core.postprocessor import SentenceTransformerRerank

        return SentenceTransformerRerank(
            model=self._settings.rag_rerank_model,
            top_n=self._settings.rag_rerank_top_n,
        )

    def build(self) -> None:
        """Подключается к готовой коллекции, собирает ретривер.

        Коллекцию наполняет офлайн-контур (IngestionService / scripts/ingest.py);
        здесь — только чтение. Если коллекция пуста/отсутствует, бросаем
        RuntimeError: lifespan поймает и выставит rag_service=None (503).
        """
        if not self._collection_ready():
            raise RuntimeError(
                f"коллекция {self._settings.rag_collection} пуста или отсутствует — "
                f"запустите `uv run python scripts/ingest.py {self._settings.rag_data_dir}`"
            )
        self._index = VectorStoreIndex.from_vector_store(self._vector_store())
        logger.info(
            "RAG: подключён к коллекции %s (%d точек)",
            self._settings.rag_collection,
            self._client.count(self._settings.rag_collection).count,
        )
        self._retriever = self._index.as_retriever(
            similarity_top_k=self._settings.rag_top_k
        )
        if self._settings.rag_rerank_enabled:
            self._postprocessors = [self._build_reranker()]

    async def retrieve(self, question: str) -> list[NodeWithScore]:
        """Retrieval: top-k из Qdrant + опциональный реранкер + обрезка до top_n."""
        if self._retriever is None:
            raise RuntimeError("RAG-индекс не инициализирован: сначала вызвать build().")
        nodes = await self._retriever.aretrieve(question)
        for postprocessor in self._postprocessors:
            nodes = postprocessor.postprocess_nodes(nodes, query_str=question)
        return nodes[: self._settings.rag_rerank_top_n]

    async def answer(self, question: str) -> dict:
        """Контракт: {answer, top_score, sources[id,file_name,page,score,snippet], confident}."""
        nodes = await self.retrieve(question)
        top_score = max((sn.score or 0.0 for sn in nodes), default=0.0)
        if not nodes or top_score < self._settings.rag_score_threshold:
            # отказ БЕЗ вызова LLM — быстрее, дешевле, надёжнее галлюцинации
            return {
                "answer": REFUSAL_TEXT,
                "top_score": round(top_score, 3),
                "sources": [],
                "confident": False,
            }

        response = await Settings.llm.acomplete(
            CITATION_QA_TEMPLATE.format(
                context_str=numbered_context(nodes), query_str=question
            )
        )
        sources = build_sources(nodes)
        return {
            "answer": parse_citations(str(response), sources),
            "top_score": round(top_score, 3),
            "sources": sources,
            "confident": True,
        }

    async def close(self) -> None:
        try:
            await self._aclient.close()
        except Exception:
            logger.debug("ошибка при закрытии async Qdrant-клиента", exc_info=True)
        try:
            self._client.close()
        except Exception:
            logger.debug("ошибка при закрытии Qdrant-клиента", exc_info=True)


def _demo() -> None:
    import asyncio
    import json

    async def run() -> None:
        service = RAGService(get_settings())
        service.build()
        for question in (
            "Что у нас есть по заявкам для КПС Тарифы?",
            "Какая завтра погода в Москве?",
        ):
            result = await service.answer(question)
            print(f"\nВопрос: {question}")
            print(json.dumps(result, ensure_ascii=False, indent=2))
        await service.close()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    asyncio.run(run())


if __name__ == "__main__":
    _demo()
