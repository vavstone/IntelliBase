"""RAG на LlamaIndex: индексация корпуса в Qdrant и ответ с цитатами.

Полный pipeline, инкапсулированный фреймворком:
SimpleDirectoryReader → SentenceSplitter → QdrantVectorStore → VectorStoreIndex
→ QueryEngine. Индекс строится один раз (build), запросы — через answer().

Запуск отдельно:
    uv run python -m app.services.rag
"""

import logging

from llama_index.core import (
    PromptTemplate,
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.base.llms.types import LLMMetadata, MessageRole
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI as _OpenAI
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import AsyncQdrantClient, QdrantClient

from app.core.config import Settings as AppSettings
from app.core.config import get_settings

logger = logging.getLogger(__name__)

# E5-префиксы обязательны для intfloat/multilingual-e5-large (иначе качество падает).
QUERY_INSTRUCTION = "query: "
TEXT_INSTRUCTION = "passage: "


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

QA_PROMPT = PromptTemplate(
    "Ниже приведён контекст из базы знаний.\n"
    "---------------------\n{context_str}\n---------------------\n"
    "Ответь на вопрос, опираясь ТОЛЬКО на контекст. Если ответа в контексте нет — "
    "честно напиши, что не нашёл ответа в базе знаний, и ничего не выдумывай. "
    "Отвечай по-русски, коротко и по делу.\n"
    "Вопрос: {query_str}\n"
    "Ответ: "
)


def format_result(response, threshold: float) -> dict:
    """Приводит ответ LlamaIndex к контракту {answer, top_score, sources}.

    Если top-1 score ниже порога — считаем, что ответа в корпусе нет, и отдаём
    честный fallback вместо сгенерированного текста.
    """
    nodes = response.source_nodes
    top_score = max((node.score or 0.0 for node in nodes), default=0.0)
    if top_score < threshold:
        answer_text = "В базе знаний нет ответа на этот вопрос."
    else:
        answer_text = str(response)
    return {
        "answer": answer_text,
        "top_score": round(top_score, 3),
        "sources": [
            {
                "text": node.text[:300],
                "source": node.metadata.get("file_name"),
                "score": round(node.score or 0.0, 3),
            }
            for node in nodes
        ],
    }


class RAGService:
    """Один экземпляр на процесс: индекс строится один раз на старте."""

    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        qdrant_key = (
            settings.qdrant_api_key.get_secret_value()
            if settings.qdrant_api_key is not None
            else None
        )

        # Глобальная конфигурация LlamaIndex — задаётся ДО построения индексов.
        Settings.embed_model = HuggingFaceEmbedding(
            model_name=settings.embedding.model,
            query_instruction=QUERY_INSTRUCTION,
            text_instruction=TEXT_INSTRUCTION,
            normalize=True,
        )
        # Ollama через OpenAI-совместимый эндпоинт (тот же приём, что в main.py).
        Settings.llm = OllamaLLM(
            model=settings.rag_llm_model,
            temperature=0.0,
            api_key="ollama",
            api_base=settings.llm.ollama_base_url,
        )
        Settings.node_parser = SentenceSplitter(
            chunk_size=settings.rag_chunk_size,
            chunk_overlap=settings.rag_chunk_overlap,
        )

        self._client = QdrantClient(url=settings.qdrant_url, api_key=qdrant_key)
        self._aclient = AsyncQdrantClient(url=settings.qdrant_url, api_key=qdrant_key)
        self._index: VectorStoreIndex | None = None
        self._engine = None

    def _vector_store(self) -> QdrantVectorStore:
        return QdrantVectorStore(
            collection_name=self._settings.rag_collection,
            client=self._client,
            aclient=self._aclient,
        )

    def _collection_ready(self) -> bool:
        """Коллекция существует и непуста — значит индексировать заново не нужно."""
        if not self._client.collection_exists(self._settings.rag_collection):
            return False
        return self._client.count(self._settings.rag_collection).count > 0

    def build(self) -> None:
        """Подключается к готовой коллекции либо индексирует корпус из файлов.

        Идемпотентно: при непустой коллекции идём по ветке from_vector_store
        без переиндексации, поэтому повторный запуск не дублирует точки.
        """
        vector_store = self._vector_store()
        if self._collection_ready():
            self._index = VectorStoreIndex.from_vector_store(vector_store)
            logger.info(
                "RAG: подключён к коллекции %s (%d точек)",
                self._settings.rag_collection,
                self._client.count(self._settings.rag_collection).count,
            )
        else:
            documents = SimpleDirectoryReader(
                input_dir=str(self._settings.rag_data_dir),
                recursive=True,
            ).load_data()
            storage = StorageContext.from_defaults(vector_store=vector_store)
            self._index = VectorStoreIndex.from_documents(documents, storage_context=storage)
            logger.info(
                "RAG: проиндексировано %d документов в коллекцию %s",
                len(documents),
                self._settings.rag_collection,
            )

        self._engine = self._index.as_query_engine(
            similarity_top_k=self._settings.rag_top_k,
            text_qa_template=QA_PROMPT,
        )

    async def answer(self, question: str) -> dict:
        if self._engine is None:
            raise RuntimeError("RAG-индекс не инициализирован: сначала вызвать build().")
        response = await self._engine.aquery(question)
        return format_result(response, self._settings.rag_score_threshold)

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
            "Что входит в состав КПС Тарифы «Реестр ОИС»?",
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
