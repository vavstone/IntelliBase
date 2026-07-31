"""
Модуль app/services/embeddings.py

Предоставляет асинхронный сервис для генерации векторных представлений текстов.
Поддерживает бэкенды OpenAI и sentence-transformers, автоматический батчинг,
повторные попытки при сетевых ошибках, нормализацию векторов и кеширование
(in-memory + опционально diskcache для персистентности между рестартами).

Для моделей семейства E5 реализованы отдельные методы embed_query и embed_documents
с добавлением соответствующих префиксов.

Кеширование:
- Ключ = sha256(f"{model}:{dimensions}:{text}")
- In-memory LRU-подобный кеш (всегда включён)
- diskcache — опционально, для сохранения кеша между рестартами
- При смене модели кеш автоматически инвалидируется (ключ содержит model + dimensions)
"""

import asyncio
import hashlib
import logging
import os
import random
from collections.abc import Sequence
from typing import List, Optional, Union

from openai import AsyncOpenAI, APIError, APIConnectionError, RateLimitError
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Конфигурация повторных попыток
DEFAULT_MAX_RETRIES = 5
DEFAULT_BACKOFF_BASE = 1.0
DEFAULT_BACKOFF_MAX = 30.0

# Максимальный размер in-memory кеша (кол-во записей).
# При превышении — удаляются старейшие записи (FIFO-подобно).
DEFAULT_MAX_CACHE_SIZE = 10_000


class EmbeddingsService:
    """
    Асинхронный сервис для получения эмбеддингов текстов.

    Поддерживает два провайдера:
        - 'openai'   : использует API OpenAI (требуется ключ)
        - 'sentence_transformers' : локальная модель (например, 'all-MiniLM-L6-v2')

    Параметры:
        provider (str): 'openai' или 'sentence_transformers'.
        model (str): имя модели (для OpenAI – идентификатор, для ST – имя модели).
        batch_size (Optional[int]): размер батча. Если не задан, выбирается разумное
            значение по умолчанию (100 для OpenAI, 16 для ST).
        api_key (Optional[str]): API ключ OpenAI (если не указан, берётся из
            переменной окружения OPENAI_API_KEY).
        max_retries (int): максимальное число повторных попыток при сетевых ошибках.
        backoff_base (float): базовая задержка для экспоненциального ожидания.
        backoff_max (float): максимальная задержка.
        cache_dir (Optional[str]): путь к директории diskcache. Если указан —
            кеш сохраняется между рестартами. Если None — только in-memory.
        max_cache_size (int): максимальный размер in-memory кеша.

    Методы:
        embed_texts(texts: Sequence[str]) -> List[List[float]]:
            Возвращает векторы для всех текстов в том же порядке.

        embed_one(text: str) -> List[float]:
            Возвращает вектор для одного текста.

        embed_query(text: str) -> List[float]:
            Для E5-моделей добавляет префикс "query: " и возвращает вектор.

        embed_documents(texts: Sequence[str]) -> List[List[float]]:
            Для E5-моделей добавляет префикс "passage: " каждому тексту и возвращает векторы.

        cache_stats() -> dict:
            Статистика кеша: hits, misses, size.
    """

    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        batch_size: Optional[int] = None,
        api_key: Optional[str] = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_base: float = DEFAULT_BACKOFF_BASE,
        backoff_max: float = DEFAULT_BACKOFF_MAX,
        cache_dir: Optional[str] = None,
        max_cache_size: int = DEFAULT_MAX_CACHE_SIZE,
    ) -> None:
        self.provider = provider.lower()
        self.model = model
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self.max_cache_size = max_cache_size

        # Размерность будет определена после первого вызова
        self._dimensions: Optional[int] = None

        # In-memory кеш: cache_key -> list[float] (для одного текста)
        # Храним отдельные тексты, не батчи — так повторный вызов embed_texts(["тот же текст"])
        # найдёт каждый текст в кеше независимо от состава батча.
        self._cache: dict[str, list[float]] = {}
        self._cache_order: list[str] = []  # для FIFO-вытеснения
        self._cache_hits: int = 0
        self._cache_misses: int = 0

        # Опциональный дисковый кеш (persistent across restarts)
        self._disk_cache: Optional[object] = None
        if cache_dir is not None:
            try:
                import diskcache
                self._disk_cache = diskcache.Cache(cache_dir)
                logger.info("diskcache initialised at %s", cache_dir)
            except ImportError:
                logger.warning("diskcache not installed — persistent cache disabled")
            except Exception as exc:
                logger.warning("diskcache init failed (%s) — persistent cache disabled", exc)

        # Установка размера батча по умолчанию
        if batch_size is None:
            if self.provider == "openai":
                self.batch_size = 100   # рекомендуемый диапазон 100–512
            elif self.provider == "sentence_transformers":
                self.batch_size = 16   # для CPU 16–32
            else:
                raise ValueError(f"Unsupported provider: {provider}")
        else:
            self.batch_size = batch_size

        self._init_client(api_key)

    def _init_client(self, api_key: Optional[str]) -> None:
        """Инициализирует клиент в зависимости от провайдера."""
        if self.provider == "openai":
            if AsyncOpenAI is None:
                raise ImportError("openai library is not installed")
            self._client = AsyncOpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
            if not self.model:
                self.model = "text-embedding-3-small"

        elif self.provider == "sentence_transformers":
            if SentenceTransformer is None:
                raise ImportError("sentence-transformers library is not installed")
            if not self.model:
                self.model = "all-MiniLM-L6-v2"
            self._model = SentenceTransformer(self.model)

        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _is_e5(self) -> bool:
        """Проверяет, относится ли модель к семейству E5 по имени."""
        return "e5" in self.model.lower()

    # ------------------------------------------------------------------
    # Кеширование
    # ------------------------------------------------------------------

    def _cache_key(self, text: str) -> str:
        """Формирует ключ кеша: sha256 от 'model:text'.

        При смене модели ключ меняется — кеш инвалидируется автоматически.
        Размерность не включается: у одной модели она фиксирована, а до первого
        вызова _dimensions ещё неизвестна (None), что ломало бы попадание в кеш.
        """
        raw = f"{self.model}:{text}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _cache_get(self, text: str) -> Optional[list[float]]:
        """Ищет вектор в кеше (in-memory → disk)."""
        key = self._cache_key(text)

        # 1) In-memory
        if key in self._cache:
            self._cache_hits += 1
            return self._cache[key]

        # 2) Disk
        if self._disk_cache is not None:
            try:
                cached = self._disk_cache.get(key)
                if cached is not None:
                    self._cache_hits += 1
                    # Подогреваем in-memory кеш
                    self._cache_set_in_memory(key, cached)
                    return cached
            except Exception as exc:
                logger.debug("disk cache read error: %s", exc)

        self._cache_misses += 1
        return None

    def _cache_set(self, text: str, vector: list[float]) -> None:
        """Сохраняет вектор в оба кеша."""
        key = self._cache_key(text)
        self._cache_set_in_memory(key, vector)

        if self._disk_cache is not None:
            try:
                self._disk_cache[key] = vector
            except Exception as exc:
                logger.debug("disk cache write error: %s", exc)

    def _cache_set_in_memory(self, key: str, vector: list[float]) -> None:
        """Сохраняет в in-memory кеш с вытеснением старых записей."""
        if key in self._cache:
            return  # уже есть

        # Вытеснение: удаляем старейшие записи при превышении лимита
        while len(self._cache) >= self.max_cache_size:
            oldest_key = self._cache_order.pop(0)
            self._cache.pop(oldest_key, None)

        self._cache[key] = vector
        self._cache_order.append(key)

    # ------------------------------------------------------------------
    # Основные методы
    # ------------------------------------------------------------------

    async def _embed_batch_with_retry(self, batch: List[str]) -> List[List[float]]:
        """
        Отправляет один батч в провайдер с повторными попытками при сетевых ошибках.
        Возвращает список векторов в порядке текстов.
        """
        if self.provider == "openai":
            return await self._embed_batch_openai(batch)
        elif self.provider == "sentence_transformers":
            return await self._embed_batch_st(batch)
        else:
            raise RuntimeError(f"Unexpected provider: {self.provider}")

    async def _embed_batch_openai(self, batch: List[str]) -> List[List[float]]:
        """Отправляет батч в OpenAI API с retry."""
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = await self._client.embeddings.create(
                    model=self.model,
                    input=batch,
                )
                # Сохраняем размерность после первого успешного вызова
                if self._dimensions is None and resp.data:
                    self._dimensions = len(resp.data[0].embedding)
                # Данные приходят в произвольном порядке, сортируем по индексу
                data = sorted(resp.data, key=lambda x: x.index)
                return [item.embedding for item in data]
            except (APIConnectionError, RateLimitError, APIError) as e:
                if attempt == self.max_retries:
                    logger.error(f"Max retries exceeded for batch: {e}")
                    raise
                wait = min(self.backoff_max, self.backoff_base * (2 ** (attempt - 1)))
                wait += random.uniform(0, 0.1 * wait)
                logger.warning(
                    f"OpenAI error: {e}. Retrying in {wait:.2f}s (attempt {attempt}/{self.max_retries})"
                )
                await asyncio.sleep(wait)
            except Exception as e:
                logger.error(f"Unrecoverable error during OpenAI embedding: {e}")
                raise
        raise RuntimeError("Failed to embed batch after retries")

    async def _embed_batch_st(self, batch: List[str]) -> List[List[float]]:
        """Выполняет кодирование батча через sentence-transformers в отдельном потоке."""
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: self._model.encode(
                batch,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        )
        # Сохраняем размерность после первого вызова
        if self._dimensions is None and len(embeddings) > 0:
            self._dimensions = embeddings[0].shape[0]
        return embeddings.tolist()

    async def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        """
        Основной метод получения эмбеддингов.

        Разбивает входные тексты на батчи, проверяет кеш для каждого текста,
        и вызывает API только для отсутствующих в кеше. Возвращает векторы
        в том же порядке.

        Аргументы:
            texts (Sequence[str]): список текстов для векторизации.

        Возвращает:
            List[List[float]]: список векторов той же длины, что и texts.
        """
        if not texts:
            return []

        # Проверяем кеш для каждого текста
        result: List[Optional[list[float]]] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for i, text in enumerate(texts):
            cached = self._cache_get(text)
            if cached is not None:
                result[i] = cached
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        # Запрашиваем у провайдера только отсутствующие тексты
        if uncached_texts:
            new_vectors: list[list[float]] = []
            for i_start in range(0, len(uncached_texts), self.batch_size):
                batch = list(uncached_texts[i_start:i_start + self.batch_size])
                batch_vectors = await self._embed_batch_with_retry(batch)
                new_vectors.extend(batch_vectors)

            # Сохраняем в кеш и заполняем результат
            for j, idx in enumerate(uncached_indices):
                vec = new_vectors[j]
                result[idx] = vec
                self._cache_set(uncached_texts[j], vec)

        return result  # type: ignore[return-value]  # все None заменены

    async def embed_one(self, text: str) -> list[float]:
        """
        Получает вектор для одного текста.

        Аргументы:
            text (str): текст.

        Возвращает:
            List[float]: вектор.
        """
        result = await self.embed_texts([text])
        return result[0]

    async def embed_query(self, text: str) -> list[float]:
        """
        Метод для кодирования запроса (query) в моделях семейства E5.
        Добавляет префикс "query: " перед текстом.

        Доступен только для моделей E5.

        Аргументы:
            text (str): текст запроса.

        Возвращает:
            List[float]: нормализованный вектор.

        Исключения:
            NotImplementedError: если модель не принадлежит семейству E5.
        """
        if not self._is_e5():
            raise NotImplementedError(
                f"embed_query is only supported for E5 models, got {self.model}"
            )
        return await self.embed_one("query: " + text)

    async def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        """
        Метод для кодирования документов (passages) в моделях семейства E5.
        Добавляет префикс "passage: " к каждому тексту.

        Доступен только для моделей E5.

        Аргументы:
            texts (Sequence[str]): список текстов документов.

        Возвращает:
            List[List[float]]: список векторов в том же порядке.

        Исключения:
            NotImplementedError: если модель не принадлежит семейству E5.
        """
        if not self._is_e5():
            raise NotImplementedError(
                f"embed_documents is only supported for E5 models, got {self.model}"
            )
        prefixed = ["passage: " + t for t in texts]
        return await self.embed_texts(prefixed)

    def cache_stats(self) -> dict:
        """Возвращает статистику кеша: hits, misses, size."""
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "in_memory_size": len(self._cache),
            "disk_enabled": self._disk_cache is not None,
        }


# Фабричная функция для удобства создания экземпляра
def create_embeddings_service(
    provider: str = "openai",
    model: Optional[str] = None,
    batch_size: Optional[int] = None,
    api_key: Optional[str] = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    cache_dir: Optional[str] = None,
) -> EmbeddingsService:
    """
    Создаёт экземпляр EmbeddingsService с заданными параметрами.

    Аргументы:
        provider: 'openai' или 'sentence_transformers'
        model: имя модели
        batch_size: размер батча (если None, выбирается по умолчанию)
        api_key: ключ OpenAI (если не указан, берётся из окружения)
        max_retries: число повторных попыток
        cache_dir: путь к директории diskcache (None — только in-memory)

    Возвращает:
        EmbeddingsService
    """
    return EmbeddingsService(
        provider=provider,
        model=model,
        batch_size=batch_size,
        api_key=api_key,
        max_retries=max_retries,
        cache_dir=cache_dir,
    )


# Пример использования (для отладки)
async def _smoke_test() -> None:
    """Простой тест для проверки работы сервиса с кешированием."""
    service = create_embeddings_service(
        provider="openai",
        model="text-embedding-3-small",
    )

    texts = ["Привет, мир", "Второй текст для примера"]
    vectors = await service.embed_texts(texts)
    print(f"Получено {len(vectors)} векторов, размерность {len(vectors[0])}")

    # Повторный вызов — должны попасть в кеш
    vectors2 = await service.embed_texts(texts)
    print(f"Повторный вызов: {len(vectors2)} векторов")
    print(f"Статистика кеша: {service.cache_stats()}")

    # Проверка E5 (если модель E5)
    if service._is_e5():
        q_vec = await service.embed_query("Что такое эмбеддинги?")
        d_vecs = await service.embed_documents(["Эмбеддинги - это ...", "Векторные представления"])
        print(f"Query вектор: {q_vec[:5]}..., документов: {len(d_vecs)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(_smoke_test())
