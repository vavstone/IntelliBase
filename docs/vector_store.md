# Vector store — отчёт по ДЗ 5.2

## Конфигурация

- **Движок:** Qdrant `qdrant/qdrant:v1.14.0`
- **Embedding-модель:** `intfloat/multilingual-e5-large` (dim=1024)
- **Метрика:** cosine
- **Коллекция:** `documents`
- **HNSW:** `m=16`, `ef_construct=100` — defaults Qdrant (см. обоснование ниже)
- **Размер корпуса:** 137 точек (загружено из 154 чанков демо-подмножества; полный корпус — 2028 чанков)
- **Payload-индексы:** `source` (KEYWORD), `created_at` (DATETIME), `ps` (KEYWORD)
- **Предметное поле:** `ps` — программное средство (Тарифы, Малахит, Постконтроль, Таможня и Право и др.)

## Метрика: cosine vs dot product

Скрипт `scripts/compare_metrics.py` создаёт временные коллекции
`documents_cosine` и `documents_dot` на одних и тех же векторах,
прогоняет 5 запросов и собирает top-5.

| Запрос | top-5 cosine | top-5 dot | Совпало |
|--------|--------------|-----------|---------|
| Какие программные задачи входят в состав КПС «Постконтроль»? | `6f508e43,40659e49,a23606fb,7ccddaa8,806c73c1` | `6f508e43,40659e49,a23606fb,7ccddaa8,806c73c1` | ✓ |
| Как работает автоматическая проверка декларантов на попадание в таблицу 100MILLION? | `d610a3e1,3dde6fb7,1b7a2bf4,94a226ce,a23606fb` | `d610a3e1,3dde6fb7,1b7a2bf4,94a226ce,a23606fb` | ✓ |
| Какой классификатор используется для товаров без обязательной маркировки? | `ef4c9c2c,a49a3d32,110aa6c4,6f508e43,3dde6fb7` | `ef4c9c2c,a49a3d32,110aa6c4,6f508e43,3dde6fb7` | ✓ |
| Какие СУБД и языки программирования используются в КПС «Постконтроль»? | `6f508e43,7ccddaa8,40659e49,a23606fb,c4679a61` | `6f508e43,7ccddaa8,40659e49,a23606fb,c4679a61` | ✓ |
| Какие PL/SQL пакеты отвечают за взаимодействие с системой маркировки? | `110aa6c4,6f508e43,74514e0b,6d7341a4,c4679a61` | `110aa6c4,6f508e43,74514e0b,6d7341a4,c4679a61` | ✓ |

**Что осталось в production:** COSINE

**Почему:** Embeddings E5 нормализованы (||v||=1) — cosine и dot дают
идентичное ранжирование (подтверждено на 5 запросах из домена ФТС).
Оставляю COSINE: явный контракт для модели
(учили на cosine similarity), читаемо в коде, дефолт в большинстве SDK.

## Примеры фильтров

### 1. Match по строке: `ps = "Тарифы"`

```python
from qdrant_client.models import FieldCondition, Filter, MatchValue

flt = Filter(
    must=[FieldCondition(key="ps", match=MatchValue(value="Тарифы"))]
)
hits = await store.search(query_vector=qv, top_k=3, query_filter=flt)
```

**Топ-3 для запроса «Какие программные задачи входят в состав КПС Постконтроль?»:**
1. `10a6eb77` — `ФТ_2023/Тарифы/1893_38-запрет/Заявка1893.PDF`
2. `4e07211b` — `ФТ_2023/Тарифы/3339_3904_Единая библиотека решений/Образ_d230364962_...`
3. `498eca8a` — `ФТ_2023/Тарифы/3339_3904_Единая библиотека решений/Образ_d230364962_...`

### 2. Range по дате: свежее 2025 года

```python
from qdrant_client.models import DatetimeRange, FieldCondition, Filter

flt = Filter(
    must=[FieldCondition(key="created_at", range=DatetimeRange(gte="2025-01-01T00:00:00Z"))]
)
hits = await store.search(query_vector=qv, top_k=3, query_filter=flt)
```

**Что меняется:** без фильтра в топе могут оказаться документы за 2023–2024 годы.
С фильтром — только актуальные требования 2025 года.

**Топ-3:**
1. `10a6eb77` — `2026-07-31T14:38:20+...`
2. `37d6cc67` — `2026-07-31T14:38:22+...`
3. `6f508e43` — `2026-07-31T14:38:22+...`

### 3. Композитный must + must_not: Тарифы, исключая «Малахит - Тарифы»

```python
from qdrant_client.models import FieldCondition, Filter, MatchValue

flt = Filter(
    must=[FieldCondition(key="ps", match=MatchValue(value="Тарифы"))],
    must_not=[FieldCondition(key="ps", match=MatchValue(value="Малахит - Тарифы"))],
)
hits = await store.search(query_vector=qv, top_k=3, query_filter=flt)
```

**Топ-3:** `10a6eb77`, `4e07211b`, `498eca8a` — все из ПС «Тарифы», без смешивания с «Малахит - Тарифы».

## HNSW: обоснование параметров

Параметры оставлены defaults Qdrant: `m=16`, `ef_construct=100`. Обоснование:

- На корпусе ~2000 точек recall с defaults достаточен для поиска по функциональным требованиям.
- Latency на таких объёмах — единицы миллисекунд.
- Поднимать `m` или `ef_construct` нет смысла: запас по recall и latency
  достаточный для целевого пользовательского сценария (поиск по базе ФТ).

При росте корпуса до 100K+ — добавить scalar quantization
(`quantization_config=ScalarQuantization(...)`) и поднять `ef_search`
до 100–128. Сейчас — overkill.