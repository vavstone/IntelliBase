# Документация IntelliBase — индекс

Карта документов проекта. «Живые» файлы обновляются на месте; отчёты и планы
хранятся с датой в имени (в `docs/tech_debt/`).

## Ядро

| Документ | Что описывает | Статус |
|----------|---------------|--------|
| [../ARCHITECTURE.md](../ARCHITECTURE.md) | Актуальная архитектура: модули, потоки данных, порты | живой |
| [architecture.md](architecture.md) | ADR: решения с контекстом и последствиями | исторический |
| [open_questions.md](open_questions.md) | Открытые вопросы и решения «на подумать» | живой |
| [chat.md](chat.md) | Чат-сервис: persistent chat, repository, sliding window | актуальный |
| [dev-setup.md](dev-setup.md) | Локальная разработка (Windows, Docker-инфраструктура) | актуальный |
| [runbook.md](runbook.md) | Операционные команды: запуск, индексация, smoke | живой |
| [env.md](env.md) | Справочник переменных окружения | живой |
| [testing.md](testing.md) | Карта тестов: что требует инфраструктуру | живой |

## RAG и поиск

| Документ | Что описывает | Статус |
|----------|---------------|--------|
| [rag.md](rag.md) | RAG на LlamaIndex: контур, score-guard, цитаты, сравнение с bare-metal | актуальный |
| [embeddings.md](embeddings.md) | Выбор embed-модели (E5, локально) | отчёт Б5.1 |
| [vector_store.md](vector_store.md) | Qdrant: конфигурация, cosine vs dot | отчёт Б5.2 |
| [chunking_experiment.md](chunking_experiment.md) | Сравнение стратегий чанкинга | отчёт Б5.4 |
| [rag_evaluation.md](rag_evaluation.md) | Оценка качества RAG: RAGAS, A/B, Phoenix-трейсинг | отчёт Б5.6 |

## Данные и техдолг

| Документ | Что описывает | Статус |
|----------|---------------|--------|
| [data_inventory.md](data_inventory.md) | Инвентаризация корпуса (70 док., разбивка по ПС) | актуальный |
| [tech_debt/](tech_debt/) | Планы доработок (дата в имени файла) | живой |

## Безопасность и инфраструктура

| Документ | Что описывает | Статус |
|----------|---------------|--------|
| [security/garak_baseline_2026-06-22.md](security/garak_baseline_2026-06-22.md) | Garak security-тесты (baseline) | отчёт |
| [security/garak_after_2026-06-22.md](security/garak_after_2026-06-22.md) | Garak security-тесты (после правок) | отчёт |
| [litellm/config.yaml](litellm/config.yaml) | Конфигурация LiteLLM-прокси | — |
| [diagrams/](diagrams/) | SVG-диаграммы (архитектура, чат) | — |
