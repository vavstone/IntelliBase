# Чат-сервис (Persistent Chat)
Модуль app/chat предоставляет REST API для асинхронного диалога с LLM с сохранением истории сообщений. 
Поддерживается два бэкенда хранилища: JSON‑файлы (по умолчанию) и PostgreSQL.

## Диаграмма
![Архитектура чат-сервиса](diagrams/chat.drawio.svg)

## Компоненты:

**routes.py** – endpoints для создания чата, отправки сообщений, получения истории и удаления сообщений.

**ChatService** – бизнес‑логика: загрузка чата, сохранение пользовательского сообщения, построение контекста (sliding window), вызов LLM и сохранение ответа ассистента.

**ChatRepository** – абстракция хранилища (Protocol). Реализации:

* JsonChatRepository – сохраняет метаданные чата в chat.json, сообщения – в messages.jsonl (с поддержкой soft‑delete через маркеры).

* PostgresChatRepository – использует SQLAlchemy async‑сессии, таблицы chats и chat_messages с полем deleted_at для мягкого удаления.

**LLM клиенты** – инкапсулированы в AsyncOpenAI (три экземпляра для разных провайдеров). Выбор провайдера определяется полем provider у чата.


## Стратегия контекста
Для управления длиной контекста применяется скользящее окно (sliding window).

В запрос к LLM включаются:

системный промпт чата (если задан);

последние N сообщений из истории, где N задаётся переменной CHAT_CONTEXT_WINDOW (по умолчанию 10).

**Обоснование:**
В корпоративных поисковых сценариях диалоги могут быть длительными, а релевантность ответа зависит от недавнего обсуждения документов. Скользящее окно позволяет сохранить актуальный контекст, не перегружая модель и не превышая лимиты по токенам, при этом отбрасывая слишком старые сообщения, которые уже не влияют на текущий запрос.


## API Endpoints

**Базовый префикс: /chats**

**1. Создание / получение чата**

**POST /chats** – идемпотентно: если для пары (owner_external_id, interface) уже существует чат, возвращается его chat_id.

Тело запроса:

`{
  "owner_external_id": "user_123",
  "interface": "telegram",
  "provider": "openai",
  "model": "gpt-4o-mini",
  "system_prompt": "Ты – помощник по документации"
}`

Ответ:

`{
  "chat_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6"
}`

Пример cURL:

cmd
`curl -X POST "http://localhost:8000/chats" -H "Content-Type: application/json" -d 
"{\"owner_external_id\":\"user_123\",\"interface\":\"telegram\",\"provider\":\"openai\",
\"model\":\"gpt-4o-mini\",\"system_prompt\":\"Ты - помощник по документации\"}"`

**2. Получение метаданных чата**
**GET /chats/{chat_id}**

Ответ: объект Chat (id, owner_external_id, interface, provider, model, system_prompt, created_at).

Пример:

cmd
`curl http://localhost:8000/chats/3fa85f64-5717-4562-b3fc-2c963f66afa6`

**4. Отправка сообщения (SSE‑стриминг)**

**POST /chats/{chat_id}/messages** – multipart/form-data, поле content.

Ответ – Server‑Sent Events (SSE) с событиями:

`{"type":"token","delta":"..."}` – фрагмент ответа LLM.

`{"type":"message_saved","message_id":"..."}` – после сохранения ответа ассистента (клиент может использовать этот ID для feedback).

В конце отсылается `{"type":"done"}` (добавляется в роутере).

Пример cURL:

cmd
`curl -X POST http://localhost:8000/chats/3fa85f64-5717-4562-b3fc-2c963f66afa6/messages -F "content=Что такое векторная база данных?"`

**5. Получение истории сообщений**

**GET /chats/{chat_id}/messages?limit=50** (по умолчанию 50, максимум 500)

Возвращает список сообщений (хронологический порядок: от старых к новым).

Пример:

cmd
`curl "http://localhost:8000/chats/3fa85f64-5717-4562-b3fc-2c963f66afa6/messages?limit=20"`

**6. Очистка истории (soft delete)**
**DELETE /chats/{chat_id}/messages** – помечает все сообщения чата как удалённые (в JSONL добавляется маркер, в БД проставляется deleted_at). Сами записи не удаляются.

Пример:

cmd
`curl -X DELETE http://localhost:8000/chats/3fa85f64-5717-4562-b3fc-2c963f66afa6/messages`


## Хранилище
Переключение между реализациями репозитория выполняется через переменную конфигурации CHAT_REPOSITORY:

| Значение | Бэкенд | Описание |
|----------|--------|----------|
| `json`   | JSON‑файлы | Метаданные – `{chat_id}/chat.json`, сообщения – `{chat_id}/messages.jsonl`. Поддерживается мягкое удаление маркерами `{"type":"soft_delete",...}`. |
| `postgres` | PostgreSQL | Используется асинхронный драйвер `asyncpg`. Таблицы создаются через Alembic (миграции в `migrations/`). |

**Настройка:**

Использовать JSON

`CHAT_REPOSITORY=json
CHAT_STORAGE_DIR=./var/chats`

Использовать PostgreSQL

`CHAT_REPOSITORY=postgres
DATABASE_URL=postgresql+asyncpg://chat:pswd@localhost:5432/intellibase`

В compose.yaml по умолчанию установлен CHAT_REPOSITORY=postgres для демонстрации.

## Схема PostgreSQL
Таблицы (см. app/chat/repositories/pg_models.py):

**chats:**

* id (UUID, pk)
* owner_external_id (string)
* interface (string)
* provider (string)
* model (string)
* system_prompt (string, nullable)
* created_at (timestamptz)

**chat_messages:**

* id (UUID, pk)
* chat_id (UUID, fk → chats.id, ON DELETE CASCADE)
* role (string)
* content (string)
* tokens (int, nullable)
* created_at (timestamptz)
* deleted_at (timestamptz, nullable) – используется для soft delete.