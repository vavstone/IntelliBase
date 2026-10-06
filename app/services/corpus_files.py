"""Файлы корпуса, которые агент может отправить получателю.

Модель называет файл так, как его показывает RAG в `sources[].file_name` —
базовым именем («ПС Тарифы. Технические требования. Версия 2.4.docx»). Путь
собирает сервер: принимается только имя файла, каталоги и `..` отбрасываются.
Иначе модель (или текст внутри документа — классическая инъекция) могла бы
вытащить из контейнера произвольный файл вроде `/app/.env`.
"""

import logging
from pathlib import Path

log = logging.getLogger(__name__)

# Ограничение Telegram Bot API на отправку документа — 50 МБ.
MAX_DOCUMENT_BYTES = 50 * 1024 * 1024

# Что вообще имеет смысл отправлять как «документ».
SENDABLE_SUFFIXES = {".pdf", ".docx", ".doc", ".txt", ".md", ".xlsx", ".csv"}


def _stem(name: str) -> str:
    """Имя без известного расширения: «Версия 2.4.docx» → «версия 2.4».

    Не `Path.stem`: он режет по последней точке, а в наших именах точек много —
    «ПС Тарифы. Технические требования. Версия 2.4» превратилось бы в «…Версия 2».
    """
    lowered = name.casefold()
    for suffix in SENDABLE_SUFFIXES:
        if lowered.endswith(suffix):
            return lowered[: -len(suffix)]
    return lowered


def _sendable_documents(root: Path) -> list[Path]:
    """Файлы корпуса, которые вообще можно отправить (документ и по размеру)."""
    documents: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.casefold() not in SENDABLE_SUFFIXES:
            continue
        if path.stat().st_size > MAX_DOCUMENT_BYTES:
            log.warning(
                "Отправка файла: %s больше лимита Telegram (%d байт)",
                path.name,
                path.stat().st_size,
            )
            continue
        documents.append(path)
    return documents


def resolve_corpus_file(name: str, root: str | Path) -> Path | None:
    """Находит файл корпуса по имени. None — если файла нет или он не отправится.

    Принимается и имя с расширением («… Версия 1.9.docx»), и без него
    («… Версия 1.9») — человек обычно пишет второе. Неоднозначное имя (несколько
    файлов с одинаковым стволом) отклоняется: угадывать, какой из них имели в
    виду, нельзя — уйдёт не тот документ.

    Отдельно отсекается путь: модель называет файл, а путь собирает сервер.
    """
    candidates = Path(str(name or "").strip())
    if not candidates.name or candidates.name in {".", ".."}:
        return None
    target = candidates.name  # только имя: путь и `..` отбрасываем
    if target != str(name).strip():
        log.warning("Отправка файла: из %r берём только имя %r", name, target)

    root_path = Path(root)
    if not root_path.is_dir():
        return None
    documents = _sendable_documents(root_path)

    exact = [p for p in documents if p.name.casefold() == target.casefold()]
    if not exact:
        stem = _stem(target)
        exact = [p for p in documents if _stem(p.name) == stem]
        if len(exact) > 1:
            log.warning(
                "Отправка файла: имя %r неоднозначно — подходят %s",
                name,
                ", ".join(sorted(p.name for p in exact)),
            )
            return None
    return exact[0] if exact else None


def resolve_document_for_agent(settings):  # noqa: ANN001, ANN201
    """Функция-резолвер для графа: имя файла → Path (или None).

    Замыкание на настройки нужно, чтобы граф не знал про `RAG_DATA_DIR`:
    в тестах резолвер подменяется.
    """

    def _resolve(name: str) -> Path | None:
        return resolve_corpus_file(name, settings.rag_data_dir)

    return _resolve


def list_corpus_documents(root: str | Path, query: str = "", limit: int = 50) -> list[dict]:
    """Каталог документов корпуса: [{category, file_name, size_kb}].

    Нужен для вопросов «какие документы есть в базе»: векторный поиск отдаёт
    top-K чанков и полного списка не даёт — модель достраивала бы его догадками.
    Здесь список берётся из файловой системы, то есть детерминированно.
    """
    root_path = Path(root)
    if not root_path.is_dir():
        return []
    needle = (query or "").strip().casefold()
    found: list[dict] = []
    for path in sorted(root_path.rglob("*")):
        if not path.is_file() or path.suffix.casefold() not in SENDABLE_SUFFIXES:
            continue
        category = path.relative_to(root_path).parts[0]
        if len(path.relative_to(root_path).parts) == 1:
            category = "raznoe"  # файл в корне корпуса — как в индексации
        if needle and needle not in path.name.casefold() and needle not in category.casefold():
            continue
        found.append(
            {
                "category": category,
                "file_name": path.name,
                "size_kb": round(path.stat().st_size / 1024),
            }
        )
        if len(found) >= limit:
            break
    return found
