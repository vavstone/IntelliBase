"""Метрики IntelliBase для демонстрации: p95 задержек, cache hit rate, RAGAS.

Цифры собираются из двух источников:

* admin-API `/chats/admin/stats` — задержки (avg/p95) по `request_metrics`, доля
  отказов RAG и счётчики кэша LLM из Redis (`ADMIN_TOKEN` и `APP_URL` берутся из
  окружения, затем из `.env`, затем из дефолтов);
* последний файл `tests/eval/results/*.csv` — метрики RAGAS (faithfulness,
  answer_relevancy, context_precision, context_recall) и доля ответов с цитатами.

Запуск:
    make metrics                                # окно 24 ч
    uv run python scripts/metrics.py --window-hours 168

Код возврата: 0 — метрики собраны, 1 — admin-API недоступен (нет ответа/токена).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import urllib.error
import urllib.request
from pathlib import Path

# load_env — общий с smoke.py: один и тот же парсер `.env` без зависимостей.
from smoke import load_env

# Русский вывод на Windows-консоли (cp1251) без этого падает UnicodeEncodeError.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "tests" / "eval" / "results"

# Колонки RAGAS-отчёта, которые усредняем (остальные — служебные).
RAGAS_COLUMNS = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "has_citation",
)
TIMEOUT = 30.0


def fetch_stats(app_url: str, admin_token: str, window_hours: int) -> dict | None:
    """GET /chats/admin/stats. None — API недоступен или токен не принят."""
    url = f"{app_url.rstrip('/')}/chats/admin/stats?window_hours={window_hours}"
    req = urllib.request.Request(url, headers={"X-Admin-Token": admin_token})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:120]
        print(f"admin-API вернул {exc.code}: {detail}")
    except Exception as exc:  # noqa: BLE001 — сеть/таймаут/DNS, детали в тексте
        print(f"admin-API недоступен: {type(exc).__name__}: {exc}")
    return None


def _as_float(raw: str | None) -> float | None:
    """Число из ячейки CSV. None — пусто, «nan» или не число (ошибка прогона).

    `has_citation` пишется как true/false, поэтому булевы значения тоже
    приводим к 1.0/0.0 — иначе метрика молча выпадала бы из сводки.
    """
    text = (raw or "").strip().lower()
    if not text or text in {"nan", "none"}:
        return None
    if text in {"true", "yes"}:
        return 1.0
    if text in {"false", "no"}:
        return 0.0
    try:
        return float(text)
    except ValueError:
        return None


def latest_ragas() -> tuple[str, dict[str, float], int] | None:
    """Средние по последнему RAGAS-прогону: (label, метрики, число вопросов).

    None — файлов с результатами нет. Пустые значения (ошибки прогона)
    пропускаются: метрика считается по тем вопросам, где она посчиталась.
    """
    files = sorted(RESULTS_DIR.glob("*.csv"))
    if not files:
        return None
    path = files[-1]
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if not rows:
        return None

    summary: dict[str, float] = {}
    for column in RAGAS_COLUMNS:
        values = []
        for row in rows:
            value = _as_float(row.get(column))
            if value is not None:
                values.append(value)
        if values:
            summary[column] = statistics.fmean(values)
    return path.stem, summary, len(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Метрики IntelliBase для демо")
    parser.add_argument(
        "--window-hours", type=int, default=24, help="окно агрегатов (по умолчанию 24 ч)"
    )
    parser.add_argument(
        "--app-url",
        default=os.environ.get("SMOKE_APP_URL", "http://localhost:8000"),
        help="базовый URL приложения",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_env(ROOT / ".env")

    admin_token = os.environ.get("ADMIN_TOKEN", "")
    app_url = args.app_url or os.environ.get("SMOKE_APP_URL", "http://localhost:8000")

    print(f"=== IntelliBase — метрики (окно {args.window_hours} ч) ===")
    stats = fetch_stats(app_url, admin_token, args.window_hours)
    if stats is None:
        print("Подсказка: стек поднят? `make up`; ADMIN_TOKEN задан в .env?")
        return 1

    hits = stats.get("cache_hits", 0)
    misses = stats.get("cache_misses", 0)
    rows = [
        ("запросов", f"{stats.get('total_requests', 0)}"),
        ("сообщений", f"{stats.get('total_messages', 0)}"),
        ("пользователей", f"{stats.get('active_users', 0)}"),
        ("средняя задержка", f"{stats.get('avg_latency_ms', 0):.1f} мс"),
        ("p95 задержки", f"{stats.get('p95_latency_ms', 0):.1f} мс"),
        (
            "cache hit rate*",
            f"{stats.get('cache_hit_rate', 0) * 100:.1f} %"
            f"  (hits {hits} / misses {misses})",
        ),
        ("отказов RAG", f"{stats.get('refusal_rate', 0) * 100:.1f} %"),
        ("отрицательных оценок", f"{stats.get('negative_feedback_rate', 0) * 100:.1f} %"),
        ("блокировок модерации", f"{stats.get('moderation_block_rate', 0) * 100:.1f} %"),
    ]
    for name, value in rows:
        print(f"  {name:<24} {value}")
    print("  * кэш — счётчики за всё время работы; задержки — без /health и /ready")

    ragas = latest_ragas()
    if ragas is None:
        print("RAGAS: файлов с результатами нет (см. `make eval`)")
    else:
        label, summary, questions = ragas
        print(f"RAGAS ({label}, вопросов {questions}):")
        for column in RAGAS_COLUMNS:
            if column in summary:
                print(f"  {column:<24} {summary[column]:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
