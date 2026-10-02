#!/usr/bin/env python3
"""Гейт качества RAG: сверка последнего прогона RAGAS с порогами.

Читает самый свежий CSV из `tests/eval/results/` (их пишет `scripts/run_eval.py`),
усредняет метрики RAGAS и сравнивает с порогами из `eval/thresholds.yaml`.
Каждое нарушение печатается отдельной строкой; код возврата 1 — «эти числа
показывать нельзя», 0 — можно.

Раньше скрипт читал G-Eval-прогоны из `eval/runs/` и сверял `correctness_avg`
(учебное задание 3.7). С переходом на RAGAS-контур (Б5.6) источником стали
`tests/eval/results/`; G-Eval-история осталась в `eval/runs/` как артефакт.

Запуск:
    uv run python eval/check_thresholds.py
    uv run python eval/check_thresholds.py --label demo   # конкретный прогон
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

# Русский вывод на Windows-консоли (cp1252/cp866) без этого падает
# UnicodeEncodeError ещё до первой строки отчёта.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

# Разбор RAGAS-CSV общий с `scripts/metrics.py` (цель `make metrics`).
from metrics import RAGAS_COLUMNS, as_float, latest_results_file, load_results  # noqa: E402

DEFAULT_THRESHOLDS = ROOT / "eval" / "thresholds.yaml"


def load_thresholds(path: Path) -> dict[str, float]:
    """Пороги из YAML: имя метрики → минимальное среднее."""
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return {str(key): float(value) for key, value in data.items()}


def averages(rows: list[dict[str, str]]) -> dict[str, tuple[float, int]]:
    """Средняя и число непустых значений по каждой метрике.

    Пустые ячейки (отказ score-guard, таймаут строки) в среднее не входят —
    иначе метрика молча занижалась бы. Число значений печатается рядом:
    видно, на скольких вопросах средняя посчитана.
    """
    out: dict[str, tuple[float, int]] = {}
    for column in RAGAS_COLUMNS:
        values = [v for row in rows if (v := as_float(row.get(column))) is not None]
        if values:
            out[column] = (statistics.fmean(values), len(values))
    return out


def check(
    summary: dict[str, tuple[float, int]], thresholds: dict[str, float]
) -> bool:
    """Печатает вердикт по каждой метрике. False — хотя бы один порог нарушен."""
    passed = True
    for metric, threshold in thresholds.items():
        if metric == "min_rows":
            continue
        found = summary.get(metric)
        if found is None:
            print(f"[НЕТ ДАННЫХ] {metric}: в прогоне нет ни одного значения "
                  f"(порог {threshold:.2f})")
            passed = False
            continue
        value, count = found
        if value < threshold:
            print(f"[ПРОВАЛ]   {metric} = {value:.3f} < {threshold:.2f}  (n={count})")
            passed = False
        else:
            print(f"[OK]       {metric} = {value:.3f} >= {threshold:.2f}  (n={count})")
    return passed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Пороги качества RAG (RAGAS)")
    parser.add_argument("--label", default=None, help="подстрока имени прогона, напр. demo")
    parser.add_argument(
        "--results-dir",
        default=None,
        help="каталог с CSV (по умолчанию tests/eval/results)",
    )
    parser.add_argument(
        "--thresholds",
        default=str(DEFAULT_THRESHOLDS),
        help="файл порогов (по умолчанию eval/thresholds.yaml)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.results_dir:
        # Позволяет проверить прогон из другого каталога, не трогая общий.
        import metrics

        metrics.RESULTS_DIR = Path(args.results_dir).resolve()

    path = latest_results_file(args.label)
    if path is None:
        print(
            f"Нет CSV с результатами RAGAS"
            f"{f' по метке «{args.label}»' if args.label else ''}. "
            "Прогон: `make eval`.",
            file=sys.stderr,
        )
        return 1

    rows = load_results(path)
    if not rows:
        print(f"Файл {path.name} пуст — прогон не дал строк.", file=sys.stderr)
        return 1

    thresholds_path = Path(args.thresholds)
    if thresholds_path.exists():
        thresholds = load_thresholds(thresholds_path)
    else:
        print(
            f"Файл порогов {thresholds_path} не найден, беру значения по умолчанию.",
            file=sys.stderr,
        )
        thresholds = {
            "faithfulness": 0.75,
            "answer_relevancy": 0.70,
            "context_recall": 0.70,
        }

    min_rows = int(thresholds.pop("min_rows", 0))
    print(f"Прогон: {path.stem}  (вопросов {len(rows)})")
    print(f"Пороги: {thresholds_path}")
    print("Проверка порогов:")

    passed = check(averages(rows), thresholds)

    if min_rows and len(rows) < min_rows:
        print(f"[ПРОВАЛ]   вопросов в прогоне {len(rows)} < min_rows {min_rows}")
        passed = False

    if passed:
        print("Все пороги пройдены — числа прогона можно показывать.")
        return 0
    print("Пороги не пройдены — числа прогона показывать нельзя.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
