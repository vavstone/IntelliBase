#!/usr/bin/env python3
"""
Скрипт проверки порогов качества.
Читает последний файл из eval/runs/, сверяет агрегаты с порогами из eval/thresholds.yaml.
При нарушении печатает ошибки и завершается с кодом 1.
Использование:
    python eval/check_thresholds.py
"""

import sys
import json
import yaml
from pathlib import Path
from typing import Dict, Any


def get_latest_run_file(runs_dir: Path) -> Path | None:
    """Возвращает путь к самому свежему файлу прогона (по имени или по дате модификации)."""
    files = list(runs_dir.glob("*.json"))
    if not files:
        return None
    # Сортируем по времени модификации (самый новый последний)
    files.sort(key=lambda p: p.stat().st_mtime)
    return files[-1]


def load_thresholds(path: Path) -> Dict[str, float]:
    """Загружает пороги из YAML."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data


def check_thresholds(run_data: Dict[str, Any], thresholds: Dict[str, float]) -> bool:
    """
    Проверяет агрегаты run_data на соответствие порогам.
    Возвращает True, если все пороги пройдены, иначе False.
    Печатает информацию о нарушениях.
    """
    aggregates = run_data.get("aggregates", {})
    passed = True
    for key, threshold in thresholds.items():
        value = aggregates.get(key)
        if value is None:
            print(f"⚠️  Агрегат '{key}' отсутствует в данных прогона")
            passed = False
            continue
        if value < threshold:
            print(f"❌ {key} = {value:.2f} < {threshold:.2f}  (порог не пройден)")
            passed = False
        else:
            print(f"✅ {key} = {value:.2f} >= {threshold:.2f}")
    return passed


def main():
    # Определяем пути
    base_dir = Path(__file__).parent
    runs_dir = base_dir / "runs"
    thresholds_path = base_dir / "thresholds.yaml"

    if not runs_dir.exists():
        print("❌ Папка eval/runs/ не существует.", file=sys.stderr)
        sys.exit(1)

    latest = get_latest_run_file(runs_dir)
    if latest is None:
        print("❌ Нет файлов прогонов в eval/runs/", file=sys.stderr)
        sys.exit(1)

    print(f"📄 Читаем последний прогон: {latest.name}")

    with open(latest, "r", encoding="utf-8") as f:
        run_data = json.load(f)

    if not thresholds_path.exists():
        print(f"⚠️  Файл порогов {thresholds_path} не найден. Используем значения по умолчанию.", file=sys.stderr)
        thresholds = {"correctness_avg": 4.0, "min_correctness": 2.0}
    else:
        thresholds = load_thresholds(thresholds_path)

    print("🔍 Проверка порогов:")
    passed = check_thresholds(run_data, thresholds)

    if passed:
        print("✅ Все пороги пройдены. Релиз разрешён.")
        sys.exit(0)
    else:
        print("❌ Некоторые пороги не пройдены. Релиз запрещён.")
        sys.exit(1)


if __name__ == "__main__":
    main()