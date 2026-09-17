"""Тихий режим HuggingFace.

`huggingface_hub` читает эти переменные в момент импорта, а тянет его за собой уже
`import langchain.*` / `transformers`. Поэтому ставить их внутри
`app/services/rag.py` поздно — модуль подключается после langchain.

Подключается в самой ранней точке входа (`app/__init__.py`), а скрипты, которые
импортируют langchain до `app.*` (например, `experiments/*`), импортируют этот
модуль явно первой строкой.

Что убирает из вывода:
- предупреждение «You are sending unauthenticated requests to the HF Hub»;
- прогресс-бар «Loading weights» при каждой загрузке embedding-модели.

Переменные ставятся через `setdefault`, поэтому их можно переопределить снаружи.
"""

import os

os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
