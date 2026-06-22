import re
from app.core.pii_patterns import PII_PATTERNS
from app.core.forbidden_phrases import FORBIDDEN_PHRASES
from app.core.suspicious_out_patterns import SUSPICIOUS_OUTPUT_PATTERNS


def filter_output(answer: str, system_prompt: str, canary: str) -> str:
    """
    Фильтрует ответ LLM:
      - проверяет утечку системного промпта (нормализация пробелов)
      - проверяет наличие канарейки
      - проверяет наличие запрещённых фраз (признаки успешной атаки)
      - маскирует персональные данные (email, телефон, карта, ИНН, паспорт)
    При обнаружении утечки или запрещённой фразы выбрасывает ValueError.
    """
    # 1. Проверка утечки системного промпта
    if system_prompt:
        # Нормализуем пробелы и приводим к нижнему регистру
        norm_answer = re.sub(r'\s+', ' ', answer).strip().lower()
        norm_system = re.sub(r'\s+', ' ', system_prompt).strip().lower()
        if norm_system and norm_system in norm_answer:
            raise ValueError("system_prompt leakage detected")

    # 2. Проверка канарейки
    if canary and canary in answer:
        raise ValueError("canary token detected in output")

    # 3. Проверка запрещённых фраз (признаки атаки)
    for pat in FORBIDDEN_PHRASES:
        if pat.search(answer):
            raise ValueError(f"forbidden phrase detected: {pat.pattern}")

    # 4. Проверка на наличие скрытых инструкций
    for pat in SUSPICIOUS_OUTPUT_PATTERNS:
        if pat.search(answer):
            raise ValueError(f"suspicious instruction pattern detected: {pat.pattern}")

    # 5. Script injection в ответе
    if re.search(r"<script|javascript:|onerror|s*=", answer, re.IGNORECASE):
        raise ValueError(f"script injection")

    # 6. Маскировка PII
    masked = answer
    for name, pattern in PII_PATTERNS.items():
        masked = pattern.sub(f"[{name}]", masked)
    return masked