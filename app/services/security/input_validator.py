# app/services/security/input_validator.py

import re
from typing import Final

from app.schemas.validation import ValidationResult

# ====== ПАТТЕРНЫ ИНЪЕКЦИЙ (естественный язык) ======
INJECTION_PATTERNS_EN: Final[list[re.Pattern]] = [
    re.compile(r"\bignore\s+(all\s+)?(previous|prior|above)\s+instructions?\b", re.IGNORECASE),
    re.compile(r"\bdisregard\s+(the\s+)?(system|previous|above)\b", re.IGNORECASE),
    re.compile(r"\byou\s+are\s+now\s+(a|an|the|dan|do anything now)\b", re.IGNORECASE),
    re.compile(r"\bforget\s+(everything|all|previous)\b", re.IGNORECASE),
    re.compile(r"\b(jailbroken|developer mode|godmode)\b", re.IGNORECASE),
    re.compile(r"\bnew\s+(instructions?|goal|objective|purpose)\b", re.IGNORECASE),
    re.compile(r"\boverride\s+(all|previous|system)\s+(instructions?|prompts?)\b", re.IGNORECASE),
    re.compile(r"\bchange\s+your\s+(role|purpose|behavior|personality)\b", re.IGNORECASE),
    re.compile(r"\b(act\s+as|pretend\s+to\s+be|you\s+are\s+now)\s+(an?\s+)?(AI|assistant|model|system)\b", re.IGNORECASE),
    re.compile(r"\b(remove|delete|erase)\s+(all\s+)?(restrictions|limitations|filters)\b", re.IGNORECASE),
    re.compile(r"\byou\s+(will|must)\s+follow\s+my\s+instructions\s+instead\b", re.IGNORECASE),
    re.compile(r"\bforget\s+the\s+above\s+and\s+do\s+this\b", re.IGNORECASE),
    re.compile(r"\bDAN\s+Mode\s+enabled\b", re.IGNORECASE),
    re.compile(r"\bsimulate\s+DAN\s+Mode\b", re.IGNORECASE),
    re.compile(r"\byou\s+are\s+going\s+to\s+act\s+as\b", re.IGNORECASE),
    re.compile(r"\bignore\s+any\s+previous\s+and\s+following\s+instructions\b", re.IGNORECASE),
    re.compile(r"\bjust\s+print\s+[\"'][^\"']+[\"']\b", re.IGNORECASE),  # "just print '...'"
]

INJECTION_PATTERNS_RU: Final[list[re.Pattern]] = [
    re.compile(r"игнорируй\s+(все\s+)?(предыдущие|вышестоящие|предыдущие|выше)\s+инструкции", re.IGNORECASE),
    re.compile(r"отбрось\s+(все\s+)?(предыдущие|системные|вышестоящие)\s+инструкции", re.IGNORECASE),
    re.compile(r"ты\s+теперь\s+(dan|делай\s+что\s+угодно|взломан|не\s+следуй\s+правилам)", re.IGNORECASE),
    re.compile(r"забудь\s+(всё|все\s+предыдущие|системный\s+промпт)", re.IGNORECASE),
    re.compile(r"(джейлбрейк|режим\s+разработчика|божественный\s+режим)", re.IGNORECASE),
    re.compile(r"новые\s+(инструкции|цели|задачи|указания)", re.IGNORECASE),
    re.compile(r"переопредели\s+(все|предыдущие|системные)\s+(инструкции|промпты)", re.IGNORECASE),
    re.compile(r"измени\s+свою\s+(роль|цель|поведение|личность)", re.IGNORECASE),
    re.compile(r"(действуй\s+как|притворись\s+|ты\s+теперь)\s+(искусственным\s+интеллектом|ассистентом|моделью|системой)", re.IGNORECASE),
    re.compile(r"(удали|отмени)\s+(все\s+)?(ограничения|фильтры|правила)", re.IGNORECASE),
    re.compile(r"ты\s+(будешь|должен)\s+следовать\s+моим\s+инструкциям\s+вместо", re.IGNORECASE),
    re.compile(r"забудь\s+вышесказанное\s+и\s+сделай\s+это", re.IGNORECASE),
]

# ====== ПАТТЕРНЫ ДЛЯ BASE64-МАРКЕРОВ ======
BASE64_PATTERNS: Final[list[re.Pattern]] = [
    # Английские фразы
    re.compile(r"\b(?:base64|b64)\s*(?:decode|encode|string|data|text|payload)\b", re.IGNORECASE),
    re.compile(r"\bdecode\s+(?:this|the)\s+base64\b", re.IGNORECASE),
    re.compile(r"\b(?:convert|transform)\s+(?:from|to)\s+base64\b", re.IGNORECASE),
    re.compile(r"\buse\s+base64\s+(?:to|for)\b", re.IGNORECASE),
    # Русские фразы
    re.compile(r"base64\s*(?:декодируй|декод|кодируй|строка|данные|текст|пейлоад)", re.IGNORECASE),
    re.compile(r"декодируй\s+(?:эту|это)\s+base64\s+строку", re.IGNORECASE),
    re.compile(r"(?:преобразуй|конвертируй)\s+(?:из|в)\s+base64", re.IGNORECASE),
    re.compile(r"используй\s+base64\s+(?:для|чтобы)", re.IGNORECASE),
]

# Объединяем все паттерны
ALL_PATTERNS: Final[list[re.Pattern]] = (
    INJECTION_PATTERNS_EN +
    INJECTION_PATTERNS_RU +
    BASE64_PATTERNS
)

# ====== ЭВРИСТИКИ ======
MAX_INPUT_CHARS: Final[int] = 4000
NON_PRINTABLE_RATIO_LIMIT: Final[float] = 0.10
# Минимальная длина подозрительной base64-подобной последовательности
MIN_BASE64_SEQUENCE_LEN: Final[int] = 20

def _looks_like_base64(text: str) -> bool:
    """
    Проверяет, содержит ли текст длинную непрерывную последовательность
    символов, допустимых в base64 (A-Z, a-z, 0-9, +, /, =).
    Это помогает выявить попытки обфускации через закодированные строки.
    """
    # Ищем последовательности без пробелов, состоящие только из base64-символов
    # Допустим, что это может быть частью большей строки, поэтому ищем любую такую подстроку
    # base64_chars = re.compile(r'[A-Za-z0-9+/=]')
    # Находим все непрерывные последовательности таких символов
    matches = re.findall(r'[A-Za-z0-9+/=]+', text)
    for seq in matches:
        # Если последовательность длиннее порога и не содержит пробелов (уже обеспечено)
        if len(seq) >= MIN_BASE64_SEQUENCE_LEN:
            # Дополнительно проверяем, что это не просто слово из букв
            # (можно добавить эвристику на соотношение букв/цифр)
            # Для простоты считаем любую длинную последовательность подозрительной
            return True
    return False


def validate_input(text: str) -> ValidationResult:
    if len(text) > MAX_INPUT_CHARS:
        return ValidationResult(False, "input too long", rule="length")

    non_printable = sum(1 for c in text if not c.isprintable() and c not in "\n\r\t")
    if non_printable / max(len(text), 1) > NON_PRINTABLE_RATIO_LIMIT:
        return ValidationResult(False, "high non-printable ratio", rule="encoding")

    # Проверка на явные инъекционные паттерны (включая base64-маркеры)
    for pat in ALL_PATTERNS:
        if pat.search(text):
            return ValidationResult(False, f"matched pattern {pat.pattern}", rule="injection")

    # Эвристика на длинные base64-подобные последовательности (без явных маркеров)
    if _looks_like_base64(text):
        return ValidationResult(False, "suspicious base64-like sequence", rule="encoding")

    return ValidationResult(True)