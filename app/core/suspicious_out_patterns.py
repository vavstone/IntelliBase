# ----- Паттерны для обнаружения скрытых инструкций в ответе -----
# Эти фразы указывают на то, что модель попыталась выполнить вредоносную инструкцию
import re
from typing import Final

SUSPICIOUS_OUTPUT_PATTERNS: Final[list[re.Pattern]] = [
    # Английские
    re.compile(r"\b(ignore|disregard|forget)\s+(the\s+)?(system|previous)\s+(instructions?|prompts?)\b", re.IGNORECASE),
    re.compile(r"\b(now|from now on)\s+you\s+are\s+(a|an|the)?\s*(dan|assistant|model)\b", re.IGNORECASE),
    re.compile(r"\b(jailbroken|developer mode|godmode)\b", re.IGNORECASE),
    re.compile(r"\b(override|bypass)\s+(restrictions|filters|limitations)\b", re.IGNORECASE),
    re.compile(r"\byou\s+must\s+follow\s+my\s+instructions\s+instead\b", re.IGNORECASE),
    # Русские
    re.compile(r"(игнорируй|отбрось|забудь)\s+(системные|предыдущие)\s+инструкции", re.IGNORECASE),
    re.compile(r"теперь\s+ты\s+(dan|делаешь\s+что\s+угодно|не\s+следуешь\s+правилам)", re.IGNORECASE),
    re.compile(r"(джейлбрейк|режим\s+разработчика)", re.IGNORECASE),
    re.compile(r"обойди\s+(ограничения|фильтры)", re.IGNORECASE),
]