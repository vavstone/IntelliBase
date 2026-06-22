import re

PII_PATTERNS = {
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"),
    "PHONE_RU": re.compile(r"(?<!\d)(?:\+7|8)\s*\(?\d{3}\)?\s*[-\s.]?\s*\d{3}\s*[-\s.]?\s*\d{2}\s*[-\s.]?\s*\d{2}(?!\d)"),
    "CARD": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
    "INN": re.compile(r"\b(?:\d{10}|\d{12})\b"),
    "PASSPORT": re.compile(r"\b\d{2}\s*\d{2}\s*\d{6}\b"),
}
