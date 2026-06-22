import hashlib

from app.core.pii_patterns import PII_PATTERNS


def redact_pii(text: str) -> str:
    for name, pattern in PII_PATTERNS.items():
        text = pattern.sub(f"[{name}]", text)
    return text

def prompt_hash(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()[:16]