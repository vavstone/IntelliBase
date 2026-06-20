from app.observability.pii import redact_pii, prompt_hash


def test_redact_pii_email():
    text = "My email is user@example.com and also user2@domain.co.uk"
    result = redact_pii(text)
    assert "user@example.com" not in result
    assert "[EMAIL]" in result
    assert "user2@domain.co.uk" not in result
    assert result.count("[EMAIL]") == 2


def test_redact_pii_phone_russian():
    text = "Call +7 912 345-67-89 or 8(912)3456789"
    result = redact_pii(text)
    assert "+7 912 345-67-89" not in result
    assert "8(912)3456789" not in result
    assert "[PHONE_RU]" in result
    # Проверяем, что оба номера заменены (их два)
    assert result.count("[PHONE_RU]") == 2


def test_redact_pii_card():
    text = "Card: 1234-5678-9012-3456"
    result = redact_pii(text)
    assert "1234-5678-9012-3456" not in result
    assert "[CARD]" in result


def test_redact_pii_mixed():
    text = "Email: a@b.com, phone: +7 999 123-45-67"
    result = redact_pii(text)
    assert "[EMAIL]" in result
    assert "[PHONE_RU]" in result


def test_prompt_hash():
    text = "Hello world"
    h = prompt_hash(text)
    assert h.startswith("sha256:")
    assert len(h) == len("sha256:") + 16  # 16 hex символов
    # Одинаковые тексты дают одинаковый хэш
    assert prompt_hash("Hello world") == h
    assert prompt_hash("Hello World") != h