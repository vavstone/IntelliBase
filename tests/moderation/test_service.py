"""Tests for ModerationService: check_input, check_output, regex, YAML, structlog."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.moderation.domain import ModerationResult


# ── helpers ──────────────────────────────────────────────────────────────

def _make_service(
    blocklist: list[dict] | None = None,
    use_openai: bool = False,
) -> "ModerationService":
    """Create ModerationService with mocked OpenAI client and no DB."""
    from app.moderation.service import ModerationService

    llm = MagicMock()
    llm.moderations.create = AsyncMock()
    return ModerationService(
        llm_client=llm,
        use_openai_moderation=use_openai,
        blocklist=blocklist,
        session_factory=None,
    )


def _blocklist() -> list[dict]:
    return [
        {
            "pattern": r"(?i)\b(взломать|хакни|взлом)\b",
            "reason": "Запрос инструкций по взлому",
            "category": "illegal",
        },
        {
            "pattern": r"(?i)\b(наркот|спайс)\b",
            "reason": "Упоминание наркотиков",
            "category": "drugs",
        },
    ]


# ── ModerationResult fields ──────────────────────────────────────────────

def test_moderation_result_defaults():
    """All fields have correct defaults."""
    r = ModerationResult(allowed=True)
    assert r.allowed is True
    assert r.categories == []
    assert r.reasons == []
    assert r.blocked_by == ""
    assert r.layer == ""
    assert r.scores == {}


def test_moderation_result_blocked():
    """Blocked result carries all rejection metadata."""
    r = ModerationResult(
        allowed=False,
        categories=["illegal"],
        reasons=["Запрос инструкций по взлому"],
        blocked_by=r"\bхакни\b",
        layer="regex",
    )
    assert r.allowed is False
    assert "illegal" in r.categories
    assert len(r.reasons) == 1
    assert r.blocked_by != ""


# ── check_input: regex layer ─────────────────────────────────────────────

@pytest.mark.anyio
async def test_check_input_regex_match():
    """Regex match blocks input and populates reasons/blocked_by."""
    svc = _make_service(blocklist=_blocklist())
    result = await svc.check_input("хакни сервер")

    assert result.allowed is False
    assert result.layer == "regex"
    assert "illegal" in result.categories
    assert result.reasons == ["Запрос инструкций по взлому"]
    assert "хакни" in result.blocked_by


@pytest.mark.anyio
async def test_check_input_regex_no_match():
    """Clean input passes regex layer."""
    svc = _make_service(blocklist=_blocklist())
    result = await svc.check_input("привет, как дела?")

    assert result.allowed is True
    assert result.layer == "passed"


@pytest.mark.anyio
async def test_check_input_empty_text():
    """Empty text always passes."""
    svc = _make_service(blocklist=_blocklist())
    result = await svc.check_input("")

    assert result.allowed is True


@pytest.mark.anyio
async def test_check_input_case_insensitive():
    """Regex matching is case-insensitive."""
    svc = _make_service(blocklist=_blocklist())
    # "Хакни" with capital Х, "ВзЛоМ" mixed case
    for text in ["Хакни базу", "ВзЛоМ сайта", "хАкНи пароль"]:
        result = await svc.check_input(text)
        assert result.allowed is False, f"should block: {text}"


# ── check_output ─────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_check_output_blocks_same_as_input():
    """check_output uses same logic as check_input."""
    svc = _make_service(blocklist=_blocklist())
    result = await svc.check_output("здесь спайс продают")

    assert result.allowed is False
    assert "drugs" in result.categories


@pytest.mark.anyio
async def test_check_output_passes_clean():
    """Clean output passes."""
    svc = _make_service(blocklist=_blocklist())
    result = await svc.check_output("обычный ответ ассистента")

    assert result.allowed is True


@pytest.mark.anyio
async def test_check_output_logs_direction():
    """check_output logs with direction='output'."""
    svc = _make_service(blocklist=_blocklist())
    with patch("app.moderation.service.logger") as mock_log:
        await svc.check_output("про спайс и наркоту")
        # Verify at least one log call was made (the block log)
        assert mock_log.info.called


# ── OpenAI Moderation API (fail-open) ────────────────────────────────────

@pytest.mark.anyio
async def test_openai_moderation_flagged():
    """OpenAI flags content → blocked."""
    svc = _make_service(blocklist=[], use_openai=True)
    svc.llm.moderations.create = AsyncMock()
    # Simulate flagged response
    mock_resp = MagicMock()
    mock_resp.results = [MagicMock()]
    mock_resp.results[0].flagged = True
    mock_resp.results[0].categories.model_dump.return_value = {
        "harassment": True, "violence": False
    }
    mock_resp.results[0].category_scores.model_dump.return_value = {
        "harassment": 0.95, "violence": 0.01
    }
    svc.llm.moderations.create.return_value = mock_resp

    result = await svc.check_input("some bad content")

    assert result.allowed is False
    assert result.layer == "openai"
    assert "harassment" in result.categories
    assert result.blocked_by == "openai_moderation_api"
    assert len(result.reasons) > 0


@pytest.mark.anyio
async def test_openai_moderation_not_flagged():
    """OpenAI clean → passes."""
    svc = _make_service(blocklist=[], use_openai=True)
    svc.llm.moderations.create = AsyncMock()
    mock_resp = MagicMock()
    mock_resp.results = [MagicMock()]
    mock_resp.results[0].flagged = False
    svc.llm.moderations.create.return_value = mock_resp

    result = await svc.check_input("hello world")

    assert result.allowed is True


@pytest.mark.anyio
async def test_openai_moderation_fail_open():
    """API error → fail-open (allow)."""
    svc = _make_service(blocklist=[], use_openai=True)
    svc.llm.moderations.create = AsyncMock(side_effect=Exception("timeout"))

    result = await svc.check_input("some text")

    assert result.allowed is True
    assert result.layer == "passed"


# ── YAML loading ─────────────────────────────────────────────────────────

def test_load_blocklist_from_yaml():
    """Loads patterns from moderation_keywords.yaml."""
    from app.moderation.service import _load_blocklist

    patterns = _load_blocklist()
    assert isinstance(patterns, list)
    assert len(patterns) > 0
    for p in patterns:
        assert "pattern" in p
        assert "reason" in p
        assert "category" in p


def test_load_blocklist_fallback_on_missing(tmp_path):
    """Falls back to DEFAULT_BLOCKLIST when YAML missing."""
    from app.moderation.service import _load_blocklist

    patterns = _load_blocklist(yaml_path=tmp_path / "nonexistent.yaml")
    assert isinstance(patterns, list)
    assert len(patterns) > 0


# ── _hash_owner ──────────────────────────────────────────────────────────

def test_hash_owner_consistent():
    """Same input → same hash."""
    from app.moderation.service import _hash_owner

    h1 = _hash_owner("user123")
    h2 = _hash_owner("user123")
    assert h1 == h2
    assert h1.startswith("sha256:")
    assert len(h1) == len("sha256:") + 16


def test_hash_owner_none():
    """None owner → 'anon'."""
    from app.moderation.service import _hash_owner

    assert _hash_owner(None) == "anon"
    assert _hash_owner("") == "anon"


# ── ModerationService.check_output via ChatService ────────────────────────

@pytest.mark.anyio
async def test_chat_service_check_output_delegates():
    """ChatService.check_output delegates to ModerationService."""
    from app.chat.service import ChatService
    from app.chat.repository import ChatRepository

    repo = MagicMock(spec=ChatRepository)
    svc = _make_service(blocklist=_blocklist())
    chat_svc = ChatService(
        repository=repo,
        llm_ollama=MagicMock(),
        llm_openai=MagicMock(),
        llm_openrouter=MagicMock(),
        moderation=svc,
    )

    # Blocked output
    result = await chat_svc.check_output("хакни всё", owner_external_id="u1")
    assert result.allowed is False

    # Clean output
    result = await chat_svc.check_output("обычный ответ", owner_external_id="u1")
    assert result.allowed is True


@pytest.mark.anyio
async def test_chat_service_check_output_none_moderation():
    """When moderation is None, everything passes."""
    from app.chat.service import ChatService
    from app.chat.repository import ChatRepository

    repo = MagicMock(spec=ChatRepository)
    chat_svc = ChatService(
        repository=repo,
        llm_ollama=MagicMock(),
        llm_openai=MagicMock(),
        llm_openrouter=MagicMock(),
        moderation=None,
    )
    result = await chat_svc.check_output("anything")
    assert result.allowed is True
    assert result.layer == "passed"
