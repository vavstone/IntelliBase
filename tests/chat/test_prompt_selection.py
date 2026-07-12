"""Tests for A/B traffic-split prompt selection (choose_by_split)."""

import pytest
from uuid import uuid4, UUID

from app.chat.domain import SystemPrompt
from app.chat.prompt_selection import choose_by_split


def _prompt(version: str, traffic_pct: int) -> SystemPrompt:
    """Helper: create a SystemPrompt with given version and traffic_pct."""
    return SystemPrompt(
        id=uuid4(),
        version=version,
        body=f"System prompt {version}",
        traffic_pct=traffic_pct,
    )


# ── basic cases ──────────────────────────────────────────────────────────

def test_empty_candidates_returns_none():
    """No candidates → None."""
    assert choose_by_split("user1", []) is None


def test_single_candidate_always_wins():
    """One candidate always returns regardless of owner."""
    p = _prompt("v1", 100)
    for uid in ["a", "b", "c", "12345", "99999"]:
        result = choose_by_split(uid, [p])
        assert result is not None
        assert result.version == "v1"


# ── sticky split ─────────────────────────────────────────────────────────

def test_same_owner_always_same_variant():
    """Deterministic: same owner_id → same bucket → same prompt."""
    a = _prompt("vA", 50)
    b = _prompt("vB", 50)
    candidates = [a, b]

    chosen = [choose_by_split("user42", candidates) for _ in range(100)]
    versions = {c.version for c in chosen}
    # Same owner always gets the same variant
    assert len(versions) == 1


def test_different_owners_may_get_different_variants():
    """With enough users, both variants get traffic."""
    a = _prompt("vA", 50)
    b = _prompt("vB", 50)
    candidates = [a, b]

    versions = set()
    for uid in [f"user{i}" for i in range(200)]:
        chosen = choose_by_split(uid, candidates)
        assert chosen is not None
        versions.add(chosen.version)

    # With 200 users and 50/50 split, both versions should appear
    assert "vA" in versions
    assert "vB" in versions


# ── traffic split accuracy ───────────────────────────────────────────────

def test_traffic_split_approximate():
    """Distribution roughly matches traffic_pct (within 10% tolerance for 1000 users)."""
    a = _prompt("vA", 30)
    b = _prompt("vB", 70)
    candidates = [a, b]

    counts = {"vA": 0, "vB": 0}
    n = 1000
    for i in range(n):
        chosen = choose_by_split(f"user_{i}", candidates)
        assert chosen is not None
        counts[chosen.version] += 1

    # 30/70 split, tolerance ±10%
    assert 0.20 <= counts["vA"] / n <= 0.40, f"vA got {counts['vA']}/{n}"
    assert 0.60 <= counts["vB"] / n <= 0.80, f"vB got {counts['vB']}/{n}"


def test_traffic_split_sum_less_than_100():
    """When sum(traffic_pct) < 100, gap falls back to first candidate."""
    a = _prompt("vA", 20)
    b = _prompt("vB", 20)
    candidates = [a, b]  # сумма 40, дыра 60

    counts = {"vA": 0, "vB": 0}
    n = 500
    for i in range(n):
        chosen = choose_by_split(f"user_{i}", candidates)
        counts[chosen.version] += 1

    # Fallback to first candidate for the gap (60% gap + 20% = 80% for vA)
    assert counts["vA"] > counts["vB"]


def test_traffic_split_all_to_one():
    """100% to one variant — all users get it."""
    a = _prompt("vA", 100)
    b = _prompt("vB", 0)  # не должен быть активным, но для теста
    candidates = [a, b]

    for i in range(100):
        chosen = choose_by_split(f"user_{i}", candidates)
        assert chosen.version == "vA"


def test_traffic_split_zero_pct_never_chosen():
    """0% variant never wins (unless it's the only candidate)."""
    a = _prompt("vA", 100)
    b = _prompt("vB", 0)
    candidates = [a, b]

    for i in range(200):
        chosen = choose_by_split(f"user_{i}", candidates)
        assert chosen.version == "vA"
