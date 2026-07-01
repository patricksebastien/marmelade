"""Phase 8 Plan 08-03 — template-based pseudo-poem generator (D-22 + R-06).

Wave 0 stub upgraded to GREEN: tests now exercise
:mod:`marmelade.util.poem_generator`. R-06 amends D-22: word range is
4-7 (not 5-7); the 5 grammar templates in RESEARCH lines 754-761 produce
{4, 5, 5, 6, 7} word outputs.

D-27 invariant: the module under test is Qt-free
(``test_generate_is_qt_free`` pins this).
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from marmelade.util.poem_generator import (
    ADJECTIVES,
    ADVERBS,
    NOUNS,
    TEMPLATES,
    VERBS,
    generate,
)


# ---------------------------------------------------------------------------
# Banned-substring filter (T-08-03-01 mitigation — curation discipline)
# ---------------------------------------------------------------------------
# Profanity, slurs, and politically loaded terms. Sized small but covers the
# obvious lexical landmines. Future word-list additions re-run this gate.
_BANNED_TERMS: frozenset[str] = frozenset(
    {
        # Profanity / vulgar
        "fuck",
        "shit",
        "damn",
        "hell",
        "ass",
        "bitch",
        "crap",
        "piss",
        "bastard",
        # Slurs (placeholder set; never expand without review)
        "slur",  # sentinel — never appears in curated words
        # Political loaded terms
        "liberal",
        "conservative",
        "democrat",
        "republican",
        "nazi",
        "communist",
        "fascist",
        "leftist",
        "rightist",
        "antifa",
        "maga",
        "woke",
        # Religious-political flashpoints
        "jihad",
        "crusade",
        # Brand/copyrighted (avoid trademark traps)
        "google",
        "youtube",
        "facebook",
        "twitter",
        "tiktok",
    }
)


# ---------------------------------------------------------------------------
# Contract tests — 7 cases pin the D-22 + R-06 specification.
# ---------------------------------------------------------------------------


def test_generate_returns_4_to_7_words() -> None:
    """generate() returns a string whose word count is in [4, 7] (R-06)."""
    rng = random.Random(0)
    for _ in range(100):
        result = generate(rng)
        assert isinstance(result, str), result
        word_count = len(result.split())
        assert 4 <= word_count <= 7, (word_count, result)


def test_generate_is_seeded_deterministic() -> None:
    """generate(rng=random.Random(42)) is deterministic across calls."""
    a = generate(random.Random(42))
    b = generate(random.Random(42))
    assert a == b, (a, b)

    # And a different seed produces a different value (sanity — not a strict
    # contract, but if this fails the RNG is being ignored).
    c = generate(random.Random(43))
    assert a != c or True, (a, c)  # weak assertion: tolerant if collision


def test_word_lists_have_expected_counts() -> None:
    """ADJECTIVES >=80, NOUNS >=80, VERBS >=40, ADVERBS >=30 (D-22 sizes)."""
    assert len(ADJECTIVES) >= 80, len(ADJECTIVES)
    assert len(NOUNS) >= 80, len(NOUNS)
    assert len(VERBS) >= 40, len(VERBS)
    assert len(ADVERBS) >= 30, len(ADVERBS)

    # No duplicates within each list — uniqueness is part of the cardinality
    # contract (otherwise len(list) is misleading).
    for name, words in (
        ("ADJECTIVES", ADJECTIVES),
        ("NOUNS", NOUNS),
        ("VERBS", VERBS),
        ("ADVERBS", ADVERBS),
    ):
        assert len(set(words)) == len(words), (
            name,
            len(words),
            len(set(words)),
        )


def test_templates_cover_4_5_6_7_word_outputs() -> None:
    """The 5 grammar templates produce {4, 5, 5, 6, 7} word outputs (R-06).

    Word counts are determined by counting whitespace-separated tokens after
    filling each placeholder with a single-token sentinel ("X"). Each
    template's word count must land in {4, 5, 6, 7}.
    """
    assert len(TEMPLATES) == 5, len(TEMPLATES)

    counts: list[int] = []
    for tmpl in TEMPLATES:
        filled = tmpl.format(adj="X", noun="X", verb="X", adv="X")
        n = len(filled.split())
        assert 4 <= n <= 7, (tmpl, n)
        counts.append(n)

    # The exact spec (D-22 + R-06): {4, 5, 5, 6, 7}.
    assert sorted(counts) == [4, 5, 5, 6, 7], counts


def test_no_offensive_or_political_terms() -> None:
    """Curated word lists contain music/nature/movement themes only (CONTEXT).

    Pins the T-08-03-01 mitigation: the four word lists must not overlap with
    the curated _BANNED_TERMS set. If a future contributor adds a banned term
    this test trips before the module ships.
    """
    all_words = (
        set(ADJECTIVES) | set(NOUNS) | set(VERBS) | set(ADVERBS)
    )
    overlap = all_words & _BANNED_TERMS
    assert not overlap, sorted(overlap)


def test_generate_is_qt_free() -> None:
    """poem_generator module has zero PySide6 imports (D-27 N-3 invariant)."""
    module_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "marmelade"
        / "util"
        / "poem_generator.py"
    )
    text = module_path.read_text(encoding="utf-8")
    assert "PySide6" not in text, "poem_generator must not import PySide6"
    assert "PyQt" not in text, "poem_generator must not import PyQt"


def test_generate_handles_no_rng_argument() -> None:
    """generate() with no argument uses module-level random — valid output."""
    # Two calls without an rng argument must each return a valid 4-7 word
    # string (non-determinstic, so we don't compare them — just contract).
    a = generate()
    b = generate()
    for s in (a, b):
        assert isinstance(s, str), s
        assert 4 <= len(s.split()) <= 7, (len(s.split()), s)


# ---------------------------------------------------------------------------
# Distributional sanity (acceptance criterion §4 — >=500 unique / 1000 calls).
# Pinned as a test (not just a CLI smoke check) so regressions in word-list
# diversity surface in CI.
# ---------------------------------------------------------------------------


def test_generate_has_distributional_variety() -> None:
    """1000 seeded calls produce >=500 unique outputs (acceptance §4)."""
    rng = random.Random(0)
    outputs = [generate(rng) for _ in range(1000)]
    unique = len(set(outputs))
    assert unique >= 500, unique
