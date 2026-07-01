"""Template-based pseudo-poem generator (Phase 8 D-22 + R-06; YT-06).

Generates a 4-7 word evocative phrase suitable as a YouTube video-title
default. Implemented as curated word lists + 5 grammar templates per
RESEARCH lines 739-773 (the canonical skeleton) with the word range
extended to 4-7 per R-06 (supersedes D-22's original 5-7).

Pure-Python, Qt-free per D-27 (N-3 invariant). Zero install cost,
instant generation. NO LLM in v1 — local-LLM options
(Qwen2.5-0.5B / Phi-3-mini / llama-cpp-python) are deferred per the
Phase 8 Deferred Ideas register.

Per D-23 ("title fields always user-editable") the caller treats
``generate()`` output as a *default* for an editable QLineEdit — the
user reviews + edits before clicking Upload. The generator is therefore
a starting point, not a forced value.

Curation discipline (T-08-03-01 mitigation): the four word lists carry
music + nature + movement themes per RESEARCH §"Pseudo-poem word list
themes" Option A. Profanity, slurs, political loaded terms, and brand
names are excluded; the curation is pinned by
``tests/util/test_poem_generator.py::test_no_offensive_or_political_terms``.

Public API
----------
- :data:`ADJECTIVES`, :data:`NOUNS`, :data:`VERBS`, :data:`ADVERBS` —
  curated word tuples. Cardinality floors: 80 / 80 / 40 / 30.
- :data:`TEMPLATES` — 5 grammar templates producing {4, 5, 5, 6, 7}
  words.
- :func:`generate` — return one phrase.

Threat register: see Phase 8 Plan 08-03 frontmatter (T-08-03-01 /
T-08-03-02 / T-08-03-03). Pure data payload — no shell, SQL, or HTML
injection surface. Plan 08-04's UploadDialog wraps the value in a
``QLineEdit`` whose contents feed YouTube's ``snippet.title`` field
(server-side length + character validation applies).
"""

from __future__ import annotations

import random
from typing import Sequence

# ---------------------------------------------------------------------------
# Word lists — curated per RESEARCH §"Pseudo-poem word list themes" Option A
# (Music + Nature blend). Cardinality floors (D-22): >=80 / >=80 / >=40 / >=30.
# Lowercase ASCII (hyphens allowed for compound words like "salt-bleached").
# No duplicates within a list — pinned by test_word_lists_have_expected_counts.
# No profanity / slurs / political terms / brand names — pinned by
# test_no_offensive_or_political_terms (T-08-03-01).
# ---------------------------------------------------------------------------

ADJECTIVES: Sequence[str] = (
    # Atmosphere / light
    "drifting", "hushed", "golden", "restless", "distant",
    "quiet", "lonely", "vivid", "fading", "gentle",
    "wild", "slow", "bright", "deep", "soft",
    "sharp", "hollow", "warm", "faint", "steady",
    "broken", "tender", "wide", "lost", "fragile",
    "ancient", "sudden", "calm", "weary", "woven",
    "sleeping", "breathing", "simmering", "glowing", "returning",
    "rising", "falling", "open", "narrow", "brimming",
    "secret", "plain", "electric", "low", "high",
    "dim", "dappled", "slanted", "threadbare", "mossy",
    # Weather + season
    "tidal", "salt-bleached", "sun-warmed", "river-cold", "lantern-lit",
    "mountain-soft", "foggy", "blurring", "midnight", "dawn-soft",
    # Colors
    "blue", "amber", "pewter", "copper", "slate",
    "silver", "ochre", "ember", "charcoal", "jade",
    "indigo", "citrine", "opal", "smoke-grey", "weather-worn",
    # Voice + memory
    "plainsong", "careful", "listening", "half-remembered", "watchful",
    "patient", "kindly", "unhurried", "easy", "honest",
)

NOUNS: Sequence[str] = (
    # Landscape + water
    "river", "ember", "prairie", "breath", "harbor",
    "lantern", "bridge", "mountain", "sparrow", "willow",
    "mirror", "garden", "hollow", "threshold", "current",
    "valley", "station", "doorway", "ribbon", "tide",
    "candle", "meadow", "shore", "signal", "longing",
    "evening", "morning", "harvest", "season", "hush",
    "rumor", "weight", "ash", "smoke", "glass",
    "oak", "cedar", "pine", "hawthorn", "lichen",
    "moss", "hillside", "shadow", "brink", "edge",
    # Domestic + textile
    "page", "margin", "kettle", "song", "whisper",
    "room", "attic", "parlor", "alley", "corridor",
    "fold", "seam", "hem", "ledge", "hinge",
    "cradle", "sleeve", "lining", "watch", "hour",
    # Time + motion
    "weather", "distance", "pause", "drift", "echo",
    "footstep", "footprint", "gesture", "hand", "eye",
    # People + roles
    "name", "story", "pattern", "witness", "keeper",
    "listener", "wanderer", "neighbor", "stranger", "traveler",
    # Music
    "chorus", "refrain", "rhythm", "measure", "verse",
)

VERBS: Sequence[str] = (
    # Motion of light / water / breath
    "shimmer", "ripple", "kindle", "drift", "gather",
    "listen", "remember", "whisper", "settle", "fade",
    "return", "carry", "hold", "find", "lose",
    "weave", "breathe", "follow", "hum", "slip",
    "brighten", "soften", "deepen", "lean", "bend",
    "mend", "tilt", "unspool", "rest", "wait",
    "become", "answer", "name", "count", "fold",
    "open", "lift", "draw", "mark", "witness",
    "trace", "ferry", "anchor", "wander", "linger",
)

ADVERBS: Sequence[str] = (
    "slowly", "gently", "restlessly", "faintly", "softly",
    "brightly", "quietly", "deeply", "suddenly", "almost",
    "barely", "far", "near", "again", "eastward",
    "westward", "homeward", "inward", "outward", "sidelong",
    "lightly", "plainly", "openly", "secretly", "weatherward",
    "riverward", "hourly", "daily", "nightly", "earnestly",
    "patiently", "carefully",
)

# ---------------------------------------------------------------------------
# Grammar templates — verbatim from RESEARCH lines 755-761.
# Word counts: {4, 5, 5, 6, 7} — covers the full R-06 4-7 word range.
# ---------------------------------------------------------------------------

TEMPLATES: Sequence[str] = (
    "{adj} {noun} {verb} {adv}",                       # 4 words
    "{noun} in the {adj} {noun}",                      # 5 words
    "{adv} {verb} the {adj} {noun}",                   # 5 words
    "{adj} {noun} of {adj} {noun} {verb}",             # 6 words
    "where {adj} {noun} {verb} {adv} like {noun}",     # 7 words
)


def generate(rng: random.Random | None = None) -> str:
    """Return a 4-7 word pseudo-poem string suitable for a YouTube title.

    Args:
        rng: Optional seeded :class:`random.Random` for deterministic tests.
            When ``None`` (production default) the module-level
            :mod:`random` is used; that RNG is seeded from ``os.urandom``
            so production calls produce non-deterministic output.

    Returns:
        A whitespace-separated string of 4 to 7 lowercase tokens. The
        exact length depends on which of the 5 templates is selected;
        the template set covers lengths ``{4, 5, 6, 7}``.

    Notes:
        - Pure data, no side effects. Safe to call from any thread.
        - The output is a *default* for an editable title field per D-23;
          callers must not treat it as a forced value.
    """
    chooser = rng if rng is not None else random
    template = chooser.choice(TEMPLATES)
    return template.format(
        adj=chooser.choice(ADJECTIVES),
        noun=chooser.choice(NOUNS),
        verb=chooser.choice(VERBS),
        adv=chooser.choice(ADVERBS),
    ).strip()
