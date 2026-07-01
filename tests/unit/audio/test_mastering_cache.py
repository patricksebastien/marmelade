"""Wave 0 RED stub — :func:`mastered_cache_path` + freshness + identity reuse.

Pinned invariants:
* Path regex validators reject malformed ``source_cache_key`` /
  ``keeper_id`` / ``config_hash`` (T-7-02 mitigation: ValueError before
  any filesystem use).
* ``is_mastered_cache_fresh`` returns True iff the file exists and is
  non-empty.
* :func:`cache_key` re-export preserves identity with
  :func:`marmelade.audio.proxy_cache.cache_key` (Reuse Discipline 8).

Phase 7 — Plan 01 Wave 0 (07-01-PLAN.md Task 1).
"""

from __future__ import annotations

from pathlib import Path

import pytest


_VALID_KEY = "0123456789abcdef"  # 16-hex
_VALID_KEEPER_ID = "0" * 32  # 32-hex (uuid4().hex)
_VALID_CONFIG_HASH = "0123456789ab"  # 12-hex


@pytest.mark.parametrize(
    "bad_key, bad_keeper, bad_hash, raises_on",
    [
        ("too-short", _VALID_KEEPER_ID, _VALID_CONFIG_HASH, "source_cache_key"),
        ("XYZ" + "0" * 13, _VALID_KEEPER_ID, _VALID_CONFIG_HASH, "source_cache_key"),
        (_VALID_KEY, "short", _VALID_CONFIG_HASH, "keeper_id"),
        (_VALID_KEY, "Z" * 32, _VALID_CONFIG_HASH, "keeper_id"),
        (_VALID_KEY, _VALID_KEEPER_ID, "too-short", "config_hash"),
        (_VALID_KEY, _VALID_KEEPER_ID, "XYZ" + "0" * 9, "config_hash"),
    ],
    ids=[
        "bad_key_short",
        "bad_key_nonhex",
        "bad_keeper_short",
        "bad_keeper_nonhex",
        "bad_hash_short",
        "bad_hash_nonhex",
    ],
)
def test_path_regex_validators_reject_invalid(
    tmp_path: Path, bad_key: str, bad_keeper: str, bad_hash: str, raises_on: str
):
    """Any of the three identifiers failing its regex must raise ValueError.

    The error message must name the offending identifier so callers can
    surface a precise diagnostic.
    """
    from marmelade.audio.mastering_cache import mastered_cache_path

    with pytest.raises(ValueError) as excinfo:
        mastered_cache_path(tmp_path, bad_key, bad_keeper, bad_hash)
    # Soft check — message should mention the failing identifier name.
    assert raises_on in str(excinfo.value).lower() or raises_on.replace("_", " ") in str(
        excinfo.value
    ).lower()


def test_path_layout(tmp_path: Path):
    """Valid inputs produce ``<root>/mastered/<key>-<keeper>-<hash>.wav``."""
    from marmelade.audio.mastering_cache import mastered_cache_path

    p = mastered_cache_path(tmp_path, _VALID_KEY, _VALID_KEEPER_ID, _VALID_CONFIG_HASH)
    assert p.parent == tmp_path / "mastered"
    assert p.name == f"{_VALID_KEY}-{_VALID_KEEPER_ID}-{_VALID_CONFIG_HASH}.wav"


def test_freshness_is_existence_plus_size(tmp_path: Path):
    """``is_mastered_cache_fresh`` is True iff the file exists AND size > 0."""
    from marmelade.audio.mastering_cache import is_mastered_cache_fresh

    missing = tmp_path / "absent.wav"
    assert not is_mastered_cache_fresh(missing)

    empty = tmp_path / "empty.wav"
    empty.write_bytes(b"")
    assert not is_mastered_cache_fresh(empty)

    populated = tmp_path / "ok.wav"
    populated.write_bytes(b"a")
    assert is_mastered_cache_fresh(populated)


def test_cache_key_reexport_identity_preserved():
    """``mastering_cache.cache_key`` must be the SAME callable as
    ``proxy_cache.cache_key`` (Reuse Discipline 8 — re-export, do not wrap).
    """
    from marmelade.audio import proxy_cache
    from marmelade.audio.mastering_cache import cache_key as masterered_cache_key

    assert masterered_cache_key is proxy_cache.cache_key
