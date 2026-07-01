"""Phase 7 Plan 07-06 Task 2 — Cache invalidation invariant (Pattern 3).

RESEARCH §Pattern 3 invariant: the mastered cache filename embeds the
source proxy's ``cache_key()`` (which carries the source's mtime_ns).
When the source proxy's mtime changes (someone re-imported a different
take with the same file name, or the user touched the file), the source
``cache_key`` changes → the mastered cache filename changes → the stale
mastered cache becomes orphan (still on disk under its old name) and
``is_mastered_cache_fresh`` returns False at the NEW path.

This is invalidation BY NEW FILENAME, not "delete the old file". The
orphan cleanup is out of scope for v1 (UI-SPEC §"Destructive actions").
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from marmelade.audio.mastering_cache import (
    cache_key,
    is_mastered_cache_fresh,
    mastered_cache_path,
)
from marmelade.paths import default_cache_root  # noqa: F401 — conftest patch target


SR = 44100


def _write_source_wav(path: Path, value: float = 0.1) -> None:
    """Tiny 1-second stereo float32 WAV — enough to exercise cache_key."""
    audio = np.full((SR, 2), value, dtype=np.float32)
    sf.write(str(path), audio, SR, subtype="FLOAT", format="WAV")


def test_source_mtime_change_produces_new_mastered_cache_filename(
    tmp_path: Path, tmp_cache_dir: Path
) -> None:
    """Touch the source → ``cache_key()`` changes → mastered_cache_path resolves a NEW filename.

    Pattern 3 invariant: invalidation happens because the path itself
    changes — there is NO "delete the old file" step. The previous
    cache file remains on disk as an orphan, but the keeper's current
    config resolves to a different file (which initially does not
    exist, so ``is_mastered_cache_fresh`` returns False).
    """
    src = tmp_path / "source.wav"
    _write_source_wav(src, value=0.1)

    keeper_id = "a" * 32
    chash = "abc123def456"

    # (1) Initial source state — compute the would-be mastered cache path
    # and pretend a mastering job wrote a file there.
    key_1 = cache_key(src)
    mastered_1 = mastered_cache_path(
        tmp_cache_dir, key_1, keeper_id, chash
    )
    mastered_1.parent.mkdir(parents=True, exist_ok=True)
    mastered_1.write_bytes(b"first cache content")
    assert is_mastered_cache_fresh(mastered_1) is True

    # (2) Touch the source — bump mtime_ns. We use os.utime with a
    # future timestamp to GUARANTEE the mtime moves on filesystems with
    # second-resolution mtime (most Linux ext4 has ns resolution but
    # the difference must exceed any rounding to be visible from
    # os.stat).
    new_mtime = time.time() + 5.0
    os.utime(str(src), (new_mtime, new_mtime))

    # (3) Same content, different mtime → different cache_key → different
    # mastered_cache_path → fresh-check at NEW path returns False because
    # nothing was written there yet.
    key_2 = cache_key(src)
    assert key_2 != key_1, (
        "Touching source must change cache_key (the mastered cache "
        "filename includes mtime_ns via cache_key)"
    )
    mastered_2 = mastered_cache_path(
        tmp_cache_dir, key_2, keeper_id, chash
    )
    assert mastered_2 != mastered_1, (
        "Different cache_key must resolve to different mastered cache path"
    )
    assert is_mastered_cache_fresh(mastered_2) is False, (
        "Pattern 3 — the new mastered cache filename has no file yet"
    )

    # (4) The old orphan still exists on disk (no auto-cleanup in v1).
    assert mastered_1.exists(), (
        "Old cache must remain on disk as orphan (cleanup out of scope v1)"
    )


def test_different_keeper_id_or_config_hash_produces_different_filename(
    tmp_path: Path, tmp_cache_dir: Path
) -> None:
    """Filename also embeds keeper_id + config_hash — both invalidate independently.

    Pattern 3 verified for ALL three dimensions of the mastered cache
    filename: ``<source_cache_key>-<keeper_id>-<config_hash>.wav``. Any
    change to any of the three produces a distinct path.
    """
    src = tmp_path / "source.wav"
    _write_source_wav(src)

    src_key = cache_key(src)
    keeper_a = "a" * 32
    keeper_b = "b" * 32
    hash_x = "111111111111"
    hash_y = "222222222222"

    p_a_x = mastered_cache_path(tmp_cache_dir, src_key, keeper_a, hash_x)
    p_a_y = mastered_cache_path(tmp_cache_dir, src_key, keeper_a, hash_y)
    p_b_x = mastered_cache_path(tmp_cache_dir, src_key, keeper_b, hash_x)

    assert p_a_x != p_a_y, "config_hash flip must change filename"
    assert p_a_x != p_b_x, "keeper_id flip must change filename"
    assert p_a_y != p_b_x
