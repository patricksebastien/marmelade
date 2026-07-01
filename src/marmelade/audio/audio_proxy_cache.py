"""Qt-free path arithmetic + disk-preflight helpers for the audio proxy layer (AUD-04).

This module is *pure Python* — no GUI-toolkit imports. N-3 invariant: the
``audio/`` package stays toolkit-free so its logic is unit-testable without a
graphical event loop. The Qt-touching neighbor is
:mod:`marmelade.audio.audio_proxy_worker`; the streaming-transcode logic
itself will live in :mod:`marmelade.audio.audio_proxy_builder` (Plan 02).
This file owns ONLY: filename resolution, freshness probe, disk-preflight
math, cache-size accounting, and the subtree-clear helper.

Canonical float32 stereo WAV cache (D-01 + D-03):

    Filenames live exclusively under ``<cache_root>/audio/<key>.proxy.wav``
    where ``<key>`` is the 16-character lowercase hex digest returned by
    :func:`marmelade.audio.proxy_cache.cache_key`. Source mtime/size/head/
    tail are encoded into the key, so a source-file change invalidates the
    proxy filename automatically — NO BBC-v2-style sidecar header is needed
    (RESEARCH §"WAV header contents — recommended: standard WAV header
    only"). The standard WAV header that ``soundfile.SoundFile(mode='w',
    subtype='FLOAT')`` writes carries channels + sample rate; nothing else
    needs to be encoded.

Cache key (D-04 single-source freshness):

    The xxh64 helper :func:`cache_key` is re-exported verbatim from
    :mod:`marmelade.audio.proxy_cache` so the waveform-proxy, heatmap,
    and audio-proxy caches share invalidation when source size/mtime/head/
    tail change. Do NOT redefine the function here — it would split the
    invariant across two modules.

Security:
    * :func:`audio_proxy_path` requires the key to match ``^[0-9a-f]{16}$``
      via :data:`_KEY_RE` BEFORE any path arithmetic (T-02.1-01 — traversal
      mitigation). Cache filenames can ONLY be 16-char lowercase hex
      digests; user-supplied path components never reach the file system.
    * The atomic-write contract (``.proxy.tmp`` → ``os.replace(.tmp, .wav)``)
      lives in the builder module (Plan 02); a partial ``.proxy.tmp`` is
      never visible under the canonical ``.proxy.wav`` name.
    * :func:`check_disk_space` refuses any build when free disk is below
      ``expected + 1 GiB`` safety margin (T-02.1-02 — disk-fill DoS
      mitigation; RESEARCH Open-Q-2).
"""

# RESEARCH Open-Q-4 / D-12 — startup `.proxy.tmp` GC pass deferred to 2.1.1
# polish (see CONTEXT § Deferred Ideas). The in-flight builder owns its own
# `.tmp` cleanup on cancel; a leftover after a hard OS kill is acceptable for
# v1 because the manual File → Clear audio proxy cache menu covers cleanup.

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

# D-04 — re-export so the heatmap, waveform-proxy and audio-proxy caches all
# share invalidation when source size/mtime/head/tail change. Imported via
# ``cache_key`` from this module by Plans 02-04 so callers see a single
# import surface; identity is preserved so a future tweak to the hash inputs
# propagates to all three caches simultaneously.
from marmelade.audio.proxy_cache import cache_key  # noqa: F401 — re-export

# Cache key shape: 16 lowercase hex chars (xxhash.xxh64 hexdigest length).
# Identical regex to ``proxy_cache._KEY_RE`` so the two caches share key
# validation (T-02.1-01).
_KEY_RE = re.compile(r"^[0-9a-f]{16}$")

# RESEARCH Open-Q-2 — 1 GiB safety margin for the disk-preflight check.
# Generous enough to avoid borderline disk-pressure on systems near the
# edge; for sub-100 MiB proxies the margin dominates which is exactly the
# right call — if the user is within 1 GiB of empty, refusing to write a
# 10 MB cache prevents an OOM downstream.
_DISK_SAFETY_MARGIN_BYTES = 1 * 1024**3


def audio_proxy_path(cache_root: Path, key: str) -> Path:
    """Return ``cache_root / 'audio' / f'{key}.proxy.wav'``.

    Traversal-guard mitigation (T-02.1-01): ``key`` MUST match
    ``^[0-9a-f]{16}$``. Anything else raises :class:`ValueError`. Cache
    filenames can ONLY be 16-char lowercase hex digests; user-supplied
    path components never reach the file system.

    Does NOT create the directory — :func:`audio_proxy_builder.build_audio_proxy`
    (Plan 02) does that immediately before writing the ``.tmp`` sibling.
    """
    if not _KEY_RE.match(key):
        raise ValueError(f"Invalid cache key: {key!r}")
    return Path(cache_root) / "audio" / f"{key}.proxy.wav"


def audio_proxy_is_fresh(cache_root: Path, source_path: Path) -> Path | None:
    """Return the proxy path if a fresh proxy exists for ``source_path``, else None.

    Fresh means: ``audio_proxy_path(cache_root, cache_key(source_path))``
    exists on disk. ``cache_key`` encodes size+mtime+head+tail, so a
    source-file change invalidates the proxy filename automatically (D-04).
    NO custom header read is needed — libsndfile owns the WAV header; if a
    file exists with the freshness-derived name, it IS the canonical proxy
    (RESEARCH §"WAV header contents").
    """
    key = cache_key(source_path)
    proxy = audio_proxy_path(cache_root, key)
    return proxy if proxy.exists() else None


def expected_proxy_bytes(duration_s: float, sample_rate: int) -> int:
    """Return the bytes the proxy will occupy on disk.

    Formula: ``int(duration_s * sample_rate) * 2 (channels=stereo) * 4
    (float32) + 128`` (128 B is a generous header overhead — the actual
    WAV header for the FLOAT subtype is ~44 B; the 3x cushion absorbs any
    LIST chunk soundfile might emit).

    D-01: the proxy is always stereo regardless of source channel count;
    mono sources are duplicated, >2-channel sources are downmixed to the
    first two channels. D-03: sample_rate matches the source — no resample.
    """
    return int(duration_s * sample_rate) * 2 * 4 + 128


def check_disk_space(cache_root: Path, expected_bytes: int) -> tuple[bool, int, int]:
    """Return ``(ok, needed_with_margin, free_bytes)`` for the proxy build.

    ``ok`` is True iff ``shutil.disk_usage(cache_root).free >= expected
    + 1 GiB``. Callers use the returned tuple to format the friendly error
    message (RESEARCH §"Error message wording" — the UI layer in Plan 04
    composes the QMessageBox text from ``needed`` and ``free``).

    The cache root is created on demand so the caller does not need to
    mkdir before the pre-flight check.
    """
    cache_root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(str(cache_root)).free
    needed = int(expected_bytes) + _DISK_SAFETY_MARGIN_BYTES
    return (free >= needed, needed, free)


def audio_cache_size_bytes(cache_root: Path) -> int:
    """Return the total bytes occupied by ``cache_root / 'audio'`` (D-09 footer).

    Uses :func:`os.scandir` to avoid materialising a list of paths. Returns
    0 when the directory does not yet exist. Best-effort: a stat failure on
    a single entry is skipped silently — the footer is informational, not
    load-bearing (T-02.1-05 accepted).
    """
    audio_dir = Path(cache_root) / "audio"
    if not audio_dir.exists():
        return 0
    total = 0
    try:
        with os.scandir(str(audio_dir)) as it:
            for entry in it:
                try:
                    total += entry.stat().st_size
                except OSError:
                    pass
    except OSError:
        pass
    return total


def clear_audio_cache(cache_root: Path) -> int:
    """Delete ``cache_root / 'audio'`` and return the bytes freed (D-08).

    Returns 0 when the directory does not exist. ``shutil.rmtree`` with
    ``ignore_errors=True`` so a transient access failure does not surface
    to the user — the operation is best-effort cleanup, not a guarantee.
    Idempotent: a second invocation on the same root returns 0.

    T-02.1-03 disposition is `accept` — the cache root is under
    :func:`marmelade.paths.default_cache_root` (OS-managed via
    QStandardPaths), so a hostile symlink replacement is out of the
    in-scope threat model.
    """
    audio_dir = Path(cache_root) / "audio"
    freed = audio_cache_size_bytes(cache_root)
    shutil.rmtree(str(audio_dir), ignore_errors=True)
    return freed
