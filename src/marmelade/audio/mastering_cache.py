"""Mastered-WAV cache path resolver — Phase 7 RESEARCH §Pattern 3.

Single-file invalidation strategy (supersedes D-10's literal
``.meta.json`` sidecar): embed the source proxy's ``cache_key`` (which
incorporates ``mtime_ns`` via :func:`marmelade.audio.proxy_cache.cache_key`)
directly into the mastered cache filename. When the source changes the
``cache_key`` changes, the filename changes, and the stale file
self-prunes naturally; no separate sidecar to keep in sync.

Threat model (T-7-02 — Tampering): all three identifiers used to
compose the cache filename are regex-validated BEFORE any filesystem
use. The alphabets are tight hex by construction (xxh64 hexdigest for
``source_cache_key``, uuid4().hex for ``keeper_id``, SHA-1 truncated
hex for ``config_hash``) but defense in depth via explicit regex
matches the heatmap_cache + sidecar_cache traversal guards.

Re-export discipline (Reuse Discipline 8): :func:`cache_key` is
re-exported from :mod:`marmelade.audio.proxy_cache` — not wrapped —
so identity assertions and worker test pins continue to hold across
the audio tier.
"""

from __future__ import annotations

import re
from pathlib import Path

# Re-export — identity preservation per Reuse Discipline 8.
# Test ``test_cache_key_reexport_identity_preserved`` pins this.
from marmelade.audio.proxy_cache import cache_key  # noqa: F401 — re-export

# T-7-02 mitigation: regex-validate all three identifiers before any
# filesystem use. The alphabets are tight hex by construction but the
# defense in depth is the load-bearing invariant.
_KEY_RE = re.compile(r"^[0-9a-f]{16}$")  # xxh64 hexdigest
_KEEPER_ID_RE = re.compile(r"^[0-9a-f]{32}$")  # uuid4().hex
_CONFIG_HASH_RE = re.compile(r"^[0-9a-f]{12}$")  # config_hash() output


def mastered_cache_path(
    cache_root: Path | str,
    source_cache_key: str,
    keeper_id: str,
    config_hash: str,
) -> Path:
    """Resolve ``<cache_root>/mastered/<src_key>-<keeper_id>-<config_hash>.wav``.

    All three identifiers are validated against tight hex regexes before
    any filesystem use (T-7-02 — no Path.join with attacker-controllable
    content). On regex mismatch raises :class:`ValueError` naming the
    offending identifier so callers can surface a precise diagnostic.

    Args:
        cache_root: Cache root directory (e.g., ``mastered_cache_dir().parent``).
        source_cache_key: 16-hex source-proxy cache key.
        keeper_id: 32-hex keeper UUID (uuid4().hex shape).
        config_hash: 12-hex chain config hash (output of
            :func:`marmelade.audio.mastering.chain.config_hash`).

    Returns:
        Absolute path under ``cache_root / "mastered"``. The directory
        is NOT created — callers run ``parent.mkdir(parents=True,
        exist_ok=True)`` immediately before writing.

    Raises:
        ValueError: with a message naming the offending identifier
            (``source_cache_key``, ``keeper_id``, or ``config_hash``)
            when any regex mismatch occurs.
    """
    if not _KEY_RE.match(source_cache_key):
        raise ValueError(f"Invalid source_cache_key: {source_cache_key!r}")
    if not _KEEPER_ID_RE.match(keeper_id):
        raise ValueError(f"Invalid keeper_id: {keeper_id!r}")
    if not _CONFIG_HASH_RE.match(config_hash):
        raise ValueError(f"Invalid config_hash: {config_hash!r}")
    return (
        Path(cache_root)
        / "mastered"
        / f"{source_cache_key}-{keeper_id}-{config_hash}.wav"
    )


def is_mastered_cache_fresh(path: Path | str) -> bool:
    """Return True iff a non-empty WAV exists at ``path``.

    Freshness against source ``mtime`` is already encoded in the
    filename (``source_cache_key`` carries ``mtime_ns`` via
    :func:`proxy_cache.cache_key`) — if the file exists at this exact
    name, it is by definition matched to the current source + chain.
    """
    p = Path(path)
    return p.exists() and p.stat().st_size > 0


__all__ = [
    "cache_key",
    "mastered_cache_path",
    "is_mastered_cache_fresh",
]
