"""Filename + dominant-trait resolver for region export (Plan 03-04a — EXP-02 / D-A4-1, D-A4-2).

Qt-free. Resolves the auto-name ``{recorded_date}_{HHMMSS}_{trait}.{ext}``
including the ``_NN`` uniqueness suffix, and computes the per-region
dominant heatmap trait by walking the heatmap cache for the source's
``cache_key``.

N-3 invariant: this module is pure stdlib. The Qt-touching
``ExportRunnable`` lives next door (Plan 03-04b) and consumes the two
public functions defined here.

D-A4-1 (filename pattern):
    * ``recorded_date``: source ``mtime`` formatted ``YYYY-MM-DD`` (v1; audio
      metadata is a stretch we have not landed yet).
    * ``source_offset``: ``region_start_sec`` formatted ``HHMMSS``
      (seconds precision — supersedes REQ-EXP-02's HHMM wording per
      CONTEXT D-A4-1).
    * ``trait``: caller-supplied — use :func:`dominant_trait_for_region`
      for live data.
    * Uniqueness: first collision → ``_02``, second → ``_03``, etc. Two-digit
      zero-padded counter that grows to ``_999`` worst case.

D-A4-2 (per-region trait derivation):
    * The AI/DSP heatmap backend was removed in quick-260701-muv as
      verified dead code. ``_HEATMAP_REGISTRY`` is now permanently empty
      and ``cache_root/'heatmaps'/{cache_key}`` is never populated, so
      trait derivation ALWAYS yields the ``"clip"`` fallback
      (``_TRAIT_FALLBACK``). Region exports already read
      ``..._clip.<ext>`` today — this is unchanged observable behavior.
    * :func:`dominant_trait_for_region` keeps its signature and its
      heatmap-directory probe, then returns ``"clip"`` directly.

Security (T-03-04a-01 / T-03-04a-02 / T-03-04a-04):
    * Trait tokens are validated against ``^[a-z0-9_-]+$`` BEFORE any path
      arithmetic — defense-in-depth even though trait labels come from
      subclasses we control.
    * Extensions are restricted to an allow-list (``mp3``, ``wav``).
    * Collision counter is capped at ``_NAMING_MAX_COLLISION = 999``;
      exhaustion raises ``RuntimeError`` rather than looping forever.
    * Every ``subclass().dominant_trait(slice_)`` call is wrapped in
      try/except — a hostile or buggy subclass cannot block the resolver
      for the other heatmaps.
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path


# quick-260701-muv — the AI/DSP heatmap backend (TensorFlow/Essentia +
# every Heatmap subclass) was removed as verified dead code. No worker
# ever populated ``cache_root/heatmaps/`` and this registry's only reader,
# :func:`dominant_trait_for_region`, always fell through to the ``clip``
# fallback in practice. The registry is now permanently EMPTY: no
# subclasses are registered, so the resolver always returns
# ``_TRAIT_FALLBACK`` ("clip"). Kept as a named symbol so the public
# contract (and any future re-introduction) has an obvious hook.
_HEATMAP_REGISTRY: dict[str, type] = {}

# Defensive cap — ``_NN`` never grows past 3 digits in practice. If 999
# collisions happen on the same offset, something is very wrong and the
# resolver fails loudly instead of looping forever.
_NAMING_MAX_COLLISION = 999

# Per CONTEXT D-A4-2: fall back to a stable token when no heatmaps are
# cached for the source. ``clip`` is filesystem-safe and conveys "we
# don't know the trait yet, but here's your export anyway".
_TRAIT_FALLBACK = "clip"

# Defense-in-depth regex applied to the trait token BEFORE any path
# arithmetic (T-03-04a-01). Lowercase alphanumerics plus hyphen and
# underscore — explicitly excludes path separators and the dot character.
_TRAIT_SAFE_RE = re.compile(r"^[a-z0-9_-]+$")

# Output formats accepted by the resolver. D-A4-4 locks 320 kbps CBR MP3
# and float32 WAV as the two first-class export formats consumed by Plan
# 03-04b's ExportRunnable. Add new extensions here AND in the export
# writer when the format set grows.
_SUPPORTED_EXTS = frozenset({"mp3", "wav"})


def resolve_filename(
    source_path: Path,
    region_start_sec: float,
    trait: str,
    ext: str,
    output_dir: Path,
) -> Path:
    """Resolve ``{recorded_date}_{HHMMSS}_{trait}.{ext}`` with ``_NN`` collision suffix.

    Per CONTEXT D-A4-1:
        * ``recorded_date`` = source ``mtime`` → ``YYYY-MM-DD`` (v1; audio
          metadata is stretch).
        * ``source_offset`` = ``region_start_sec`` → ``HHMMSS`` (seconds
          precision per CONTEXT D-A4-1).
        * ``trait`` = caller-supplied; use :func:`dominant_trait_for_region`
          for live data.
        * Uniqueness suffix: first collision → ``_02``, second → ``_03``,
          etc. (two-digit zero-padded counter; grows past two digits at
          ``_100``).

    Args:
        source_path: Original recording — used only to read ``mtime`` for
            the ``recorded_date`` token.
        region_start_sec: Region start as seconds from source start.
        trait: Dominant trait label or ``"clip"`` fallback. Must be
            filesystem-safe (regex ``^[a-z0-9_-]+$``).
        ext: ``"mp3"`` or ``"wav"`` — D-A4-4.
        output_dir: Target directory. Caller is responsible for ensuring
            the directory exists.

    Returns:
        A :class:`Path` that does NOT yet exist in ``output_dir``.

    Raises:
        ValueError: ``trait`` is not filesystem-safe, or ``ext`` is not
            in the allow-list. Raised BEFORE any path arithmetic.
        RuntimeError: ``_NAMING_MAX_COLLISION`` was not enough — extremely
            unlikely in practice.
    """
    if not _is_filesystem_safe_trait(trait):
        raise ValueError(f"Unsafe trait token: {trait!r}")
    if ext not in _SUPPORTED_EXTS:
        raise ValueError(
            f"Unsupported ext: {ext!r} (must be one of {sorted(_SUPPORTED_EXTS)})"
        )
    # WR-06 — os.stat can raise FileNotFoundError if the source was
    # deleted between file-open and export, or PermissionError if the
    # file's directory had its permissions changed. The exception used
    # to propagate up out of MainWindow._on_export_region_requested
    # (which has no try/except around resolve_filename) and surface as
    # a Python traceback in the Qt event loop. Fall back to "today's
    # date" so the export proceeds with a best-effort recorded_date
    # token rather than failing the whole export.
    try:
        st = os.stat(source_path)
        recorded_date = datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d")
    except OSError:
        recorded_date = datetime.now().strftime("%Y-%m-%d")
    total = int(region_start_sec)
    hhmmss = (
        f"{total // 3600:02d}"
        f"{(total % 3600) // 60:02d}"
        f"{total % 60:02d}"
    )
    base = f"{recorded_date}_{hhmmss}_{trait}"
    candidate = output_dir / f"{base}.{ext}"
    if not candidate.exists():
        return candidate
    # Collision — append ``_NN`` starting at ``_02``. The format spec
    # ``{n:02d}`` zero-pads to two digits but grows naturally past three
    # digits at ``_100`` so the cap at 999 only short-circuits, never
    # silently truncates.
    for n in range(2, _NAMING_MAX_COLLISION + 1):
        candidate = output_dir / f"{base}_{n:02d}.{ext}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find a unique name for {base}")


def _is_filesystem_safe_trait(trait: str) -> bool:
    """Trait token allow-list: lowercase letters, digits, hyphen, underscore.

    Anything else — including path separators, dots, whitespace, uppercase
    letters, non-ASCII — fails the regex. Defense-in-depth for
    T-03-04a-01: even though trait labels are emitted by Heatmap
    subclasses we control, the regex protects against a buggy or hostile
    subclass that returns a label with a path separator.
    """
    return bool(_TRAIT_SAFE_RE.fullmatch(trait))


def dominant_trait_for_region(
    cache_root: Path,
    cache_key_hex: str,
    region_start_sec: float,
    region_end_sec: float,
) -> str:
    """Return the per-region dominant trait label — now always ``"clip"``.

    Historically (CONTEXT D-A4-2) this walked every cached
    ``<heatmap_name>.dat`` under ``cache_root / 'heatmaps' / {cache_key}``,
    sliced to ``[region_start_sec, region_end_sec]``, and picked the
    highest-scoring subclass label. The AI/DSP heatmap backend was removed
    in quick-260701-muv as verified dead code — nothing ever populated the
    heatmap cache directory, so this function always fell through to the
    ``"clip"`` fallback in practice. With the empty registry it now returns
    ``_TRAIT_FALLBACK`` ("clip") directly (after the unchanged directory
    probe). Signature and observable behavior are preserved.

    Args:
        cache_root: Output of
            :func:`marmelade.paths.default_cache_root` (or override).
        cache_key_hex: 16-char lowercase hex — matches
            :func:`marmelade.audio.proxy_cache.cache_key`.
        region_start_sec: Lower bound in seconds.
        region_end_sec: Upper bound in seconds.

    Returns:
        Trait label string. Always filesystem-safe (matches
        ``^[a-z0-9_-]+$``).
    """
    heatmap_dir = cache_root / "heatmaps" / cache_key_hex
    if not heatmap_dir.exists() or not heatmap_dir.is_dir():
        return _TRAIT_FALLBACK

    # quick-260701-muv — the AI/DSP heatmap backend was removed, so
    # ``_HEATMAP_REGISTRY`` is permanently empty and no on-disk
    # ``<name>.dat`` can ever resolve to a subclass. The old scan/load/
    # slice loop could therefore never produce a winning label; the only
    # possible answer is the ``clip`` fallback. Return it directly. The
    # ``heatmap_dir.exists()`` guard above is retained so the signature
    # and directory-probe behavior are unchanged for callers.
    return _TRAIT_FALLBACK
