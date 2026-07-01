"""Pure-numpy waveform render-mode registry (quick-260627-gb7).

The waveform viewer's ``render_proxy`` builds an aggregated ``(N, 2)`` int16
min/max proxy (at most ``MAX_RENDER_PROXY_PAIRS`` bins — viewport density,
USER FEEDBACK 2026-05-13). This module turns that SAME aggregated array into
the flat saw-wave ``y`` array + the ``(y_min, y_max)`` range that the view
hands to ``PlotDataItem.setData`` / ``setYRange(..., padding=0)``, under one
of three Tier-1 render modes:

* CLASSIC — identity passthrough (today's behavior, byte-identical).
* DB      — log/dB amplitude scale that lifts quiet detail (sign-preserving).
* ENERGY  — single-sided rectified PEAK envelope (max(|min|, |max|) per bin).

Extensibility (CLAUDE.md): dispatch is a ``dict`` keyed by :class:`RenderMode`,
so a 4th mode is one new ``_REGISTRY`` entry + one ``MODE_LABELS`` entry — no
widget rewiring. Spectral-centroid tint and RGB band modes are DEFERRED to a
future Spectrogram phase (they need spectral data the min/max proxy lacks) and
are NOT implemented here.

NOTE: ENERGY is a PEAK-energy envelope, NOT true RMS — true RMS is unavailable
from min/max proxy data. dB and Energy are pure viewport-density render
transforms over the same aggregated array; no new precompute, no FFT/STFT, no
background jobs.

N-3 invariant: this module is part of ``audio/`` and is toolkit-free — pure
NumPy + stdlib, with NO GUI-toolkit imports. All transforms are vectorized
(no per-bin Python loop), matching :func:`peak_builder.windowed_minmax` style.

Transform contract:
    Each transform takes an aggregated ``pairs: (N, 2) int16`` array (min in
    column 0, max in column 1) and returns
    ``(y_flat: np.ndarray[float32], y_range: tuple[float, float])`` where
    ``y_flat`` has length ``2 * N`` laid out as the saw-wave interleave
    ``[v0_lo, v0_hi, v1_lo, v1_hi, ...]`` matching the existing render
    contract, and ``y_range`` is the ``(min, max)`` for
    ``setYRange(*y_range, padding=0)``. An empty ``(0, 2)`` input returns an
    empty float32 array + the mode's nominal y_range (a cleared plot must not
    crash).
"""

from __future__ import annotations

import enum
from typing import Callable

import numpy as np

# int16 full-scale magnitude (a +full-scale sample is +32767).
INT16_FULL_SCALE = 32767.0

# Quiet-detail floor for the dB mode, in decibels. A sample at or below this
# magnitude maps to ~0 on the plotted scale; a full-scale sample maps to the
# top. -60 dB is a generous floor that lifts quiet jam detail without turning
# the noise floor into visual mush.
DB_FLOOR = -60.0


class RenderMode(enum.Enum):
    """The waveform render modes (CLASSIC is first/default).

    Tier-1 (gb7) are amplitude transforms over the min/max proxy (CLASSIC, DB,
    ENERGY). Tier-2 (Phase 11) are SPECTRAL modes that consume per-column
    spectral arrays (centroid / band energies) rather than min/max pairs and so
    are NOT in :data:`_REGISTRY`; their color derivations live in the Qt-free
    ``*_colors`` helpers below and are APPLIED by ``WaveformView`` (N-3 split).
    """

    CLASSIC = "classic"
    DB = "db"
    ENERGY = "energy"
    # Tier-2 spectral modes (Phase 11). These do NOT use _REGISTRY: they consume
    # spectral arrays, not min/max pairs.
    SPECTROGRAM = "spectrogram"
    CENTROID = "centroid"
    RGB_BAND = "rgb_band"


# User-facing combo labels, one per mode.
MODE_LABELS: dict["RenderMode", str] = {
    RenderMode.CLASSIC: "Classic",
    RenderMode.DB: "dB",
    RenderMode.ENERGY: "Energy",
    RenderMode.SPECTROGRAM: "Spectrogram",
    RenderMode.CENTROID: "Spectral centroid (tint)",
    RenderMode.RGB_BAND: "Frequency bands (RGB)",
}

# Default centroid Hz log-scale bounds (full audible range). The view may pass
# a tighter (fmin, fmax) to better use the LUT for a given file.
CENTROID_FMIN_HZ = 20.0
CENTROID_FMAX_HZ = 24000.0
# Divide-by-zero floor for degenerate (all-silent) columns in RGB-band peak
# normalization (T-11-04 mitigation).
_BAND_PEAK_FLOOR = 1e-9


def _classic(pairs: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    """Identity passthrough — value-identical to today's int16 saw-wave.

    ``y_flat`` is ``pairs.reshape(-1)`` cast to float32 (the cast does not
    change any value in the int16 domain); ``y_range`` is the full int16 span.
    """
    y_flat = pairs.reshape(-1).astype(np.float32)
    return y_flat, (-32768.0, INT16_FULL_SCALE)


def _db(pairs: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    """Sign-preserving log/dB amplitude scale that lifts quiet detail.

    For each signed amplitude ``a`` in ``[-32768, 32767]``:
        mag = |a| / 32767
        db  = 20 * log10(max(mag, floor))   # floor avoids log(0)
        db  = clip(db, DB_FLOOR, 0)
        y   = sign(a) * (db - DB_FLOOR) / (-DB_FLOOR) * 32767

    So a full-scale sample maps to ~ +/-32767 and a DB_FLOOR-magnitude sample
    maps to ~0. Fully vectorized (np.log10 over the flat array); the explicit
    magnitude floor keeps an all-zero input finite (no inf/nan).
    """
    if pairs.shape[0] == 0:
        return np.zeros(0, dtype=np.float32), (-32768.0, INT16_FULL_SCALE)

    flat = pairs.reshape(-1).astype(np.float64)
    sign = np.sign(flat)
    mag = np.abs(flat) / INT16_FULL_SCALE
    # Explicit floor BEFORE log10 — clamp magnitude to 10**(DB_FLOOR/20) so
    # log(0) never produces -inf (T-gb7-02 mitigation).
    mag_floor = 10.0 ** (DB_FLOOR / 20.0)
    mag = np.maximum(mag, mag_floor)
    db = 20.0 * np.log10(mag)
    db = np.clip(db, DB_FLOOR, 0.0)
    # Normalize [DB_FLOOR, 0] -> [0, 1], re-apply sign, scale to int16 range.
    norm = (db - DB_FLOOR) / (-DB_FLOOR)
    y_flat = (sign * norm * INT16_FULL_SCALE).astype(np.float32)
    return y_flat, (-32768.0, INT16_FULL_SCALE)


def _energy(pairs: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    """Single-sided rectified PEAK envelope.

    ``env = max(|min|, |max|)`` per bin (vectorized). The saw-wave interleave
    is ``[+env0, +env0, +env1, +env1, ...]`` so the curve draws a filled
    single-sided envelope hugging the top of each bin (chosen over the
    ``[0, env]`` baseline-rise layout because the doubled-peak form keeps the
    same two-points-per-bin saw-wave shape the render contract expects and
    reads as a solid loudness band rather than a comb). ``y_range`` is
    single-sided ``(0, 32767)``.
    """
    if pairs.shape[0] == 0:
        return np.zeros(0, dtype=np.float32), (0.0, INT16_FULL_SCALE)

    # Widen to int64 before abs so -32768 (whose negation overflows int16)
    # is handled correctly, then take the per-bin max magnitude.
    wide = pairs.astype(np.int64)
    env = np.maximum(np.abs(wide[:, 0]), np.abs(wide[:, 1])).astype(np.float32)
    # Interleave [env0, env0, env1, env1, ...] -> length 2*N.
    y_flat = np.repeat(env, 2)
    return y_flat, (0.0, INT16_FULL_SCALE)


_REGISTRY: dict["RenderMode", Callable[[np.ndarray], tuple[np.ndarray, tuple[float, float]]]] = {
    RenderMode.CLASSIC: _classic,
    RenderMode.DB: _db,
    RenderMode.ENERGY: _energy,
}


def transform_for(
    mode: "RenderMode",
) -> Callable[[np.ndarray], tuple[np.ndarray, tuple[float, float]]]:
    """Return the registered transform callable for ``mode``.

    Raises:
        KeyError: if ``mode`` is not a registered :class:`RenderMode`.
    """
    return _REGISTRY[mode]


# ---------------------------------------------------------------------------
# Tier-2 spectral color math (Phase 11, R-5 / R-6).
#
# These are NOT amplitude transforms over min/max pairs (so they are not in
# _REGISTRY). They take per-column spectral arrays and return a per-column
# ``(N, 3)`` uint8 RGB array that ``WaveformView`` applies as pens / an
# ImageItem row. Pure NumPy — no Qt, no pyqtgraph import (N-3): the Magma LUT is
# INJECTED by the view; here a pure-numpy dark->bright ramp is the default so the
# math stays unit-testable in isolation.
# ---------------------------------------------------------------------------

# Cached fallback LUT so repeated calls do not re-allocate.
_DEFAULT_TINT_LUT: np.ndarray | None = None


def _default_tint_lut() -> np.ndarray:
    """A 256x3 uint8 dark->bright ramp used when no Magma LUT is injected.

    Monotonically increasing luminance from near-black to near-white so the
    centroid-tint contract (higher centroid = brighter) holds without importing
    pyqtgraph here (N-3). The view passes the real Magma LUT for the perceptual
    look; this keeps the math testable and the module Qt-free.
    """
    global _DEFAULT_TINT_LUT
    if _DEFAULT_TINT_LUT is None:
        ramp = np.linspace(0.0, 255.0, 256, dtype=np.float64)
        _DEFAULT_TINT_LUT = np.round(
            np.stack([ramp, ramp, ramp], axis=1)
        ).astype(np.uint8)
    return _DEFAULT_TINT_LUT


def centroid_tint_colors(
    centroid: np.ndarray,
    lut: np.ndarray | None = None,
    *,
    fmin: float = CENTROID_FMIN_HZ,
    fmax: float = CENTROID_FMAX_HZ,
) -> np.ndarray:
    """Map per-column spectral centroid to a per-column ``(N, 3)`` uint8 tint.

    A higher centroid (more treble) yields a higher LUT index — a brighter color
    on the injected Magma (or default dark->bright) palette — so the tint shares
    the spectrogram's visual language (R-5 / D-01).

    Args:
        centroid: ``(N,)`` per-column spectral value. Values already in ``[0, 1]``
            are treated as a NORMALISED position on the LUT; values outside that
            range are treated as **Hz** and mapped on a log scale between
            ``fmin`` and ``fmax`` (``log(clip(hz, fmin, fmax)/fmin) /
            log(fmax/fmin)``). This lets callers pass either normalised positions
            (tests / proxies) or raw centroid Hz.
        lut: optional ``(256, 3+)`` uint8 lookup table (the view injects Magma).
            When ``None`` a pure-numpy dark->bright ramp is used (keeps this
            module Qt-free — N-3).
        fmin: low Hz bound for the log mapping (default full audible).
        fmax: high Hz bound for the log mapping.

    Returns:
        ``(N, 3)`` uint8 RGB, one tint per input column.
    """
    c = np.asarray(centroid, dtype=np.float64).reshape(-1)
    if c.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.uint8)

    # Decide normalised position in [0, 1].
    finite = c[np.isfinite(c)]
    treat_as_hz = finite.size > 0 and float(finite.max()) > 1.0
    if treat_as_hz:
        clipped = np.clip(c, fmin, fmax)
        norm = np.log(clipped / fmin) / np.log(fmax / fmin)
    else:
        norm = np.clip(c, 0.0, 1.0)
    norm = np.nan_to_num(norm, nan=0.0, posinf=1.0, neginf=0.0)

    table = _default_tint_lut() if lut is None else np.asarray(lut)
    n_rows = table.shape[0]
    idx = np.clip(np.round(norm * (n_rows - 1)).astype(np.intp), 0, n_rows - 1)
    # Take RGB only (drop alpha if the injected LUT carries one).
    colors = np.ascontiguousarray(table[idx, :3]).astype(np.uint8)
    return colors


def rgb_band_colors(
    low: np.ndarray,
    mid: np.ndarray,
    high: np.ndarray,
) -> np.ndarray:
    """Map per-column low/mid/high band energy to a per-column ``(N, 3)`` RGB.

    R tracks ``low`` (bass), G tracks ``mid``, B tracks ``high`` (treble). Each
    column is normalised to its own peak channel and scaled to ``[0, 255]`` so a
    bass-only column reads red, a mid-only column green, a treble-only column
    blue, and a balanced full-band column reads near-white (R-6 / D-03 splits
    bass < 250 Hz, mid 250 Hz-4 kHz, high > 4 kHz — the split itself is done by
    the spectral cache; here we only colorize the three energies).

    Args:
        low: ``(N,)`` bass-band energy per column.
        mid: ``(N,)`` mid-band energy per column.
        high: ``(N,)`` treble-band energy per column.

    Returns:
        ``(N, 3)`` uint8 RGB, one color per column.
    """
    lo = np.asarray(low, dtype=np.float64).reshape(-1)
    md = np.asarray(mid, dtype=np.float64).reshape(-1)
    hi = np.asarray(high, dtype=np.float64).reshape(-1)
    if lo.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.uint8)

    rgb = np.stack([lo, md, hi], axis=1)  # (N, 3): R=low, G=mid, B=high
    # Per-column peak normalisation; floor avoids divide-by-zero on all-silent
    # columns (T-11-04). Negative inputs clamped to 0 (energy is non-negative).
    rgb = np.maximum(rgb, 0.0)
    peak = np.maximum(rgb.max(axis=1, keepdims=True), _BAND_PEAK_FLOOR)
    colors = np.round((rgb / peak) * 255.0).astype(np.uint8)
    return colors
