"""Block-streaming STFT spectral-proxy builder (Phase 11 — R-1 / R-2).

Qt-free, CPU-only sibling of :mod:`marmelade.audio.peak_builder`. Where the
peak builder reduces streamed blocks to a min/max waveform pyramid, this module
reduces the SAME streamed blocks to the three spectral lanes that back the
Phase 11 render modes — a mel-magnitude image, a spectral centroid track, and
low/mid/high band energies — and persists them via
:func:`marmelade.audio.spectral_cache.write_spectral`.

Memory contract (CLAUDE.md / R-1): the source is iterated block-by-block via
:func:`audio_file.iter_blocks` (≤ 131_072 samples / 512 KiB per block). One
STFT is computed per block with a retained ``N_FFT - HOP`` leftover so frames
never straddle a block seam (``center=False``, Pattern 1 seam arithmetic). STFT
frames are MAX-pooled to ~1 stored column/second *inside* the block loop before
they are appended, so peak RSS stays MB-scale regardless of the 8 h source
length (no pre-pool frame accumulation — Pitfall #2).

Performance contract (R-2): pure numpy + librosa, no ``torch`` / no GPU stack.
The mel/centroid/band reduction is a handful of matmuls per block, comfortably
inside the 10-min/8 h budget. Measured (Task 2 / test_spectral_budget): a 60 s
48 kHz fixture builds in ~0.30 s → linearly extrapolated 8 h wall-clock ≈ 144 s,
well under the 600 s (10 min) ceiling. A 60 s build yields 62 stored columns →
~29 760 columns for 8 h, comfortably above the 4000-column render floor so zoom
stays crisp. Peak RSS does not scale with source duration (pooling inside the
loop keeps the carry < POOL frames — pinned by test_bounded_rss).

Cancellation contract (D-16): ``cancel_check()`` is polled at the TOP of each
block. On ``True`` a shared :class:`marmelade.audio.peak_builder.BuildCancelled`
is raised; the write path is atomic (tmp + os.replace) and any leftover
``*.dat.tmp`` siblings are defensively unlinked before the exception
propagates, so a cancelled build leaves no partial ``.dat`` (T-11-03).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import librosa
import numpy as np

from marmelade.audio import proxy_cache, spectral_cache
from marmelade.audio.audio_file import iter_blocks, probe
from marmelade.audio.peak_builder import BuildCancelled

# ---------------------------------------------------------------------------
# DSP constants (RESEARCH / locked).
# ---------------------------------------------------------------------------
N_FFT = 2048
HOP = 1024
N_MELS = 128
SR = 48000
FMAX = 24000

# Stored time resolution: ~1 column / ~0.98 s. At HOP=1024 / SR=48000 each STFT
# frame spans 1024/48000 ≈ 21.3 ms, so MAX-pooling POOL=46 frames yields one
# stored column every ~0.98 s. For 8 h that is ~29.4k columns — comfortably
# above MAX_RENDER_SPECTRAL_COLS (4000) so zoom stays crisp (Task 2 budget).
POOL = 46

# Band split (D-03): low < 250 Hz, mid 250–4000 Hz, high >= 4000 Hz.
_BAND_LO_HZ = 250.0
_BAND_HI_HZ = 4000.0

# Module-level DSP tables built ONCE (cheap; ~(128,1025) + (1025,)). Reused
# across every build so per-call cost is just the streamed matmuls.
_MEL_BASIS = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmax=FMAX)
_FFT_FREQS = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
_LO_MASK = _FFT_FREQS < _BAND_LO_HZ
_MID_MASK = (_FFT_FREQS >= _BAND_LO_HZ) & (_FFT_FREQS < _BAND_HI_HZ)
_HI_MASK = _FFT_FREQS >= _BAND_HI_HZ


def _pool_columns(
    mel_frames: np.ndarray,
    cent_frames: np.ndarray,
    band_frames: np.ndarray,
    pool: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], int]:
    """MAX/mean/sum-pool a frame run into whole stored columns.

    Returns ``(mel_cols, cent_cols, band_cols, consumed_frames)`` where each
    list holds the fully-pooled columns drawn from the leading ``consumed``
    frames; the trailing ``< pool`` frames are left for the caller to carry.

    Pooling rules (per the plan / anti-pattern guard):
        * mel  → MAX over the window (preserve transients; NEVER mean)
        * centroid → mean over the window
        * bands → sum over the window
    """
    n = mel_frames.shape[1]
    n_cols = n // pool
    mel_cols: list[np.ndarray] = []
    cent_cols: list[np.ndarray] = []
    band_cols: list[np.ndarray] = []
    if n_cols == 0:
        return mel_cols, cent_cols, band_cols, 0

    consumed = n_cols * pool
    mel_view = mel_frames[:, :consumed].reshape(N_MELS, n_cols, pool)
    cent_view = cent_frames[:consumed].reshape(n_cols, pool)
    band_view = band_frames[:, :consumed].reshape(3, n_cols, pool)

    mel_cols.append(mel_view.max(axis=2))          # (N_MELS, n_cols)
    cent_cols.append(cent_view.mean(axis=1))       # (n_cols,)
    band_cols.append(band_view.sum(axis=2))        # (3, n_cols)
    return mel_cols, cent_cols, band_cols, consumed


def _pool_remainder(
    mel_frames: np.ndarray,
    cent_frames: np.ndarray,
    band_frames: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Pool a final partial frame run (< POOL frames) into one column."""
    if mel_frames.shape[1] == 0:
        return None
    mel_col = mel_frames.max(axis=1, keepdims=True)        # (N_MELS, 1)
    cent_col = np.atleast_1d(cent_frames.mean())           # (1,)
    band_col = band_frames.sum(axis=1, keepdims=True)      # (3, 1)
    return mel_col, cent_col, band_col


def build_spectral_proxy(
    src: str | os.PathLike,
    cache_root: str | os.PathLike,
    *,
    progress_cb: Callable[[int], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> spectral_cache.SpectralHeader:
    """Stream ``src`` → spectral proxy under ``cache_root`` (R-1 / R-2).

    Computes one block-aligned STFT per streamed block (``center=False`` with a
    retained ``N_FFT - HOP`` leftover so frames never straddle a block seam),
    derives mel (power), spectral centroid (Hz), and low/mid/high band energies
    from the SAME magnitude STFT, MAX/mean/sum-pools them to ~1 column/s inside
    the loop, and writes the three lanes atomically via
    :func:`spectral_cache.write_spectral`.

    Args:
        src: Source audio path (canonical 48 kHz mono float32 when this runs).
        cache_root: Cache root; output lands under ``<cache_root>/spectra/<key>/``.
        progress_cb: Optional ``progress_cb(pct)`` fired only on strict integer
            increase (0..100 inclusive, monotone) — peak_builder contract.
        cancel_check: Optional ``cancel_check()`` polled at the top of each
            block; ``True`` raises :class:`BuildCancelled` leaving no partial.

    Returns:
        The :class:`spectral_cache.SpectralHeader` parsed back from the freshly
        written ``mel.dat``.

    Raises:
        BuildCancelled: when ``cancel_check()`` returns True between blocks.
    """
    src_p = Path(src)
    key = proxy_cache.cache_key(src_p)
    total_frames = probe(src_p).frames

    # Leftover SAMPLE buffer (retains the N_FFT-HOP overlap across blocks) and
    # the pre-pool FRAME carries (kept < POOL so RSS stays bounded — Pitfall #2).
    buffer = np.empty(0, dtype=np.float32)
    mel_carry = np.empty((N_MELS, 0), dtype=np.float64)
    cent_carry = np.empty(0, dtype=np.float64)
    band_carry = np.empty((3, 0), dtype=np.float64)

    mel_chunks: list[np.ndarray] = []
    cent_chunks: list[np.ndarray] = []
    band_chunks: list[np.ndarray] = []

    decoded = 0
    last_pct = -1

    try:
        for block, _off in iter_blocks(src_p, mono=True):
            # Poll cancel at the TOP, before any work for this block.
            if cancel_check is not None and cancel_check():
                raise BuildCancelled()

            if buffer.size == 0:
                buffer = np.ascontiguousarray(block, dtype=np.float32)
            else:
                buffer = np.concatenate([buffer, block])

            # Whole frames fully contained in the buffer (center=False).
            if buffer.size >= N_FFT:
                n_frames = 1 + (buffer.size - N_FFT) // HOP
            else:
                n_frames = 0

            if n_frames > 0:
                consumed = n_frames * HOP
                # Pattern 1 seam arithmetic: feed exactly the samples those
                # n_frames cover (consumed + the N_FFT-HOP window tail) so the
                # next block continues from the retained 1024-sample overlap.
                window_len = consumed + (N_FFT - HOP)
                S = np.abs(
                    librosa.stft(
                        buffer[:window_len],
                        n_fft=N_FFT,
                        hop_length=HOP,
                        center=False,
                    )
                )  # (1025, n_frames)

                power = S.astype(np.float64) ** 2
                mel = _MEL_BASIS @ power                       # (N_MELS, n_frames)
                denom = np.maximum(S.sum(axis=0), 1e-9)
                centroid = (_FFT_FREQS[:, None] * S).sum(axis=0) / denom  # (n_frames,)
                bands = np.stack(
                    [
                        power[_LO_MASK].sum(axis=0),
                        power[_MID_MASK].sum(axis=0),
                        power[_HI_MASK].sum(axis=0),
                    ]
                )  # (3, n_frames)

                # Retain the N_FFT-HOP overlap for the next block.
                buffer = buffer[consumed:].copy()

                # Append to the carry, then pool whole columns out immediately.
                mel_carry = np.concatenate([mel_carry, mel], axis=1)
                cent_carry = np.concatenate([cent_carry, centroid])
                band_carry = np.concatenate([band_carry, bands], axis=1)

                mc, cc, bc, used = _pool_columns(
                    mel_carry, cent_carry, band_carry, POOL
                )
                if used > 0:
                    mel_chunks.extend(mc)
                    cent_chunks.extend(cc)
                    band_chunks.extend(bc)
                    mel_carry = mel_carry[:, used:].copy()
                    cent_carry = cent_carry[used:].copy()
                    band_carry = band_carry[:, used:].copy()

            decoded += int(block.size)
            if total_frames > 0:
                pct = int(100 * decoded / total_frames)
                if pct > last_pct:
                    last_pct = pct
                    if progress_cb is not None:
                        progress_cb(pct)

        # Flush any partial-POOL carry as a final column (don't drop the tail).
        rem = _pool_remainder(mel_carry, cent_carry, band_carry)
        if rem is not None:
            mel_chunks.append(rem[0])
            cent_chunks.append(rem[1])
            band_chunks.append(rem[2])

        # Final 100% notification (peak_builder:184-186 guard).
        if progress_cb is not None and last_pct < 100:
            progress_cb(100)

        if mel_chunks:
            mel_mag = np.concatenate(mel_chunks, axis=1)       # (N_MELS, n_cols)
            centroid_f32 = np.concatenate(cent_chunks).astype(np.float32)
            bands_f32 = np.concatenate(band_chunks, axis=1).astype(np.float32)
        else:
            mel_mag = np.zeros((N_MELS, 1), dtype=np.float64)
            centroid_f32 = np.zeros(1, dtype=np.float32)
            bands_f32 = np.zeros((3, 1), dtype=np.float32)

        # Normalise mel magnitudes into [0, 1] over the fixed dB window upstream
        # (the cache's write_spectral stores a LINEAR uint8 of the [0,1] input).
        mel_u8 = spectral_cache.quantize_mel_db(
            np.sqrt(mel_mag),  # power → magnitude for the dB conversion
            db_floor=spectral_cache.DB_FLOOR,
            db_ref=spectral_cache.DB_REF,
        )
        mel_norm = mel_u8.astype(np.float64) / 255.0

        spectral_cache.write_spectral(
            cache_root,
            key,
            sample_rate=SR,
            mel=mel_norm,
            centroid=centroid_f32,
            bands=bands_f32,
            hop_length=HOP,
            n_fft=N_FFT,
            db_floor=spectral_cache.DB_FLOOR,
            db_ref=spectral_cache.DB_REF,
        )
    except BuildCancelled:
        # write_spectral is atomic, but if a tmp survived a partial write,
        # unlink it defensively before re-raising (mirror peak_builder:199-207).
        spectra_dir = Path(cache_root) / "spectra" / key
        if spectra_dir.is_dir():
            for tmp in spectra_dir.glob("*.dat.tmp"):
                try:
                    os.remove(str(tmp))
                except FileNotFoundError:
                    pass
        raise

    mel_path = spectral_cache.spectral_path(cache_root, key, "mel")
    _, header = spectral_cache.load_mel(mel_path)
    return header
