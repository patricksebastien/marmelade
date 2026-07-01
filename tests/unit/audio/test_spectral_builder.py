"""RED scaffold — R-1 spectral builder (bounded RSS / seam continuity / cancel).

Phase 11 Wave 0 (plan 11-01). These tests PIN the acceptance criteria for the
not-yet-existing :func:`marmelade.audio.spectral_builder.build_spectral_proxy`.
They are EXPECTED TO FAIL until plan 11-02 lands the builder — that is the
success condition for this scaffold (test-first, gb7 precedent).

Design rule (must collect cleanly): the production import lives INSIDE each test
function. Collection therefore never raises ImportError on the test module; the
test instead fails on a clean ImportError/AttributeError (RED) once invoked, and
turns GREEN when the symbol lands.

R-1 acceptance criteria covered:
    * test_bounded_rss          — peak RSS does not scale with input duration.
    * test_block_seam_continuity — a sine straddling a 131072-sample block
      boundary produces no spurious mel-column spike at the seam frame.
    * test_cancel_leaves_no_partial — cancel mid-build raises BuildCancelled and
      leaves neither mel.dat nor mel.dat.tmp behind (atomic-write contract).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# tests/ on sys.path so `from fixtures.synthesize import ...` resolves the same
# way the rest of the suite imports synthetic fixtures.
_TESTS_ROOT = Path(__file__).resolve().parents[2]
if str(_TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_TESTS_ROOT))

from fixtures.synthesize import make_sine  # noqa: E402

# Canonical rate + the audio_file block size the builder iterates in.
SR = 48000
BLOCK_SAMPLES = 131_072  # marmelade.audio.audio_file.BLOCK_SAMPLES (2**17)


def _build_spectral_proxy():
    """Import the (later-wave) builder entry point. Raises ImportError now."""
    from marmelade.audio.spectral_builder import build_spectral_proxy

    return build_spectral_proxy


def _peak_rss_bytes() -> int:
    """Peak resident set size in bytes (resource.getrusage, Linux ru_maxrss=KiB)."""
    import resource

    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


@pytest.mark.slow
def test_bounded_rss(tmp_path: Path) -> None:
    """R-1: building a 2x-longer file must not roughly double peak RSS.

    The builder must stream the source in blocks (CLAUDE.md memory contract —
    never load the whole 8 h WAV). We build a short fixture and a 4x-longer
    fixture and assert the incremental RSS growth across the long build is a
    small fraction of the fixture's on-disk size, i.e. RSS is bounded by the
    block working set, not the input length.
    """
    build_spectral_proxy = _build_spectral_proxy()

    short = make_sine(
        tmp_path / "short.wav", freq_hz=440.0, duration_s=4.0, sample_rate=SR
    )
    long = make_sine(
        tmp_path / "long.wav", freq_hz=440.0, duration_s=16.0, sample_rate=SR
    )

    short_dir = tmp_path / "short_cache"
    short_dir.mkdir()
    build_spectral_proxy(str(short), str(short_dir))

    before = _peak_rss_bytes()
    long_dir = tmp_path / "long_cache"
    long_dir.mkdir()
    build_spectral_proxy(str(long), str(long_dir))
    after = _peak_rss_bytes()

    growth = max(0, after - before)
    long_file_bytes = Path(long).stat().st_size
    # If RSS scaled with duration, growth would be on the order of the file
    # size. Bounded streaming keeps it well under that (block working set only).
    assert growth < long_file_bytes, (
        f"peak RSS grew {growth} B building a {long_file_bytes} B file — "
        "builder appears to load the whole file (R-1 bounded-RSS violation)"
    )


def test_block_seam_continuity(tmp_path: Path) -> None:
    """R-1: a steady sine spanning a BLOCK_SAMPLES boundary has no seam spike.

    The builder accumulates STFT frames across streamed blocks; if a frame is
    dropped or duplicated at the 131072-sample block seam the mel column at that
    frame index spikes (or dips) relative to its neighbours. A pure 1 kHz sine
    should produce a near-constant mel column over time, so we assert the column
    at the seam frame is within epsilon of the surrounding columns' median.
    """
    build_spectral_proxy = _build_spectral_proxy()
    from marmelade.audio import spectral_cache

    # ~3.5 blocks long so the build crosses at least two block seams at 48 kHz.
    duration_s = (BLOCK_SAMPLES * 3.5) / SR
    src = make_sine(
        tmp_path / "seam.wav", freq_hz=1000.0, duration_s=duration_s, sample_rate=SR
    )
    dst = tmp_path / "seam_cache"
    dst.mkdir()
    build_spectral_proxy(str(src), str(dst))

    mel, header = spectral_cache.load_mel(
        spectral_cache.spectral_path(str(dst), _key_for(src), "mel")
    )
    mel = np.asarray(mel, dtype=np.float64)
    # mel is (n_mels, n_frames) row-major per Pitfall #4. Per-frame energy.
    per_frame = mel.sum(axis=0)
    n_frames = per_frame.shape[0]
    assert n_frames > 8, "need several frames to test seam continuity"

    # The seam frames correspond to BLOCK_SAMPLES boundaries. Without coupling
    # to the exact hop, assert NO single interior frame deviates wildly from the
    # global median (a dropped/duplicated frame at a seam shows up as an outlier).
    median = float(np.median(per_frame))
    spread = float(np.median(np.abs(per_frame - median))) + 1e-9
    worst = float(np.max(np.abs(per_frame[1:-1] - median)))
    assert worst < 8.0 * spread, (
        f"interior mel frame deviates {worst:.3g} from median {median:.3g} "
        f"(spread {spread:.3g}) — block-seam discontinuity (R-1)"
    )


def test_cancel_leaves_no_partial(tmp_path: Path) -> None:
    """R-1: cancel mid-build raises BuildCancelled and leaves no .dat/.tmp.

    Reuses the existing :class:`marmelade.audio.peak_builder.BuildCancelled`
    (D-16 — the cancel exception is shared across builders). cancel_check
    returns True after the first poll, so the build aborts before completing;
    the atomic-write contract requires that neither the final mel.dat nor any
    mel.dat.tmp sibling remains in the destination directory.
    """
    build_spectral_proxy = _build_spectral_proxy()
    from marmelade.audio.peak_builder import BuildCancelled

    src = make_sine(
        tmp_path / "cancel.wav", freq_hz=440.0, duration_s=8.0, sample_rate=SR
    )
    dst = tmp_path / "cancel_cache"
    dst.mkdir()

    polls = {"n": 0}

    def cancel_check() -> bool:
        polls["n"] += 1
        return polls["n"] >= 1  # cancel at the first between-block poll

    with pytest.raises(BuildCancelled):
        build_spectral_proxy(str(src), str(dst), cancel_check=cancel_check)

    leftover = [p.name for p in dst.rglob("*") if p.is_file()]
    assert not any(name.endswith(".dat") for name in leftover), (
        f"cancelled build left a .dat: {leftover}"
    )
    assert not any(name.endswith(".tmp") for name in leftover), (
        f"cancelled build left a .tmp: {leftover}"
    )


def _key_for(src_path) -> str:
    """Resolve the cache key the builder uses for ``src_path``.

    The builder keys spectral output by the same source-fingerprint scheme as
    the proxy/heatmap caches. This helper defers to the production keying
    function so the test does not hard-code the digest algorithm.
    """
    from marmelade.audio.proxy_cache import cache_key

    return cache_key(Path(src_path))
