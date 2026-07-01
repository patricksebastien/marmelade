"""RED scaffold — R-2 spectral precompute budget (<=10 min/8 h, no torch).

Phase 11 Wave 0 (plan 11-01). PINs the R-2 performance contract for the
not-yet-existing :func:`marmelade.audio.spectral_builder.build_spectral_proxy`:

    * Build a short (60 s) 48 kHz fixture, time it, extrapolate LINEARLY to an
      8-hour file, and assert the extrapolated wall-clock is <= 600 s (10 min).
      No 8 h fixture exists in CI; the timed-short-fixture + documented linear
      extrapolation is the automated proxy for the manual 8 h verification.
    * Assert ``torch`` is NOT imported by the build (CPU/numpy STFT path only —
      the spectral lane must not pull in the heavy GPU stack for a simple mel).

Marked @pytest.mark.slow so `-m "not slow"` deselects it in the quick loop.
RED until plan 11-02 lands the builder.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

_TESTS_ROOT = Path(__file__).resolve().parents[1]
if str(_TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_TESTS_ROOT))

from fixtures.synthesize import make_sine  # noqa: E402

SR = 48000
FIXTURE_SECONDS = 60.0
EIGHT_HOURS_SECONDS = 8 * 60 * 60
BUDGET_SECONDS = 600.0  # 10 minutes per 8 h (R-2 / CLAUDE.md performance)


def _build_spectral_proxy():
    from marmelade.audio.spectral_builder import build_spectral_proxy

    return build_spectral_proxy


@pytest.mark.slow
def test_8h_budget_extrapolated_under_10min(tmp_path: Path) -> None:
    """R-2: 60 s build time extrapolated to 8 h must be <= 600 s."""
    build_spectral_proxy = _build_spectral_proxy()

    src = make_sine(
        tmp_path / "min.wav", freq_hz=440.0, duration_s=FIXTURE_SECONDS, sample_rate=SR
    )
    dst = tmp_path / "cache"
    dst.mkdir()

    t0 = time.perf_counter()
    build_spectral_proxy(str(src), str(dst))
    elapsed = time.perf_counter() - t0

    extrapolated = elapsed * (EIGHT_HOURS_SECONDS / FIXTURE_SECONDS)
    assert extrapolated <= BUDGET_SECONDS, (
        f"60 s build took {elapsed:.2f}s -> 8 h extrapolated {extrapolated:.0f}s "
        f"exceeds the {BUDGET_SECONDS:.0f}s (10 min) R-2 budget"
    )


@pytest.mark.slow
def test_build_does_not_import_torch(tmp_path: Path) -> None:
    """R-2: the spectral build path must not import torch (CPU/numpy STFT only)."""
    build_spectral_proxy = _build_spectral_proxy()

    # If torch is already resident from another test, this assertion can't prove
    # the builder avoided it — so require a clean slate first.
    assert "torch" not in sys.modules, (
        "torch already imported before the build — cannot attribute it; run this "
        "test in isolation"
    )

    src = make_sine(
        tmp_path / "notorch.wav", freq_hz=440.0, duration_s=2.0, sample_rate=SR
    )
    dst = tmp_path / "cache"
    dst.mkdir()
    build_spectral_proxy(str(src), str(dst))

    assert "torch" not in sys.modules, (
        "build_spectral_proxy imported torch — R-2 requires a CPU/numpy STFT path, "
        "not the GPU stack, for the mel/centroid/band precompute"
    )
