"""Unit tests for the fade-ramp helpers in :mod:`marmelade.audio.export_builder`.

W-7 invariant: ramps use ``np.linspace(endpoint=True)`` so:
    * first fade-in sample is exactly 0.0
    * LAST fade-in sample is exactly 1.0
    * first fade-out sample is exactly 1.0
    * LAST fade-out sample is exactly 0.0

The block layout helpers (pedalboard ``(n_channels, n_frames)`` vs
soundfile ``(n_frames, n_channels)``) share the same global-index
arithmetic — these tests pin both shapes.
"""

from __future__ import annotations

import numpy as np
import pytest

from marmelade.audio.export_builder import (
    _apply_fade_pedalboard_layout,
    _apply_fade_soundfile_layout,
)
from marmelade.paths import default_cache_root  # noqa: F401 — conftest patch target


SR = 44100


def test_apply_fade_pedalboard_layout_full_2s_fade() -> None:
    """All-in-one block: 10s constant-1.0 with 2s linear fade head + tail."""
    total_frames = SR * 10
    fade_frames = SR * 2
    block = np.ones((2, total_frames), dtype=np.float32)

    _apply_fade_pedalboard_layout(block, 0, total_frames, fade_frames)

    # First fade-in sample is exactly 0.0
    assert block[0, 0] == pytest.approx(0.0)
    assert block[1, 0] == pytest.approx(0.0)
    # LAST fade-in sample is exactly 1.0 (W-7 endpoint=True)
    assert block[0, fade_frames - 1] == pytest.approx(1.0)
    # First sample after fade-in is 1.0 (unscaled)
    assert block[0, fade_frames] == pytest.approx(1.0)
    # First fade-out sample is exactly 1.0 (W-7 endpoint=True at start of out-ramp)
    assert block[0, total_frames - fade_frames] == pytest.approx(1.0)
    # LAST fade-out sample is exactly 0.0 (W-7 endpoint=True at end)
    assert block[0, total_frames - 1] == pytest.approx(0.0)


def test_apply_fade_soundfile_layout_full_2s_fade() -> None:
    """Mirror test with soundfile layout (n_frames, n_channels)."""
    total_frames = SR * 10
    fade_frames = SR * 2
    block = np.ones((total_frames, 2), dtype=np.float32)

    _apply_fade_soundfile_layout(block, 0, total_frames, fade_frames)

    assert block[0, 0] == pytest.approx(0.0)
    assert block[fade_frames - 1, 0] == pytest.approx(1.0)
    assert block[fade_frames, 0] == pytest.approx(1.0)
    assert block[total_frames - fade_frames, 0] == pytest.approx(1.0)
    assert block[total_frames - 1, 0] == pytest.approx(0.0)


def test_apply_fade_short_region_auto_scaled() -> None:
    """1s region with fade=0.5s — no overlap between fade-in/out endpoints."""
    total_frames = SR
    fade_frames = SR // 2
    block = np.ones((2, total_frames), dtype=np.float32)

    _apply_fade_pedalboard_layout(block, 0, total_frames, fade_frames)

    # No overlap: fade-in ends at index fade_frames-1 (= total//2 - 1),
    # fade-out begins at total - fade_frames (= total//2). These are
    # adjacent indices; the sample at the meeting point on each side
    # should be exactly 1.0.
    assert block[0, fade_frames - 1] == pytest.approx(1.0)
    assert block[0, fade_frames] == pytest.approx(1.0)


def test_apply_fade_zero_fade_is_noop() -> None:
    """fade_frames=0 leaves the block bit-identical to the input."""
    total_frames = 1024
    block = np.ones((2, total_frames), dtype=np.float32) * 0.7
    expected = block.copy()

    _apply_fade_pedalboard_layout(block, 0, total_frames, 0)

    np.testing.assert_array_equal(block, expected)


def test_apply_fade_block_boundary_correctness() -> None:
    """Block spans the fade boundary — first fade_frames ramp 0→1, rest are 1."""
    fade_frames = 100
    total_frames = 1000
    n = fade_frames + 10  # block crosses the fade-in boundary by 10
    block = np.ones((2, n), dtype=np.float32)

    _apply_fade_pedalboard_layout(block, 0, total_frames, fade_frames)

    # First sample 0.0, last fade sample 1.0, post-fade samples all 1.0.
    assert block[0, 0] == pytest.approx(0.0)
    assert block[0, fade_frames - 1] == pytest.approx(1.0)
    # Ramp is monotonically non-decreasing across the fade-in window
    assert np.all(np.diff(block[0, :fade_frames]) >= -1e-7)
    # Tail of the block (post fade-in) is 1.0
    np.testing.assert_allclose(block[0, fade_frames:n], 1.0, atol=1e-7)


def test_apply_fade_block_outside_fade_window() -> None:
    """Block entirely between fade-in and fade-out → bit-identical to input."""
    fade_frames = 100
    total_frames = 10_000
    # Block of 200 frames starting at frame 1000 — well inside the
    # central plateau.
    block_start = 1000
    n = 200
    block = np.ones((2, n), dtype=np.float32) * 0.5
    expected = block.copy()

    _apply_fade_pedalboard_layout(block, block_start, total_frames, fade_frames)

    np.testing.assert_array_equal(block, expected)


def test_apply_fade_fade_frames_equals_one_does_not_div_by_zero() -> None:
    """fade_frames=1 → degenerate ramp.

    CR-02: the lone sample of a 1-sample fade is BOTH the first AND the
    last sample of the fade window, so under the W-7 endpoint=True
    invariant it must take the boundary value 0.0 (the fade-in
    multiplier the linspace would emit at endpoint index 0, and the
    fade-out multiplier the linspace would emit at endpoint index
    fade_frames-1 == 0). The pre-CR-02 implementation substituted
    np.ones (a no-op multiplier) — convenient against divide-by-zero
    but semantically wrong. This test pins the corrected boundary
    semantic. The caller's fade auto-scale never produces fade_frames=1
    in practice, so the change is observable only in this unit test.
    """
    total_frames = 100
    block = np.ones((2, total_frames), dtype=np.float32)

    # Should not raise ZeroDivisionError / NaN.
    _apply_fade_pedalboard_layout(block, 0, total_frames, 1)

    # The single fade-in sample (index 0) and single fade-out sample
    # (index total_frames-1) become 0.0 under the corrected boundary
    # semantic. All other samples remain 1.0.
    assert not np.isnan(block).any()
    assert block[0, 0] == pytest.approx(0.0)
    assert block[0, total_frames - 1] == pytest.approx(0.0)
    # Interior samples (outside both fade windows) are untouched.
    np.testing.assert_allclose(block[0, 1:total_frames - 1], 1.0, atol=1e-7)


def test_apply_fade_pedalboard_monotonic_ramp() -> None:
    """Fade-in ramp is monotonically non-decreasing across the whole window."""
    total_frames = SR * 4
    fade_frames = SR
    block = np.ones((2, total_frames), dtype=np.float32)

    _apply_fade_pedalboard_layout(block, 0, total_frames, fade_frames)

    fade_in = block[0, :fade_frames]
    assert np.all(np.diff(fade_in) >= -1e-7), "fade-in not monotonically non-decreasing"
    fade_out = block[0, total_frames - fade_frames:]
    assert np.all(np.diff(fade_out) <= 1e-7), "fade-out not monotonically non-increasing"
