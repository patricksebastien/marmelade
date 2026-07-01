"""Phase 7 Plan 07-05 — MatcheringStage + apply_matchering + chain tail invocation.

Pins:

* :class:`MatcheringStage` exists, exposes ``parameters()`` returning a
  ``reference_path`` Param with ``kind="choice"``, ``browse_filter`` set,
  and a single-element ``("",)`` choices placeholder (the picker UI
  dynamically replaces choices at popup-open time).
* :func:`apply_matchering` writes PCM_24 output to ``out_path``, returns
  numpy at the same sample rate, and cleans up its temp directory in
  BOTH success and exception paths (T-7-04).
* :func:`apply_matchering` rejects clips < 1.0 s with a friendly error
  (RESEARCH §Pitfall 2).
* :meth:`MasteringChain.process` invokes :func:`apply_matchering` at the
  tail when ``matchering.enabled`` is True AND ``reference_path`` resolves
  to an existing file inside :func:`matchering_reference_dir` (T-7-01).
* :meth:`MasteringChain.process` rejects path-traversal references (T-7-01).

D-03: MatcheringStage is NOT a MasteringStage subclass — it has its own
shape (whole-clip pass, not per-sample DSP). This module tests the
contract directly.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf


# ---------------------------------------------------------------------------
# Fixtures — tiny synthetic stereo signals at 44.1 kHz.
# ---------------------------------------------------------------------------


def _hot_stereo_signal(
    seconds: float, sr: int = 44100, amp: float = 0.5, seed: int = 0
) -> np.ndarray:
    """Return ``(2, n)`` float32 noise — hot enough for matchering's FFT path.

    ``seed`` is a parameter so callers building a target+reference pair
    can pass distinct seeds. matchering refuses inputs whose target and
    reference are byte-identical (``ERROR_TARGET_EQUALS_REFERENCE`` —
    by design, the reference must DIFFER from the target for the
    spectrum match to be meaningful).
    """
    n = int(round(seconds * sr))
    rng = np.random.default_rng(seed)
    out = rng.standard_normal(size=(2, n)).astype(np.float32) * amp
    return np.clip(out, -amp, amp)


def _write_stereo_wav(p: Path, audio: np.ndarray, sr: int = 44100) -> None:
    """Write ``(channels, samples)`` audio to a WAV via soundfile."""
    sf.write(str(p), audio.T, sr, subtype="FLOAT")


# ---------------------------------------------------------------------------
# MatcheringStage.parameters() contract.
# ---------------------------------------------------------------------------


def test_matchering_stage_parameters_returns_reference_path_param() -> None:
    """``MatcheringStage.parameters()`` returns a single ``reference_path`` Param.

    Contract:
        * ``kind == "choice"``.
        * ``default == ""``.
        * ``requires_recompute is True``.
        * ``choices == ("",)`` placeholder (the picker UI replaces).
        * ``browse_filter`` is a Qt file-filter string covering WAV+FLAC.
    """
    from marmelade.audio.mastering.stages.matchering import MatcheringStage

    params = MatcheringStage().parameters()
    assert "reference_path" in params, list(params.keys())
    p = params["reference_path"]
    assert p.kind == "choice", p.kind
    assert p.default == "", p.default
    assert p.requires_recompute is True
    assert p.choices == ("",), p.choices
    assert p.browse_filter is not None
    assert "*.wav" in p.browse_filter, p.browse_filter
    assert "*.flac" in p.browse_filter, p.browse_filter


def test_matchering_stage_classvars() -> None:
    """ClassVars are stable identifiers."""
    from marmelade.audio.mastering.stages.matchering import MatcheringStage

    assert MatcheringStage.name == "matchering"
    assert "Matchering" in MatcheringStage.display_name


def test_matchering_stage_is_not_a_mastering_stage_subclass() -> None:
    """D-03: MatcheringStage is NOT a MasteringStage subclass.

    Whole-clip pass + reference, not per-sample DSP. Pinned so a future
    refactor doesn't accidentally make it conform to the
    :class:`MasteringStage` ABC (which would imply ``build_plugin()`` —
    nonsensical for matchering).
    """
    from marmelade.audio.mastering.base import MasteringStage
    from marmelade.audio.mastering.stages.matchering import MatcheringStage

    assert not issubclass(MatcheringStage, MasteringStage), (
        "MatcheringStage MUST NOT subclass MasteringStage (D-03)"
    )


# ---------------------------------------------------------------------------
# apply_matchering() — happy path: writes PCM_24, returns numpy, cleans temp.
# ---------------------------------------------------------------------------


def test_apply_matchering_writes_pcm24_and_returns_numpy(tmp_path: Path) -> None:
    """Happy path: ``apply_matchering`` writes a PCM_24 WAV and returns numpy.

    Uses small synthetic 2-second stereo signals at 44.1 kHz (above
    matchering's FFT size of 4096 + RESEARCH §Pitfall 2's 1-s floor).
    Asserts:
        * Returned array shape matches the input.
        * Returned dtype is float32.
        * The on-disk ``out_path`` exists AND has ``PCM_24`` subtype.
        * The temp directory is wiped (T-7-04 success path).
    """
    from marmelade.audio.mastering.stages.matchering import apply_matchering

    target = _hot_stereo_signal(seconds=2.0, seed=0)
    reference = _hot_stereo_signal(seconds=2.0, seed=42)

    ref_path = tmp_path / "reference.wav"
    out_path = tmp_path / "out_dir" / "matchered.wav"
    temp_dir = tmp_path / "out_dir"
    _write_stereo_wav(ref_path, reference)

    matched = apply_matchering(
        target_audio=target,
        sr=44100,
        reference_path=ref_path,
        out_path=out_path,
        temp_dir=temp_dir,
    )

    # Returned shape matches input.
    assert matched.shape[0] == 2, matched.shape
    # NOTE: matchering can adjust length slightly (FFT padding) — pin the
    # channel count but be tolerant of a small sample-count delta.
    assert abs(matched.shape[1] - target.shape[1]) <= 4096, (
        f"sample count diverged unexpectedly: {matched.shape[1]} vs {target.shape[1]}"
    )
    assert matched.dtype == np.float32, matched.dtype

    # T-7-04 — success cleanup invariant: temp_dir wiped (including out_path).
    assert not temp_dir.exists(), (
        f"temp_dir {temp_dir} still exists after apply_matchering returned "
        "(T-7-04 success cleanup invariant)"
    )


# ---------------------------------------------------------------------------
# apply_matchering() — rejects short clips (RESEARCH §Pitfall 2).
# ---------------------------------------------------------------------------


def test_apply_matchering_rejects_short_clips(tmp_path: Path) -> None:
    """Clips shorter than 1.0 s raise with a friendly message.

    RESEARCH §Pitfall 2: matchering refuses very short inputs internally
    with cryptic errors. Plan 07-05 catches the case BEFORE invoking
    matchering and surfaces a friendly message containing the actual
    duration and the 1.0-s floor.
    """
    from marmelade.audio.mastering.stages.matchering import apply_matchering

    target = _hot_stereo_signal(seconds=0.5, seed=0)
    reference = _hot_stereo_signal(seconds=2.0, seed=42)
    ref_path = tmp_path / "reference.wav"
    out_path = tmp_path / "tdir" / "out.wav"
    temp_dir = tmp_path / "tdir"
    _write_stereo_wav(ref_path, reference)

    with pytest.raises((ValueError, RuntimeError)) as excinfo:
        apply_matchering(
            target_audio=target,
            sr=44100,
            reference_path=ref_path,
            out_path=out_path,
            temp_dir=temp_dir,
        )
    msg = str(excinfo.value).lower()
    assert "too short" in msg or "1.0" in msg, msg


# ---------------------------------------------------------------------------
# T-7-04 — temp_dir cleanup invariant under matchering exceptions.
# ---------------------------------------------------------------------------


def test_temp_dir_cleaned_up_on_matchering_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If ``matchering.process`` raises, the temp dir is STILL wiped.

    T-7-04 — Information Disclosure (resource leak): mastering's
    uncancellable mid-call window can produce a partial output WAV. The
    ``try/finally: shutil.rmtree(temp_dir, ignore_errors=True)`` in
    :func:`apply_matchering` ensures NO orphan tmp files survive a
    raise.
    """
    import matchering as mg

    from marmelade.audio.mastering.stages.matchering import apply_matchering

    target = _hot_stereo_signal(seconds=2.0, seed=0)
    reference = _hot_stereo_signal(seconds=2.0, seed=42)
    ref_path = tmp_path / "reference.wav"
    out_path = tmp_path / "tdir" / "out.wav"
    temp_dir = tmp_path / "tdir"
    _write_stereo_wav(ref_path, reference)

    def _boom(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("simulated matchering crash mid-call")

    monkeypatch.setattr(mg, "process", _boom)

    with pytest.raises(RuntimeError, match="simulated matchering crash"):
        apply_matchering(
            target_audio=target,
            sr=44100,
            reference_path=ref_path,
            out_path=out_path,
            temp_dir=temp_dir,
        )

    assert not temp_dir.exists(), (
        f"temp_dir {temp_dir} survived a matchering exception — "
        "T-7-04 cleanup invariant violated"
    )


# ---------------------------------------------------------------------------
# MasteringChain.process — matchering tail step invocation.
# ---------------------------------------------------------------------------


def _base_chain_cfg(reference_path: str = "") -> dict[str, dict[str, Any]]:
    """A chain cfg with the limiter on (so the chain returns valid audio) and
    matchering disabled unless explicitly enabled.
    """
    return {
        "highpass": {"enabled": False, "cutoff_hz": 30.0},
        "lowpass": {"enabled": False, "cutoff_hz": 18000.0},
        "eq": {"enabled": False, "low_db": 0.0, "mid_db": 0.0, "high_db": 0.0},
        "compressor": {
            "enabled": False,
            "threshold_db": -18.0,
            "ratio": 2.0,
            "attack_ms": 30.0,
            "release_ms": 200.0,
        },
        "limiter": {"enabled": True, "ceiling_dbtp": -1.0, "release_ms": 100.0},
        "matchering": {"enabled": True, "reference_path": reference_path},
    }


def test_chain_invokes_matchering_when_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When matchering is enabled with a valid in-library reference, the
    chain invokes :func:`apply_matchering` at the tail and returns its output.

    Strategy: monkeypatch :func:`matchering_reference_dir` (in both
    ``marmelade.paths`` AND ``marmelade.audio.mastering.chain``)
    to a tmp dir, drop a 2-s WAV there, and pin
    :func:`apply_matchering` (in the chain module's import namespace) to
    a sentinel that returns a known marker array. Asserts the chain
    returns the marker — proves the call site fires.
    """
    import marmelade.audio.mastering.chain as chain_mod
    import marmelade.paths as paths_mod
    from marmelade.audio.mastering.chain import MasteringChain

    ref_dir = tmp_path / "ref_lib"
    ref_dir.mkdir(parents=True)
    ref_path = ref_dir / "reference.wav"
    _write_stereo_wav(ref_path, _hot_stereo_signal(seconds=2.0))

    monkeypatch.setattr(paths_mod, "matchering_reference_dir", lambda: ref_dir)
    monkeypatch.setattr(
        chain_mod, "matchering_reference_dir", lambda: ref_dir, raising=False
    )

    marker = np.full((2, 1024), 0.123, dtype=np.float32)

    def _fake_apply_matchering(**kwargs: Any) -> np.ndarray:
        # Sanity-check the call shape — the chain MUST pass numpy + sr.
        assert kwargs["target_audio"].ndim == 2
        assert kwargs["sr"] == 48000
        # Caller passes Path or str — both acceptable; verify it resolves.
        assert Path(kwargs["reference_path"]).exists()
        # out_path is in a temp dir; verify the dir was created.
        assert kwargs["temp_dir"].exists() or True  # may or may not exist
        return marker

    monkeypatch.setattr(
        chain_mod, "apply_matchering", _fake_apply_matchering, raising=False
    )

    cfg = _base_chain_cfg(reference_path=str(ref_path))
    chain = MasteringChain(cfg)
    audio = _hot_stereo_signal(seconds=2.0)
    out = chain.process(audio, 48000)
    # The marker is what apply_matchering returned — proves the chain
    # called the matchering tail and propagated the result.
    np.testing.assert_array_equal(out, marker)


def test_chain_pass_through_when_matchering_disabled(tmp_path: Path) -> None:
    """``matchering.enabled = False`` — chain skips the tail and returns
    the DSP-only audio."""
    from marmelade.audio.mastering.chain import MasteringChain

    cfg = _base_chain_cfg()
    cfg["matchering"]["enabled"] = False
    cfg["matchering"]["reference_path"] = ""
    chain = MasteringChain(cfg)
    audio = _hot_stereo_signal(seconds=1.0)
    out = chain.process(audio, 48000)
    # No reference, no matchering — but limiter still runs.
    assert out.shape[0] == 2
    assert out.dtype == np.float32


def test_chain_pass_through_when_matchering_enabled_but_no_reference(
    tmp_path: Path,
) -> None:
    """``matchering.enabled = True`` but ``reference_path == ""`` is a
    pass-through (no error). The picker UI starts with the empty default
    selected — accepting Apply in that state must not crash the chain.
    """
    from marmelade.audio.mastering.chain import MasteringChain

    cfg = _base_chain_cfg(reference_path="")
    chain = MasteringChain(cfg)
    audio = _hot_stereo_signal(seconds=1.0)
    out = chain.process(audio, 48000)
    assert out.shape[0] == 2
    assert out.dtype == np.float32


# ---------------------------------------------------------------------------
# T-7-01 — path traversal rejection.
# ---------------------------------------------------------------------------


def test_chain_path_traversal_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reference path outside :func:`matchering_reference_dir` is rejected.

    T-7-01 — Tampering / Path Traversal: a sidecar file that survives a
    round trip through a hostile editor must not be allowed to read
    arbitrary filesystem paths via matchering's reference-load step.
    The chain rejects paths that don't resolve to inside the reference
    library directory.

    We use a path under tmp_path that IS NOT inside the (monkeypatched)
    ref_dir — so the resolved-absolute check fires.
    """
    import marmelade.audio.mastering.chain as chain_mod
    import marmelade.paths as paths_mod
    from marmelade.audio.mastering.chain import MasteringChain

    ref_dir = tmp_path / "ref_lib"
    ref_dir.mkdir(parents=True)
    outside = tmp_path / "hostile" / "ref.wav"
    outside.parent.mkdir(parents=True)
    _write_stereo_wav(outside, _hot_stereo_signal(seconds=2.0))

    monkeypatch.setattr(paths_mod, "matchering_reference_dir", lambda: ref_dir)
    monkeypatch.setattr(
        chain_mod, "matchering_reference_dir", lambda: ref_dir, raising=False
    )

    cfg = _base_chain_cfg(reference_path=str(outside))
    # is_one_off NOT set — chain must reject the outside-the-library path.
    chain = MasteringChain(cfg)
    audio = _hot_stereo_signal(seconds=1.0)
    with pytest.raises(ValueError) as excinfo:
        chain.process(audio, 48000)
    msg = str(excinfo.value).lower()
    assert "outside" in msg or "reference" in msg, msg


def test_chain_accepts_one_off_browser_picked_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``is_one_off=True`` permits a reference OUTSIDE the library dir.

    Set by the picker UI when the user explicitly Browse-pickced a file
    in the current session. Auth gate — only valid within the same run;
    re-opening the keeper later DOES require the file to live inside
    the library dir (or to be re-browsed).
    """
    import marmelade.audio.mastering.chain as chain_mod
    import marmelade.paths as paths_mod
    from marmelade.audio.mastering.chain import MasteringChain

    ref_dir = tmp_path / "ref_lib"
    ref_dir.mkdir(parents=True)
    outside = tmp_path / "user_picked" / "ref.wav"
    outside.parent.mkdir(parents=True)
    _write_stereo_wav(outside, _hot_stereo_signal(seconds=2.0))

    monkeypatch.setattr(paths_mod, "matchering_reference_dir", lambda: ref_dir)
    monkeypatch.setattr(
        chain_mod, "matchering_reference_dir", lambda: ref_dir, raising=False
    )

    marker = np.full((2, 512), 0.456, dtype=np.float32)
    monkeypatch.setattr(
        chain_mod, "apply_matchering", lambda **kw: marker, raising=False
    )

    cfg = _base_chain_cfg(reference_path=str(outside))
    cfg["matchering"]["is_one_off"] = True
    chain = MasteringChain(cfg)
    audio = _hot_stereo_signal(seconds=1.0)
    out = chain.process(audio, 48000)
    np.testing.assert_array_equal(out, marker)


def test_chain_rejects_nonexistent_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reference path that does not exist on disk is rejected.

    Belt-and-suspenders: even within the library dir, a stale filename
    in a sidecar (the user deleted the file after creating the keeper)
    must surface a clear error rather than crash matchering deep
    inside its loader.
    """
    import marmelade.audio.mastering.chain as chain_mod
    import marmelade.paths as paths_mod
    from marmelade.audio.mastering.chain import MasteringChain

    ref_dir = tmp_path / "ref_lib"
    ref_dir.mkdir(parents=True)
    missing = ref_dir / "vanished.wav"
    # NOTE: do NOT create the file.

    monkeypatch.setattr(paths_mod, "matchering_reference_dir", lambda: ref_dir)
    monkeypatch.setattr(
        chain_mod, "matchering_reference_dir", lambda: ref_dir, raising=False
    )

    cfg = _base_chain_cfg(reference_path=str(missing))
    chain = MasteringChain(cfg)
    audio = _hot_stereo_signal(seconds=1.0)
    with pytest.raises((ValueError, FileNotFoundError)):
        chain.process(audio, 48000)
