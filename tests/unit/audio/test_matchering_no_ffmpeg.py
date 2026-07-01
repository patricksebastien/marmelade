"""Phase 7 Plan 07-05 — matchering does NOT spawn ffmpeg for WAV references.

RESEARCH §Pitfall 3 — matchering's :func:`load` function calls
``subprocess.check_call(["ffmpeg", ...])`` if and only if
``soundfile.read`` raises ``RuntimeError`` with ``"unknown format"`` or
``"Format not recognised"`` in the message. WAV and FLAC files succeed
via ``soundfile.read`` directly and the ffmpeg fallback never fires —
which means the user does NOT need ffmpeg on PATH for the Phase 7
reference-picker workflow (the picker UI restricts choices to WAV +
FLAC via its ``browse_filter``).

This test pins the no-subprocess invariant by patching
``subprocess.check_call`` to set a flag, then running
:func:`apply_matchering` with a real WAV reference. The flag must
remain False post-call.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf


def _hot_stereo_signal(
    seconds: float = 2.0, sr: int = 44100, amp: float = 0.5, seed: int = 1
) -> np.ndarray:
    n = int(round(seconds * sr))
    rng = np.random.default_rng(seed)
    out = rng.standard_normal(size=(2, n)).astype(np.float32) * amp
    return np.clip(out, -amp, amp)


def _write_stereo_wav(p: Path, audio: np.ndarray, sr: int = 44100) -> None:
    sf.write(str(p), audio.T, sr, subtype="FLOAT")


def test_no_subprocess_for_wav_reference(
    tmp_path: Path, monkeypatch
) -> None:
    """``apply_matchering`` with a WAV reference must NOT spawn ffmpeg.

    RESEARCH §Pitfall 3. The picker UI restricts choices to WAV + FLAC
    so ffmpeg is never a runtime requirement; pin the invariant on the
    apply_matchering path so a regression here (e.g. someone removes
    the WAV-only restriction in the picker) is caught at the audio
    tier.
    """
    from marmelade.audio.mastering.stages.matchering import apply_matchering

    target = _hot_stereo_signal(seconds=2.0, seed=1)
    reference = _hot_stereo_signal(seconds=2.0, seed=99)

    ref_path = tmp_path / "reference.wav"
    out_path = tmp_path / "out_dir" / "matchered.wav"
    temp_dir = tmp_path / "out_dir"
    _write_stereo_wav(ref_path, reference)

    spawned = {"ffmpeg": False, "popen": False}

    original_check_call = subprocess.check_call
    original_popen = subprocess.Popen

    def _flag_check_call(cmd, *args, **kwargs):
        # Detect ffmpeg in cmd list.
        if isinstance(cmd, (list, tuple)) and cmd and "ffmpeg" in str(cmd[0]):
            spawned["ffmpeg"] = True
        return original_check_call(cmd, *args, **kwargs)

    class _FlagPopen(original_popen):  # type: ignore[misc]
        def __init__(self, cmd, *args, **kwargs):
            if isinstance(cmd, (list, tuple)) and cmd and "ffmpeg" in str(cmd[0]):
                spawned["popen"] = True
            elif isinstance(cmd, str) and "ffmpeg" in cmd:
                spawned["popen"] = True
            super().__init__(cmd, *args, **kwargs)

    monkeypatch.setattr(subprocess, "check_call", _flag_check_call)
    monkeypatch.setattr(subprocess, "Popen", _FlagPopen)

    apply_matchering(
        target_audio=target,
        sr=44100,
        reference_path=ref_path,
        out_path=out_path,
        temp_dir=temp_dir,
    )

    assert not spawned["ffmpeg"], (
        "ffmpeg subprocess was spawned for a WAV reference — Pitfall 3 regression"
    )
    assert not spawned["popen"], (
        "ffmpeg Popen was invoked for a WAV reference — Pitfall 3 regression"
    )
