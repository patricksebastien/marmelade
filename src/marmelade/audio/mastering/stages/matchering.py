"""Matchering reference-match tail step — D-03 + T-7-01 + T-7-04 + Pitfalls 2/3/5/7.

Phase 7 Plan 07-05 — Matchering integration. Owns MAS-02.

Public surface:

* :class:`MatcheringStage` — Qt-free; declares the single ``reference_path``
  Param that the picker UI populates dynamically. NOT a
  :class:`~marmelade.audio.mastering.base.MasteringStage` subclass
  (D-03: matchering is a whole-clip + reference pass, not per-sample DSP).
* :func:`apply_matchering` — wraps :func:`matchering.process` and returns
  numpy. Writes ``target.tmp.wav`` and matchering's output INSIDE a
  caller-provided ``temp_dir`` (per-render UUID-named) that is wiped in
  BOTH success and exception paths (T-7-04, RESEARCH §Pitfall 5).

Design notes:

* Read-back-as-numpy (PATTERNS §Pattern 6, planner-chosen). The chain
  ALWAYS returns numpy; the worker (Plan 01 Task 4
  :class:`MasteringRunnable`) owns the atomic write to the final cache
  path. The chain never touches the final cache file — separation of
  concerns between "produce audio" (chain) and "land it durably"
  (worker).
* :func:`matchering.Config` is constructed with ``max_length =
  24 * 60 * 60`` (24 h, Pitfall 7) — matchering's default 15-min cap is
  far below Phase 7's 60-minute keeper ceiling.
* The 1-second floor (Pitfall 2) is enforced BEFORE invoking
  matchering — surfacing a friendly error rather than letting matchering
  raise its cryptic internal validation error.
* matchering's :func:`load` only spawns ffmpeg when :func:`soundfile.read`
  raises ``RuntimeError("unknown format"/"Format not recognised")``. WAV
  + FLAC succeed via ``sf.read`` so the ffmpeg subprocess path is never
  triggered for the references the picker UI allows (Pitfall 3 —
  ``tests/unit/audio/test_matchering_no_ffmpeg.py`` pins this).
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import ClassVar

import matchering
import numpy as np
import soundfile as sf

from marmelade.audio.mastering.params import Param


class MatcheringStage:
    """Declarative descriptor for the Matchering reference-match tail step.

    D-03: NOT a :class:`MasteringStage` subclass. The DSP chain's stages
    return a :func:`pedalboard.Plugin` via ``build_plugin()``; matchering
    is a whole-clip + reference pass with no plugin shape. The picker UI
    introspects :meth:`parameters` to render the reference combobox; the
    chain orchestrator invokes :func:`apply_matchering` directly at the
    tail when ``chain_cfg["matchering"]["enabled"]`` is True.
    """

    name: ClassVar[str] = "matchering"
    display_name: ClassVar[str] = "Matchering (reference match)"

    REFERENCE_PATH_DEFAULT: ClassVar[str] = ""

    def parameters(self) -> dict[str, Param]:
        """Return the single ``reference_path`` choice Param.

        Choices = ``("",)`` placeholder — the picker UI replaces this at
        popup-open time with the contents of
        :func:`marmelade.paths.matchering_reference_dir` (sorted WAV
        + FLAC filenames). ``browse_filter`` triggers ParamsDialog's
        Browse-button rendering (Plan 07-02 Task 1).
        """
        return {
            "reference_path": Param(
                name="reference_path",
                label="Reference track",
                kind="choice",
                default=self.REFERENCE_PATH_DEFAULT,
                requires_recompute=True,
                choices=(self.REFERENCE_PATH_DEFAULT,),
                browse_filter="Audio files (*.wav *.flac);;All files (*)",
                description=(
                    "Drop pro-mastered reference tracks (WAV or FLAC) into "
                    "~/Music/Marmelade/References/. The picker scans the "
                    "directory at popup-open time."
                ),
            ),
        }


# ---------------------------------------------------------------------------
# apply_matchering — RESEARCH §Pattern 6 verbatim, with T-7-04 finally block.
# ---------------------------------------------------------------------------


_MIN_DURATION_SECONDS: float = 1.0


def apply_matchering(
    target_audio: np.ndarray,
    sr: int,
    reference_path: Path | str,
    out_path: Path,
    temp_dir: Path,
) -> np.ndarray:
    """Render ``target_audio`` matched to ``reference_path`` via matchering.

    Per RESEARCH §Pattern 6 read-back-as-numpy design. ``out_path`` is a
    per-render TEMP WAV inside ``temp_dir`` — NOT the final cache file.
    The worker (Plan 01 Task 4 :class:`MasteringRunnable`) owns the
    atomic write to the cache; this function returns numpy and the
    chain returns the same numpy verbatim.

    T-7-04 — Information Disclosure (resource leak): ``temp_dir`` is
    wiped in BOTH success AND exception paths via a ``try / finally``
    block. matchering's internals can write a partial output WAV before
    raising; the cleanup guarantees no orphan tmp files.

    Pitfall 2: clips shorter than 1.0 s are rejected with a friendly
    message BEFORE invoking matchering (matchering's own validation
    surfaces a cryptic ``ModuleError(Code.ERROR_VALIDATION)`` that
    users cannot self-diagnose).

    Pitfall 7: ``max_length`` is lifted to 24 h so Phase 7's 60-min
    keeper ceiling is well within range (matchering default is 15 min).

    Args:
        target_audio: ``(num_channels, num_samples)`` float32 numpy.
        sr: Sample rate in Hz. Canonical rate: 48000 (quick-260615-f77 —
            reverses Phase 2.1 D-04). matchering has NO hard guard and
            works at the target's rate; this is documentation only.
        reference_path: Filesystem path to a WAV or FLAC reference.
            Caller (the chain) is responsible for validating the path
            against the library directory (T-7-01).
        out_path: Per-render TEMP WAV inside ``temp_dir`` that matchering
            writes its output to. The function reads it back into numpy
            and returns the array; the worker handles the atomic write
            to the final cache path. ``out_path`` is destroyed when
            ``temp_dir`` is wiped at function exit.
        temp_dir: Per-render UUID-named temporary directory. Created by
            ``mkdir(parents=True, exist_ok=True)`` on entry and wiped
            via :func:`shutil.rmtree(..., ignore_errors=True)` in
            ``finally`` (T-7-04).

    Returns:
        ``(num_channels, num_samples)`` float32 numpy array — the
        matchered audio. The sample count may differ by up to one FFT
        window from the input (matchering pads to internal block size).

    Raises:
        ValueError: target audio shorter than 1.0 s (Pitfall 2).
        Any exception from :func:`matchering.process` — re-raised after
            temp cleanup.
    """
    if target_audio.shape[-1] < int(_MIN_DURATION_SECONDS * sr):
        duration_s = target_audio.shape[-1] / sr
        raise ValueError(
            f"Keeper is too short ({duration_s:.2f}s) for Matchering "
            f"reference match (need at least {_MIN_DURATION_SECONDS:.1f} s). "
            "Disable Matchering for this keeper or use a longer clip."
        )

    temp_dir = Path(temp_dir)
    out_path = Path(out_path)
    target_tmp = temp_dir / "target.tmp.wav"

    try:
        temp_dir.mkdir(parents=True, exist_ok=True)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Write target as FLOAT WAV — preserves precision; matchering
        # internally normalizes during its load step.
        sf.write(str(target_tmp), target_audio.T, sr, subtype="FLOAT")

        # matchering.Config — Pitfall 7 max_length lift + explicit
        # temp_folder so matchering's intermediates stay within the
        # per-render dir we own (T-7-04 cleanup discipline).
        cfg = matchering.Config(
            internal_sample_rate=sr,
            max_length=24 * 60 * 60,  # 24 h — lift the 15-min default cap.
            temp_folder=str(temp_dir),
        )

        # Invoke matchering. The Result subtype is "PCM_24" — small WAV
        # bit-depth that survives quantization to consumer audio chains
        # downstream. matchering can be slow (FFT analysis + match);
        # RESEARCH §Pitfall 5 — uncancellable mid-call, the caller's
        # cancel-check fires only between stages.
        matchering.process(
            target=str(target_tmp),
            reference=str(reference_path),
            results=[matchering.Result(str(out_path), "PCM_24")],
            config=cfg,
        )

        # Read the matchered output back into numpy. Always float32.
        matched, sr_read = sf.read(str(out_path), dtype="float32", always_2d=True)
        if sr_read != sr:
            raise RuntimeError(
                f"matchering produced sr={sr_read}, expected sr={sr} "
                "(internal sample-rate mismatch — D-04 invariant)"
            )
        # soundfile returns (samples, channels) — transpose to (channels, samples).
        return matched.T.astype(np.float32, copy=False)

    finally:
        # T-7-04 cleanup invariant — wipe temp_dir in BOTH success AND
        # exception paths. ignore_errors=True so a vanished file or
        # permission error during cleanup does NOT mask the original
        # exception (or break a successful return).
        try:
            target_tmp.unlink(missing_ok=True)
        except OSError:
            pass
        shutil.rmtree(temp_dir, ignore_errors=True)


__all__ = ["MatcheringStage", "apply_matchering"]
