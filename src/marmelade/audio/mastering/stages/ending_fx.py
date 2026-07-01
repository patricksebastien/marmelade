"""Ending FX stage — per-keeper tail-appending ring-out (Phase 07.1).

A NEW virtual tail stage that rings out PAST the music: a reverb wash, dub
delay, tape-stop, filter close, bitcrush collapse, codec rot, … instead of a
plain fade. The effect is baked into the per-keeper mastered cache so the
B-mode audition (which plays the cache to its natural EOF) and the export
(which reads the longer cache) both hear the tail for free.

How it works — :func:`apply_ending_fx`:

1. OFF path (disabled / ``None`` / no ``effect_type``) returns the input
   array byte-identical, so today's behavior is unchanged when off
   (SC-6 no-regression).
2. Append ``tail_sec`` of SILENCE to the END of the audio on the samples
   axis so wet effects (reverb / delay) have room to ring out into.
3. Look up a per-``effect_type`` builder in :data:`EFFECT_BUILDERS` via
   ``.get`` — an UNKNOWN / legacy / hostile ``effect_type`` is a graceful
   dry-with-tail no-op (never a ``KeyError``), because this runs in the
   render worker thread where a crash would kill the export.
4. Blend the processed result over the dry body with a time-varying wet
   ENVELOPE so this is an ENDING FX, not a whole-keeper color: the body
   stays DRY, the effect ramps in linearly over the last ``onset_sec`` of
   the body, and the appended tail is fully WET (the ring-out). The ``wet``
   knob scales the maximum wetness.
5. A short safety fade-to-zero on the very tail guarantees the file ends in
   TRUE silence (no abrupt reverb cutoff).

Like :class:`LoudnessStage` / :class:`NormalizeStage`, this is NOT a single
pedalboard plugin: the chain applies it via :func:`apply_ending_fx`.
:meth:`EndingFxStage.build_plugin` therefore raises
:class:`NotImplementedError` (required by the ABC but never called).

Effects use Pedalboard built-ins ONLY (no Convolution / IR — see
07.1-CONTEXT.md "Effects palette"). The 48 kHz canonical-rate guard lives in
the chain, NOT here: ``apply_ending_fx`` honors whatever ``sr`` it is given.

N-3 invariant: no PySide6 / QtWidgets / QtGui imports.
"""

from __future__ import annotations

from typing import Callable, ClassVar

import numpy as np
import pedalboard

from marmelade.audio.mastering.base import MasteringStage
from marmelade.audio.mastering.params import Param

# Default tail length (seconds) per effect_type — locked here so the preset
# module (Task 2) and downstream plans never invent numbers.
EFFECT_TAIL_DEFAULTS: dict[str, float] = {
    "hall_wash": 6.0,
    "dub_echo": 5.0,
    "tape_stop": 2.5,
    "filter_close": 3.0,
    "shimmer_freeze": 7.0,
    "bitcrush_collapse": 3.5,
    "codec_rot": 3.0,
    "glitch_stutter": 3.0,
    "overdrive_bloom": 4.0,
    "smear": 5.0,
}


def _clamp(value: float, low: float, high: float) -> float:
    """Clamp ``value`` into ``[low, high]`` (sidecar tampering defense, T-07.1-01)."""
    return float(min(max(value, low), high))


# ---------------------------------------------------------------------------
# Per-effect_type plugin-chain builders.
#
# Each builder takes the normalized ``primary`` knob in [0, 1] and returns a
# list[pedalboard.Plugin]. Pedalboard built-ins ONLY (no Convolution / IR).
# ``primary`` is clamped to [0, 1] before mapping (sidecar tampering defense).
# ---------------------------------------------------------------------------


def _build_hall_wash(primary: float) -> list[pedalboard.Plugin]:
    p = _clamp(primary, 0.0, 1.0)
    return [
        pedalboard.Reverb(
            room_size=0.85 + 0.14 * p,
            wet_level=0.6,
            dry_level=0.4,
            width=1.0,
            damping=0.3,
        )
    ]


def _build_dub_echo(primary: float) -> list[pedalboard.Plugin]:
    p = _clamp(primary, 0.0, 1.0)
    return [
        pedalboard.Delay(delay_seconds=0.375, feedback=0.55 + 0.4 * p, mix=0.5),
        pedalboard.Reverb(room_size=0.5, wet_level=0.3, dry_level=0.7),
    ]


def _build_tape_stop(primary: float, sr: int) -> list[pedalboard.Plugin]:
    # A true per-sample pitch-drop ramp would need a time-varying PitchShift
    # the offline board cannot do; this Resample-down approximation + a steep
    # filter close is the chosen concrete, implementable approach
    # (07.1-CONTEXT "pick a concrete, implementable approach").
    p = _clamp(primary, 0.0, 1.0)
    target = sr * (1.0 - 0.6 * p)
    return [
        pedalboard.Resample(
            target_sample_rate=float(target),
            quality=pedalboard.Resample.Quality.Linear,
        ),
        pedalboard.LadderFilter(
            mode=pedalboard.LadderFilter.Mode.LPF24,
            cutoff_hz=8000.0,
            resonance=0.2,
        ),
    ]


def _build_filter_close(primary: float) -> list[pedalboard.Plugin]:
    p = _clamp(primary, 0.0, 1.0)
    return [
        pedalboard.LadderFilter(
            mode=pedalboard.LadderFilter.Mode.LPF24,
            cutoff_hz=12000.0 * (1.0 - 0.9 * p) + 200.0,
            resonance=0.3 + 0.4 * p,
            drive=1.0,
        )
    ]


def _build_shimmer_freeze(primary: float) -> list[pedalboard.Plugin]:
    # ``primary`` left as a future knob (shimmer is character-fixed); the
    # octave-up + huge verb defines the effect.
    _clamp(primary, 0.0, 1.0)
    return [
        pedalboard.PitchShift(semitones=12.0),
        pedalboard.Reverb(room_size=0.95, wet_level=0.7, dry_level=0.3, damping=0.1),
    ]


def _build_bitcrush_collapse(primary: float) -> list[pedalboard.Plugin]:
    p = _clamp(primary, 0.0, 1.0)
    bit_depth = int(round(8 - 6 * p))
    bit_depth = max(1, bit_depth)  # clamp to a valid pedalboard range
    return [
        pedalboard.Bitcrush(bit_depth=bit_depth),
        pedalboard.Reverb(room_size=0.6, wet_level=0.4, dry_level=0.6),
    ]


def _build_codec_rot(primary: float) -> list[pedalboard.Plugin]:
    p = _clamp(primary, 0.0, 1.0)
    return [
        pedalboard.GSMFullRateCompressor(),
        pedalboard.MP3Compressor(vbr_quality=2.0 + 7.0 * p),
    ]


def _build_glitch_stutter(primary: float) -> list[pedalboard.Plugin]:
    p = _clamp(primary, 0.0, 1.0)
    return [
        pedalboard.Delay(delay_seconds=0.0625, feedback=0.7 + 0.25 * p, mix=0.6),
        pedalboard.Distortion(drive_db=6.0),
    ]


def _build_overdrive_bloom(primary: float) -> list[pedalboard.Plugin]:
    p = _clamp(primary, 0.0, 1.0)
    return [
        pedalboard.Distortion(drive_db=12.0 + 18.0 * p),
        pedalboard.Reverb(room_size=0.7, wet_level=0.5, dry_level=0.5),
    ]


def _build_smear(primary: float) -> list[pedalboard.Plugin]:
    # Chorus character is fixed; ``primary`` reserved.
    _clamp(primary, 0.0, 1.0)
    return [
        pedalboard.Chorus(rate_hz=0.4, depth=0.9, mix=0.7),
        pedalboard.Reverb(room_size=0.8, wet_level=0.6, dry_level=0.4, damping=0.5),
    ]


# Public registry. ``effect_type`` -> callable(primary) -> list[Plugin].
# ``tape_stop`` needs ``sr`` (Resample target), so it is wrapped in a closure
# that captures ``sr`` inside :func:`apply_ending_fx`. The registry value is
# the sr-independent builder; ``apply_ending_fx`` special-cases tape_stop.
EFFECT_BUILDERS: dict[str, Callable[[float], list[pedalboard.Plugin]]] = {
    "hall_wash": _build_hall_wash,
    "dub_echo": _build_dub_echo,
    # tape_stop is sr-dependent; the registry entry ignores sr and uses a
    # safe default target, but apply_ending_fx rebuilds it with the real sr.
    "tape_stop": lambda primary: _build_tape_stop(primary, 48000),
    "filter_close": _build_filter_close,
    "shimmer_freeze": _build_shimmer_freeze,
    "bitcrush_collapse": _build_bitcrush_collapse,
    "codec_rot": _build_codec_rot,
    "glitch_stutter": _build_glitch_stutter,
    "overdrive_bloom": _build_overdrive_bloom,
    "smear": _build_smear,
}


def _apply_safety_fade(audio: np.ndarray, sr: int) -> np.ndarray:
    """Fade-to-zero so the file ends in TRUE silence — no abrupt reverb cutoff.

    Two stacked linear ramps:

    * A gentle ~50 ms fade over the final stretch smooths a long reverb tail
      down toward zero (musical, no click).
    * A steep final ~10 ms ramp guarantees the very tail is deep silence
      (``< 1e-3``) regardless of how loud the effect was — a 50 ms linear
      fade alone leaves ~0.1·peak at the 5 ms mark, which a hot reverb wash
      can keep above the silence floor.

    The very last sample is forced to exactly 0.0.
    """
    out = audio.astype(np.float32, copy=True)
    n = out.shape[1]
    if n == 0:
        return out
    # Gentle long fade.
    long_len = int(min(0.05 * sr, n))
    if long_len > 0:
        out[:, -long_len:] *= np.linspace(1.0, 0.0, long_len, dtype=np.float32)
    # Steep final ramp → deep silence in the last ~10 ms.
    short_len = int(min(0.01 * sr, n))
    if short_len > 0:
        out[:, -short_len:] *= np.linspace(1.0, 0.0, short_len, dtype=np.float32)
    out[:, -1] = 0.0
    return out


def apply_ending_fx(audio: np.ndarray, sr: int, cfg: dict | None) -> np.ndarray:
    """Append a ``tail_sec`` ring-out tail to ``audio`` and return float32.

    Args:
        audio: ``(channels, samples)`` float array (any float dtype).
        sr: Sample rate in Hz. Effects are built at this rate; there is NO
            internal 48 kHz assumption (the 48 kHz guard lives in the chain).
        cfg: The bare ``ending_fx`` stage dict
            ``{"enabled", "effect_type", "tail_sec", "wet", "primary"}`` or
            ``None``.

    Returns:
        float32 ``(channels, samples)``. When the stage is OFF (disabled /
        ``None`` / no ``effect_type``) the input is returned byte-identical.
        Otherwise the output is LONGER than the input by ~``tail_sec`` and
        ends in TRUE silence via a short safety fade.

    An UNKNOWN ``effect_type`` (not in :data:`EFFECT_BUILDERS`) is a graceful
    dry-with-tail no-op: the padded array is returned with the safety fade,
    NO effect applied, and the function NEVER raises (render-worker safety —
    T-07.1-03).
    """
    # 1. OFF path — byte-identical pass-through (SC-6 no-regression).
    if cfg is None or not cfg.get("enabled", False) or not cfg.get("effect_type"):
        return audio

    effect_type = str(cfg.get("effect_type"))
    tail_sec = _clamp(float(cfg.get("tail_sec", 4.0)), 0.5, 12.0)
    wet = _clamp(float(cfg.get("wet", 1.0)), 0.0, 1.0)
    primary = _clamp(float(cfg.get("primary", 0.5)), 0.0, 1.0)
    onset_sec = _clamp(float(cfg.get("onset_sec", 2.0)), 0.1, 8.0)

    channels = audio.shape[0]
    dry = audio.astype(np.float32, copy=False)
    body_len = audio.shape[1]  # capture BEFORE padding for the onset envelope

    # 2. Append silence so wet effects can ring out into the tail.
    tail_room = int(round(tail_sec * sr))
    padded = np.concatenate(
        [dry, np.zeros((channels, tail_room), dtype=np.float32)], axis=1
    )

    # 3. Look up the builder. UNKNOWN effect_type → graceful no-op (never
    #    KeyError): treat the padded array as the wet output (dry pass-through).
    if effect_type == "tape_stop":
        plugins = _build_tape_stop(primary, sr)
    else:
        builder = EFFECT_BUILDERS.get(effect_type)
        plugins = builder(primary) if builder is not None else None

    if plugins is None:
        wet_out = padded
    else:
        board = pedalboard.Pedalboard(plugins)
        wet_out = np.asarray(board(padded, sr), dtype=np.float32)

    # 4. Wet/dry mix. Length-align in case a resampling effect changed the
    #    sample count, padding the shorter to the longer with zeros.
    n = max(padded.shape[1], wet_out.shape[1])
    if padded.shape[1] < n:
        padded = np.concatenate(
            [padded, np.zeros((channels, n - padded.shape[1]), np.float32)], axis=1
        )
    if wet_out.shape[1] < n:
        wet_out = np.concatenate(
            [wet_out, np.zeros((channels, n - wet_out.shape[1]), np.float32)], axis=1
        )

    # 4b. Time-varying wet ENVELOPE so this is an ENDING FX, not a whole-keeper
    #     color: env=0 over the dry body, a linear ramp 0→1 over the last
    #     onset_sec of the body, env=1 over the appended wet tail. Multiplying
    #     by the user `wet` knob scales the maximum wetness.
    env = np.zeros(n, dtype=np.float32)
    onset_n = min(int(onset_sec * sr), body_len)  # clamp to short keepers
    if onset_n > 0:
        env[body_len - onset_n : body_len] = np.linspace(
            0.0, 1.0, onset_n, dtype=np.float32
        )
    env[body_len:] = 1.0  # fully wet through the ring-out tail
    e = (wet * env).reshape(1, n)  # broadcast over channels
    mix = ((1.0 - e) * padded + e * wet_out).astype(np.float32)

    # 5. Safety fade → true silence at the very end.
    out = _apply_safety_fade(mix, sr)
    return np.ascontiguousarray(out, dtype=np.float32)


class EndingFxStage(MasteringStage):
    """Declare the ending-FX stage's tunable surface (effect/tail/onset/wet/primary).

    Like :class:`LoudnessStage` / :class:`NormalizeStage`, this is NOT a
    pedalboard plugin: the chain applies it via :func:`apply_ending_fx`.
    This class only declares the five Params so the Mastering dialog can
    render the gear dialog via the existing auto-render path.
    """

    name: ClassVar[str] = "ending_fx"
    display_name: ClassVar[str] = "Ending FX"

    EFFECT_TYPE_DEFAULT: ClassVar[str] = "hall_wash"
    TAIL_SEC_DEFAULT: ClassVar[float] = 4.0
    WET_DEFAULT: ClassVar[float] = 1.0
    PRIMARY_DEFAULT: ClassVar[float] = 0.5
    ONSET_SEC_DEFAULT: ClassVar[float] = 2.0

    def parameters(self) -> dict[str, Param]:
        effect_ids = tuple(EFFECT_BUILDERS.keys())
        return {
            "effect_type": Param(
                name="effect_type",
                label="Effect",
                kind="choice",
                default=self.EFFECT_TYPE_DEFAULT,
                requires_recompute=True,
                choices=effect_ids,
                description="Tail ring-out effect baked into the keeper's mastered cache.",
            ),
            "tail_sec": Param(
                name="tail_sec",
                label="Tail length",
                kind="float",
                default=self.TAIL_SEC_DEFAULT,
                requires_recompute=True,
                min=0.5,
                max=12.0,
                step=0.5,
                unit="s",
                description="Seconds of ring-out appended past the region end.",
            ),
            "onset_sec": Param(
                name="onset_sec",
                label="FX onset",
                kind="float",
                default=self.ONSET_SEC_DEFAULT,
                requires_recompute=True,
                min=0.1,
                max=8.0,
                step=0.1,
                unit="s",
                description="Seconds before the region end where the effect blends in from dry.",
            ),
            "wet": Param(
                name="wet",
                label="Wet/Dry",
                kind="float",
                default=self.WET_DEFAULT,
                requires_recompute=True,
                min=0.0,
                max=1.0,
                step=0.05,
                description="Effect mix: 1.0 = fully wet tail, 0.0 = dry (silent tail).",
            ),
            "primary": Param(
                name="primary",
                label="Amount",
                kind="float",
                default=self.PRIMARY_DEFAULT,
                requires_recompute=True,
                min=0.0,
                max=1.0,
                step=0.05,
                description="Per-effect primary knob (decay / feedback / crush depth).",
            ),
        }

    def build_plugin(self) -> pedalboard.Plugin:
        """Not a pedalboard plugin — the chain applies ending FX directly.

        Mirrors LoudnessStage / NormalizeStage: the orchestrator special-cases
        this stage (:func:`apply_ending_fx`) and never calls
        :meth:`build_plugin`. Required by the ABC; raising keeps the contract
        honest (any accidental factory call surfaces loudly).
        """
        raise NotImplementedError(
            "EndingFxStage is applied directly via apply_ending_fx in "
            "MasteringChain.process, not as a pedalboard plugin."
        )
