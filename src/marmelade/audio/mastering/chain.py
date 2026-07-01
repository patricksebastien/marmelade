"""Mastering chain orchestrator — config hash, session snapshot, run_dsp_chain.

Phase 7 — 07-RESEARCH.md §Pattern 2 / §Pattern 4 verbatim.

Public surface:

* :data:`_STAGE_ORDER` — fixed stage order (HP/LP/EQ/Compressor/Limiter
  then ``matchering`` as the tail stage).
* :data:`_SESSION_DEFAULTS` — initial values for an empty QSettings session
  chain (per-stage ``enabled`` flags + UI-SPEC default parameter values).
* :func:`load_session_chain_snapshot` — snapshot the current QSettings
  ``mastering/session/`` sub-tree into a plain ``dict`` (D-04 — keepers
  capture this verbatim at creation time).
* :func:`config_hash` — stable 12-hex SHA-1 over canonical-form JSON of a
  mastering config (golden vectors A..E in
  ``tests/unit/audio/test_config_hash.py``).
* :func:`run_pre_limiter_stages` — apply HP→LP→EQ→Compressor.
* :func:`run_dsp_chain` — apply the full pre-limiter chain plus the
  limiter sub-chain when enabled.
* :class:`MasteringChain` — orchestrator instance holding the per-keeper
  config; integrates the DSP chain + LUFS makeup + ISP verification.

QSettings boundary import (PATTERNS Reuse Discipline 5 option (b)): the
``load_session_chain_snapshot`` helper crosses the audio/Qt boundary
once, deliberately, so the audio tier remains the single source of truth
for the canonical-form config dict.

Cancellation: :class:`MasteringChain.process` polls the injected
``_cancel_check`` callable between stages (never mid-call) and raises
:class:`marmelade.audio.peak_builder.BuildCancelled` — the same
exception class used everywhere else in the codebase (Reuse Discipline
10 — no peer ``_MasteringCancelled`` class).
"""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pedalboard
# Deliberate boundary import — see module docstring. PATTERNS Reuse
# Discipline 5 option (b). The N-3 invariant test (test_n3_invariant.py)
# explicitly allows QtCore.QSettings under audio/mastering/ and forbids
# only the QtWidgets / QtGui surface.
from PySide6.QtCore import QSettings

from marmelade.audio.mastering.lufs import (
    apply_lufs_makeup,
    normalize_to_lufs_target,
    run_isp_verification,
)
from marmelade.audio.mastering.stages.compressor import CompressorStage
from marmelade.audio.mastering.stages.delay import DelayStage
from marmelade.audio.mastering.stages.distortion import DistortionStage
from marmelade.audio.mastering.stages.ending_fx import apply_ending_fx
from marmelade.audio.mastering.stages.eq import EqStage
from marmelade.audio.mastering.stages.highpass import HighPassStage
from marmelade.audio.mastering.stages.limiter import build_limiter_subchain
from marmelade.audio.mastering.stages.lowpass import LowPassStage
from marmelade.audio.mastering.stages.matchering import apply_matchering
from marmelade.audio.mastering.stages.reverb import ReverbStage
from marmelade.audio.mastering.stages.vst3 import apply_vst3
from marmelade.audio.normalize import normalize_array
from marmelade.audio.peak_builder import BuildCancelled
from marmelade.paths import mastered_cache_dir, matchering_reference_dir

# Stage order is FIXED (D-03 + UI-SPEC §Mastering dock per-stage rows).
# ``matchering`` is the tail stage; the DSP prefix is the only thing
# ``run_dsp_chain`` iterates over (Plan 05 lands the matchering branch).
# NOTE: this tuple is the PANEL/DISPLAY order (MasteringDock + MasteringDialog
# render rows by iterating it) and the snapshot key set — it is NOT the DSP
# processing order. Processing order is driven by ``_PRE_LIMITER_STAGES`` plus
# the hardcoded tail in ``MasteringChain.process`` (limiter → LUFS → ISP →
# matchering → normalize). ``normalize`` is listed FIRST so it appears at the
# top of the mastering panel, but it is still APPLIED LAST in the chain.
# ``config_hash`` is order-insensitive (json.dumps sort_keys=True), so this
# ordering does not affect mastering-cache validity.
_STAGE_ORDER: tuple[str, ...] = (
    # quick-260626-o9y — OUTPUT-TIME fade in/out. Listed FIRST so it is the
    # top display row (above normalize) in the Mastering dialog/dock. This is
    # DISPLAY-ONLY: fade is NEVER applied in process() (no _PRE_LIMITER_STAGES
    # / _STAGE_FACTORY entry); the actual envelope is applied at export/preview
    # by main_window via fade_params(). It is dropped UNCONDITIONALLY from
    # config_hash so toggling/editing it never busts the mastered cache.
    "fade",
    "normalize",
    # quick-260623-l7l — absolute LUFS target stage; placed adjacent to
    # ``normalize`` in the DISPLAY order. Apply order is hardcoded in
    # process() (loudness is applied AFTER matchering, BEFORE the trailing
    # normalize block, and bypasses that normalize block when enabled).
    "loudness",
    "highpass",
    "lowpass",
    "eq",
    "compressor",
    # quick-260629 — whole-clip color stages (drive / echo / ambience). Added
    # after compressor and before the VST3/limiter so the true-peak limiter
    # still guards the output. All three are normal pedalboard plugins in the
    # _PRE_LIMITER_STAGES factory; a DISABLED instance canonicalizes away in
    # config_hash (like ending_fx / vst3) so adding them does NOT bust existing
    # keepers' mastered caches or break genre-preset matching.
    "distortion",
    "delay",
    "reverb",
    # quick-260625 — external VST3 plugin slot. DISPLAY order places it among
    # the processing stages, IMMEDIATELY BEFORE the limiter; APPLY order
    # (hardcoded in process()) also runs it after the built-in pre-limiter
    # chain and before the true-peak limiter so the limiter still guards the
    # output. config_hash is order-insensitive, so the position is cosmetic.
    "vst3",
    "limiter",
    # Phase 07.1 — per-keeper ending-FX tail. DISPLAY order only; placed
    # adjacent to ``matchering`` for panel readability, IMMEDIATELY BEFORE it
    # so ``matchering`` stays the LAST display entry (pinned by
    # test_normalize_is_first_in_stage_order). APPLY order is hardcoded in
    # process() (ending_fx is applied AFTER matchering, BEFORE the loudness
    # tail, so its ring-out is included in the LUFS measurement + final
    # ISP/normalize). config_hash is order-insensitive, so this position does
    # not affect mastering-cache validity.
    "ending_fx",
    "matchering",
)

# quick-260629 — UI DISPLAY grouping only. The creative effect stages are shown
# together under an "FX" sub-section at the TOP of the mastering panel (dock +
# per-keeper dialog): the three whole-clip color stages (distortion/delay/
# reverb), then the output-time fade in/out, then the ending-FX tail. This does
# NOT change ``_STAGE_ORDER`` (the snapshot key set + panel anchors
# fade/normalize/…/matchering — ``fade`` stays _STAGE_ORDER[0]) NOR the DSP
# apply order — distortion/delay/reverb still run in ``_PRE_LIMITER_STAGES``
# (after compressor, before the limiter), ending_fx still runs as the tail
# (after matchering), and fade is still applied at export/preview time. It is
# purely the set the UI lifts into the FX box.
_FX_STAGES: tuple[str, ...] = (
    "distortion",
    "delay",
    "reverb",
    "fade",
    "ending_fx",
)

# Initial defaults — used as the seed values when QSettings has no
# entry yet. These are the values the empty-settings session dock shows
# on first launch. Mirror :data:`_STAGE_ORDER` shape.
_SESSION_DEFAULTS: dict[str, dict[str, Any]] = {
    # quick-260626-o9y — output-time fade. Default ON @ 2.0 s reproduces
    # today's forced 2.0 s fade. Never applied in process(); read at
    # export/preview via fade_params(). Excluded from config_hash entirely.
    "fade": {"enabled": True, "duration_sec": 2.0},
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
    # quick-260629 — whole-clip color stages. Default DISABLED so existing
    # sessions/keepers are byte-identical no-ops; defaults mirror the stage
    # ClassVars (Reverb/Distortion/Delay).
    "distortion": {"enabled": False, "drive_db": 25.0},
    "delay": {
        "enabled": False,
        "delay_seconds": 0.375,
        "feedback": 0.3,
        "mix": 0.3,
    },
    "reverb": {
        "enabled": False,
        "room_size": 0.5,
        "damping": 0.5,
        "wet_level": 0.33,
        "dry_level": 0.4,
        "width": 1.0,
    },
    "limiter": {"enabled": True, "ceiling_dbtp": -1.0, "release_ms": 100.0},
    "matchering": {"enabled": False, "reference_path": ""},
    # quick-260625 — external VST3 plugin slot. Default DISABLED with empty
    # path so existing sessions/keepers are byte-identical no-ops. Loaded +
    # applied directly in process() via apply_vst3 (NOT a pedalboard-list
    # factory entry). plugin state is captured from the native editor and
    # stored as base64 (see stages.vst3).
    "vst3": {
        "enabled": False,
        "plugin_path": "",
        "plugin_name": "",
        "state_b64": "",
    },
    # quick-260621-gfq — normalize is the FINAL chain stage (default OFF @
    # 0 dBFS). Default 0.0 (locked decision #6) replaces the old -6.0 keeper
    # default. Applied directly via normalize_array, not a pedalboard plugin.
    "normalize": {"enabled": False, "target_db": 0.0},
    # quick-260623-l7l — absolute LUFS target (default OFF @ -14 LUFS for
    # streaming delivery). Virtual stage: applied directly in process() via
    # normalize_to_lufs_target + run_isp_verification, NOT a pedalboard plugin.
    "loudness": {"enabled": False, "target_lufs": -14.0},
    # Phase 07.1 — per-keeper ending FX. Virtual tail stage applied directly
    # in process() via apply_ending_fx (NOT a pedalboard plugin). Default
    # DISABLED so existing sessions/keepers are unaffected; mirrors
    # EndingFxStage Param defaults.
    "ending_fx": {
        "enabled": False,
        "effect_type": "hall_wash",
        "tail_sec": 4.0,
        "onset_sec": 2.0,
        "wet": 1.0,
        "primary": 0.5,
    },
}


def _coerce_like(value: Any, default: Any) -> Any:
    """Coerce a ``QSettings``-typed value to the default's type.

    QSettings returns ``str`` for bool/float on some platforms; we coerce
    against the default's type so the round-trip is stable across OSes.
    """
    if isinstance(default, bool):
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes")
        return bool(value)
    if isinstance(default, float):
        return float(value)
    if isinstance(default, int):
        return int(value)
    if value is None:
        return default
    return str(value)


def load_session_chain_snapshot() -> dict[str, dict[str, Any]]:
    """Snapshot the current QSettings session chain into a plain ``dict``.

    D-04: returned dict is stored verbatim in a new keeper's sidecar
    ``mastering`` field. Subsequent QSettings edits do NOT propagate to
    existing keepers.

    D-16: explicit ``("Marmelade","Marmelade")`` org/app pair so the
    helper shares the test-mode sandbox the MainWindow side uses
    (``conftest.py`` monkeypatches ``QCoreApplication.organizationName``
    / ``applicationName`` to ``"Marmelade-test"``).

    Missing keys fall back to :data:`_SESSION_DEFAULTS[stage][key]`.
    """
    s = QSettings("Marmelade", "Marmelade")
    out: dict[str, dict[str, Any]] = {}
    for stage in _STAGE_ORDER:
        out[stage] = {}
        s.beginGroup(f"mastering/session/{stage}")
        try:
            for k in s.childKeys():
                default = _SESSION_DEFAULTS[stage].get(k)
                raw = s.value(k, default)
                out[stage][k] = _coerce_like(raw, default)
            # Fill missing keys from defaults so the snapshot is COMPLETE.
            for k, v in _SESSION_DEFAULTS[stage].items():
                out[stage].setdefault(k, v)
        finally:
            s.endGroup()
    return out


def config_hash(mastering: dict[str, dict[str, Any]]) -> str:
    """Return a stable 12-hex SHA-1 hash over the canonical-form chain config.

    Canonical form (RESEARCH §Pattern 2):
      * ``sort_keys=True`` — key order is deterministic.
      * ``ensure_ascii=True`` — non-ASCII reference paths hash
        deterministically across platforms.
      * Float normalization: round to 6 decimal places (pedalboard plugin
        ranges have resolution well coarser than 1e-6).
      * Disabled stages contribute ONLY ``{"enabled": False}`` — their
        other params are dropped before hashing so toggling a cutoff
        while a stage is disabled does not change the hash (Vector A).

    Truncated to 12 hex chars (48 bits) — collision probability
    negligible at the scale of one user's keepers per file.
    """
    canon: dict[str, dict[str, Any]] = {}
    for stage_name, stage_cfg in mastering.items():
        # quick-260626-o9y — ``fade`` is an OUTPUT-TIME stage applied at
        # export/preview, NOT baked into the mastered cache. It must therefore
        # contribute NOTHING to the cache key: drop it UNCONDITIONALLY (whether
        # enabled OR disabled) so a fade toggle / duration edit NEVER busts the
        # mastered cache. This is distinct from the ending_fx/vst3 case below,
        # which only drops a DISABLED instance. A cfg with fade enabled, fade
        # disabled, and no "fade" key at all all hash EQUAL.
        if stage_name == "fade":
            continue
        if not stage_cfg.get("enabled", False):
            # Phase 07.1 — a DISABLED ``ending_fx`` canonicalizes away
            # entirely (dropped from the hash input) so it hashes EQUAL to a
            # config that has no ``ending_fx`` key at all. Legacy keepers
            # predate this stage and never carry the key; a new keeper that
            # merely toggles the stage off must NOT bust their mastered cache.
            # ``ending_fx`` AND ``vst3`` (quick-260625) are special-cased — a
            # disabled instance canonicalizes away entirely so it hashes EQUAL
            # to a config that lacks the key. This keeps the genre-preset
            # match working (presets never carry these per-keeper-only keys)
            # and keeps legacy keepers' cache filenames stable. Every OTHER
            # disabled stage keeps contributing ``{"enabled": False}``
            # (preserves the existing golden vectors A..E).
            # quick-260629 — reverb / distortion / delay join this set: a
            # DISABLED instance drops out entirely so it hashes EQUAL to a
            # config that lacks the key. This keeps existing keepers' mastered
            # cache filenames stable and lets the genre-preset matcher (which
            # never carries these keys) still match when the stages are off.
            if stage_name in (
                "ending_fx",
                "vst3",
                "reverb",
                "distortion",
                "delay",
            ):
                continue
            canon[stage_name] = {"enabled": False}
            continue
        canon[stage_name] = {
            k: (round(v, 6) if isinstance(v, float) else v)
            for k, v in stage_cfg.items()
        }
    payload = json.dumps(canon, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Pedalboard chain assembly (RESEARCH §Pattern 4)
# ---------------------------------------------------------------------------


def _build_eq_plugin(cfg: dict[str, Any]) -> pedalboard.Plugin:
    """Build the nested 3-band EQ Pedalboard from a stage config dict.

    Single source of truth: the band center frequencies + Q are read from
    :class:`EqStage` ClassVars (NOT hardcoded here). DSP-identical to
    ``EqStage.build_plugin()`` given matching low_db/mid_db/high_db.
    """
    return pedalboard.Pedalboard(
        [
            pedalboard.LowShelfFilter(
                cutoff_frequency_hz=EqStage.LOW_HZ,
                gain_db=float(cfg.get("low_db", EqStage.LOW_DB_DEFAULT)),
                q=EqStage.Q_DEFAULT,
            ),
            pedalboard.PeakFilter(
                cutoff_frequency_hz=EqStage.MID_HZ,
                gain_db=float(cfg.get("mid_db", EqStage.MID_DB_DEFAULT)),
                q=EqStage.Q_DEFAULT,
            ),
            pedalboard.HighShelfFilter(
                cutoff_frequency_hz=EqStage.HIGH_HZ,
                gain_db=float(cfg.get("high_db", EqStage.HIGH_DB_DEFAULT)),
                q=EqStage.Q_DEFAULT,
            ),
        ]
    )


_STAGE_FACTORY: dict[str, Callable[[dict[str, Any]], pedalboard.Plugin]] = {
    "highpass": lambda cfg: pedalboard.HighpassFilter(
        cutoff_frequency_hz=float(cfg["cutoff_hz"]),
    ),
    "lowpass": lambda cfg: pedalboard.LowpassFilter(
        cutoff_frequency_hz=float(cfg["cutoff_hz"]),
    ),
    "eq": lambda cfg: _build_eq_plugin(cfg),
    "compressor": lambda cfg: pedalboard.Compressor(
        threshold_db=float(cfg["threshold_db"]),
        ratio=float(cfg["ratio"]),
        attack_ms=float(cfg["attack_ms"]),
        release_ms=float(cfg["release_ms"]),
    ),
    # quick-260629 — whole-clip color stages. Defaults read from the stage
    # ClassVars (single source of truth) so a missing key falls back to the
    # same value the stage's parameters() declares.
    "distortion": lambda cfg: pedalboard.Distortion(
        drive_db=float(cfg.get("drive_db", DistortionStage.DRIVE_DB_DEFAULT)),
    ),
    "delay": lambda cfg: pedalboard.Delay(
        delay_seconds=float(
            cfg.get("delay_seconds", DelayStage.DELAY_SECONDS_DEFAULT)
        ),
        feedback=float(cfg.get("feedback", DelayStage.FEEDBACK_DEFAULT)),
        mix=float(cfg.get("mix", DelayStage.MIX_DEFAULT)),
    ),
    "reverb": lambda cfg: pedalboard.Reverb(
        room_size=float(cfg.get("room_size", ReverbStage.ROOM_SIZE_DEFAULT)),
        damping=float(cfg.get("damping", ReverbStage.DAMPING_DEFAULT)),
        wet_level=float(cfg.get("wet_level", ReverbStage.WET_LEVEL_DEFAULT)),
        dry_level=float(cfg.get("dry_level", ReverbStage.DRY_LEVEL_DEFAULT)),
        width=float(cfg.get("width", ReverbStage.WIDTH_DEFAULT)),
    ),
}


_PRE_LIMITER_STAGES: tuple[str, ...] = (
    "highpass",
    "lowpass",
    "eq",
    "compressor",
    # quick-260629 — whole-clip color stages, applied after the corrective
    # chain and before the limiter so the true-peak ceiling still holds.
    "distortion",
    "delay",
    "reverb",
)


def _check_cancel(cancel_check: Callable[[], bool] | None) -> None:
    """Raise :class:`BuildCancelled` if the cancel callback returns True."""
    if cancel_check is not None and cancel_check():
        raise BuildCancelled()


def _validate_reference_path(path: str, is_one_off: bool) -> Path:
    """T-7-01 mitigation — resolve & containment-check a matchering reference path.

    Resolves ``path`` to an absolute :class:`pathlib.Path`, asserts it
    points to an existing file, and (unless ``is_one_off`` is True)
    asserts it lives inside :func:`matchering_reference_dir`.

    ``is_one_off`` is set by the reference-picker UI on Browse
    selection (the user explicitly chose a file outside the library).

    WR-01 (Phase 7 review) — IMPORTANT correction to the prior docstring:
    ``is_one_off`` IS persisted into the sidecar JSON (and into
    QSettings via the session-chain dock); it is NOT transient across
    sessions. The earlier wording ("transient flag") was wrong. The
    flag survives file close/reopen, so a hostile sidecar with
    ``{"matchering": {"is_one_off": true, "reference_path":
    "/etc/some-readable-wav"}}`` would bypass the library-containment
    check here. Sidecar-load validation in
    :func:`marmelade.audio.sidecar_cache._validate_mastering_dict`
    is the gate that rejects such sidecars before they ever reach
    this function — see that function's docstring for the hardened
    matchering-stage rules added under WR-01.

    The attack model — "user opens a sidecar from a hostile party" —
    is the relevant threat surface, and the project's mitigation is
    twofold: (1) reject hostile sidecars at load time
    (sidecar_cache), and (2) defense-in-depth check the resolved-
    absolute path here against the library dir whenever
    ``is_one_off`` is False.

    Args:
        path: The configured ``mastering.matchering.reference_path``
            string. Absolute or relative; relative paths are resolved
            against the current working directory.
        is_one_off: When True, skip the library-containment check (the
            picker UI sets this on Browse selection). Persisted into
            the sidecar — sanitized at sidecar load time.

    Returns:
        The resolved-absolute :class:`pathlib.Path`.

    Raises:
        FileNotFoundError: ``path`` does not point to an existing file.
        ValueError: ``path`` resolves outside the library directory and
            ``is_one_off`` is False (T-7-01).
    """
    p = Path(path).expanduser().resolve()
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(
            f"Matchering reference path does not point to a file: {p}"
        )
    if is_one_off:
        return p
    ref_dir = matchering_reference_dir().resolve()
    # is_relative_to in py3.9+ — D-04 invariant Python 3.10+ in pyproject.
    # WR-08 (Phase 7 review) — best-effort containment: `resolve()`
    # follows symlinks at validate time, but the window between this
    # check and the matchering call opening the file is non-zero, so
    # a TOCTOU swap of a symlink target is theoretically possible.
    # The single-user desktop threat model treats the local filesystem
    # under ~/Music/Marmelade/References/ as user-owned, so a
    # hostile concurrent process is out of scope. If a future
    # contributor needs to harden this further (multi-user system,
    # automounter), capture os.stat() here and re-verify after
    # matchering reads — see review note.
    if not p.is_relative_to(ref_dir):
        raise ValueError(
            f"Matchering reference path {p} resolves outside the configured "
            f"reference library directory ({ref_dir}). Either drop the file "
            "into the references library or re-pick it via the Browse button."
        )
    return p


def run_pre_limiter_stages(
    audio: np.ndarray,
    sr: int,
    chain_cfg: dict[str, dict[str, Any]],
    cancel_check: Callable[[], bool] | None = None,
) -> np.ndarray:
    """Apply HP→LP→EQ→Compressor (limiter NOT included).

    Disabled stages are skipped. Cancellation is polled between stages
    (RESEARCH §Pitfall 5 — pedalboard chains are not interruptible
    mid-call). Audio shape: ``(num_channels, num_samples)`` float32.

    The MasteringChain orchestrator calls this to obtain the
    pre-limiter audio so :func:`apply_lufs_makeup` has a reference
    against which to compute makeup gain.
    """
    plugins: list[pedalboard.Plugin] = []
    for name in _PRE_LIMITER_STAGES:
        cfg = chain_cfg.get(name, {})
        if not cfg.get("enabled", False):
            continue
        _check_cancel(cancel_check)
        plugins.append(_STAGE_FACTORY[name](cfg))

    if not plugins:
        return audio

    _check_cancel(cancel_check)
    pb = pedalboard.Pedalboard(plugins)
    return pb(audio, sr)


def run_dsp_chain(
    audio: np.ndarray,
    sr: int,
    chain_cfg: dict[str, dict[str, Any]],
    cancel_check: Callable[[], bool] | None = None,
) -> np.ndarray:
    """Apply the full DSP chain (HP→LP→EQ→Compressor→Limiter).

    Stage order is FIXED (D-03). Disabled stages are skipped. The
    limiter sub-chain (``build_limiter_subchain``) is appended when
    ``limiter.enabled`` is True. Audio shape: ``(num_channels,
    num_samples)`` float32.

    NOTE: this helper does NOT apply LUFS makeup or ISP verification —
    those are :class:`MasteringChain.process`'s job because they need
    the pre-limiter reference audio. Use :func:`run_pre_limiter_stages`
    to obtain that reference.
    """
    plugins: list[pedalboard.Plugin] = []
    for name in _PRE_LIMITER_STAGES:
        cfg = chain_cfg.get(name, {})
        if not cfg.get("enabled", False):
            continue
        _check_cancel(cancel_check)
        plugins.append(_STAGE_FACTORY[name](cfg))

    limiter_cfg = chain_cfg.get("limiter", {})
    if limiter_cfg.get("enabled", False):
        _check_cancel(cancel_check)
        plugins.extend(build_limiter_subchain(limiter_cfg))

    if not plugins:
        return audio

    _check_cancel(cancel_check)
    pb = pedalboard.Pedalboard(plugins)
    return pb(audio, sr)


# ---------------------------------------------------------------------------
# Orchestrator class — bundles the DSP chain + LUFS makeup + ISP guard.
# ---------------------------------------------------------------------------


class MasteringChain:
    """Bundle a per-keeper mastering config + the rendering pipeline.

    Instantiated by the worker shell:

    .. code-block:: python

        chain = MasteringChain(keeper_mastering_cfg)
        chain._cancel_check = self._is_cancelled       # getattr idiom
        chain._stage_progress_cb = self._on_stage      # getattr idiom
        out = chain.process(audio, sr)

    The cancel-check and progress-callback are injected via attribute
    assignment (Phase 6 getattr idiom — no positional argument
    coupling). ``process()`` reads them via :func:`getattr` with
    ``None`` defaults so direct unit-test callers can omit them.
    """

    def __init__(self, mastering_cfg: dict[str, dict[str, Any]]) -> None:
        self._cfg = mastering_cfg

    @property
    def cfg(self) -> dict[str, dict[str, Any]]:
        """Read-only view of the chain's config (used by tests)."""
        return self._cfg

    def _emit_stage_progress(self, pct: int) -> None:
        """Forward to the injected progress callback if present."""
        cb = getattr(self, "_stage_progress_cb", None)
        if cb is not None:
            try:
                cb(int(pct))
            except Exception:
                # Defensive: a flaky callback must not abort the chain.
                pass

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Render ``audio`` through the mastering pipeline.

        Stages:

        1. Apply pre-limiter DSP chain (HP→LP→EQ→Compressor).
        2. Capture pre-limit reference for LUFS measurement.
        3. Apply limiter sub-chain (when enabled).
        4. Apply LUFS makeup gain (clamped to headroom).
        5. Apply ISP verification fallback gain (provable -X dBTP).

        Stage-progress emit points (RESEARCH §Pattern 4 — 5/50/70/85/95
        percent; the worker emits the 5/100% bookends itself):

            50% — pre-limiter chain complete.
            70% — limiter sub-chain complete.
            85% — LUFS makeup complete.
            95% — ISP verification complete.

        Args:
            audio: ``(num_channels, num_samples)`` float32 numpy array.
            sr: Sample rate in Hz. MUST equal 48000 (canonical rate
                per quick-260615-f77 — reverses Phase 2.1 D-04; the
                pipeline now standardizes on 48 kHz, resampling
                non-48 kHz sources to 48 kHz on open).

        Returns:
            ``(num_channels, num_samples)`` float32 numpy array with
            sample peak ≤ ``ceiling_dbtp`` dBTP (provable when limiter
            is enabled).

        Raises:
            ValueError: if ``sr != 48000`` OR if the matchering branch
                fires with a reference path that resolves outside the
                library directory (T-7-01).
            FileNotFoundError: if the matchering branch fires with a
                reference path that does not exist on disk.
            BuildCancelled: if the injected cancel-check returns True
                between stages. (matchering itself is uncancellable
                mid-call — RESEARCH §Pitfall 5.)
        """
        if sr != 48000:
            raise ValueError(
                "MasteringChain requires sr=48000 (canonical rate, "
                f"quick-260615-f77 — reverses Phase 2.1 D-04), got {sr}"
            )
        cancel_check: Callable[[], bool] | None = getattr(self, "_cancel_check", None)
        _check_cancel(cancel_check)

        # (1) Pre-limiter DSP chain — gives us the reference audio for
        # the LUFS makeup-gain measurement.
        audio_pre_limit = run_pre_limiter_stages(audio, sr, self._cfg, cancel_check)
        self._emit_stage_progress(50)
        _check_cancel(cancel_check)

        # (1b) External VST3 plugin slot (quick-260625) — runs AFTER the
        # built-in pre-limiter chain and BEFORE the limiter so our true-peak
        # limiter still guards the output even if the plugin pushes hot. The
        # result becomes the new pre-limit reference, so the LUFS makeup-gain
        # measurement compares what actually entered vs exited the limiter.
        # Disabled / unconfigured / missing-file is a passthrough no-op.
        audio_pre_limit = apply_vst3(
            audio_pre_limit, sr, self._cfg.get("vst3", {}), cancel_check
        )
        _check_cancel(cancel_check)

        # (2) Limiter sub-chain (when enabled).
        limiter_cfg = self._cfg.get("limiter", {})
        limiter_enabled = bool(limiter_cfg.get("enabled", False))
        if limiter_enabled:
            sub = build_limiter_subchain(limiter_cfg)
            pb = pedalboard.Pedalboard(sub)
            audio_post_limit = pb(audio_pre_limit, sr)
        else:
            audio_post_limit = audio_pre_limit
        self._emit_stage_progress(70)
        _check_cancel(cancel_check)

        # (3) LUFS makeup gain (only meaningful when something limited).
        if limiter_enabled:
            ceiling_dbtp = float(limiter_cfg.get("ceiling_dbtp", -1.0))
            audio_after_makeup = apply_lufs_makeup(
                audio_pre_limit, audio_post_limit, sr, ceiling_dbtp=ceiling_dbtp
            )
        else:
            audio_after_makeup = audio_post_limit
        self._emit_stage_progress(85)
        _check_cancel(cancel_check)

        # (4) ISP verification — provable -X dBTP at upsampled rate.
        if limiter_enabled:
            ceiling_dbtp = float(limiter_cfg.get("ceiling_dbtp", -1.0))
            audio_final = run_isp_verification(audio_after_makeup, sr, ceiling_dbtp=ceiling_dbtp)
        else:
            audio_final = audio_after_makeup
        self._emit_stage_progress(95)
        _check_cancel(cancel_check)

        # (5) Matchering tail — D-03 whole-clip pass + reference match.
        # Enabled only when (a) the user toggled it on AND (b) a
        # reference_path was picked. Pass-through otherwise (the
        # combobox default state is empty string — see MatcheringStage).
        mcfg = self._cfg.get("matchering", {})
        if mcfg.get("enabled", False) and mcfg.get("reference_path"):
            _check_cancel(cancel_check)
            # T-7-01 — resolve and validate the reference path BEFORE
            # the expensive matchering call. is_one_off is set by the
            # picker UI on Browse selection (transient — see helper).
            ref_path = _validate_reference_path(
                str(mcfg["reference_path"]),
                bool(mcfg.get("is_one_off", False)),
            )
            # Per-render temp dir under cache/mastered/tmp/<hex8>.
            # apply_matchering wipes it in its finally block (T-7-04).
            temp_dir = mastered_cache_dir() / "tmp" / uuid.uuid4().hex[:8]
            matchered_tmp = temp_dir / "matchered.wav"
            audio_final = apply_matchering(
                target_audio=audio_final,
                sr=sr,
                reference_path=ref_path,
                out_path=matchered_tmp,
                temp_dir=temp_dir,
            )

        # (5a) Ending FX tail — Phase 07.1. Appended AFTER matchering so the
        # ring-out is included in the LUFS measurement + final ISP/normalize.
        # The cancel poll before the call honors the existing between-stages
        # cancellation contract; apply_ending_fx itself is uncancellable
        # mid-call (like matchering — RESEARCH §Pitfall 5). DISABLED / absent
        # path does NOT call apply_ending_fx, so the off path stays
        # array-identical to today's pipeline (SC-6 no-regression).
        efx = self._cfg.get("ending_fx", {})
        if efx.get("enabled", False):
            _check_cancel(cancel_check)
            audio_final = apply_ending_fx(audio_final, sr, efx)
            self._emit_stage_progress(96)

        # (5b) Loudness tail — quick-260623-l7l absolute LUFS target. Applied
        # as the FINAL loudness step (after matchering) when enabled, then an
        # ISP true-peak verification pass scales DOWN if the upward target gain
        # overshot the ceiling. Decision #2: the upstream relative
        # apply_lufs_makeup call (step 3) is left UNCHANGED — when loudness is
        # ON the absolute target re-measures and re-gains regardless, so the
        # earlier makeup is harmless and the OFF path stays byte-identical.
        lcfg = self._cfg.get("loudness", {})
        loudness_enabled = bool(lcfg.get("enabled", False))
        if loudness_enabled:
            _check_cancel(cancel_check)
            # Use the limiter ceiling even when the limiter stage is disabled
            # (decision #3) — it is the canonical true-peak budget.
            ceiling = float(self._cfg.get("limiter", {}).get("ceiling_dbtp", -1.0))
            audio_final = normalize_to_lufs_target(
                audio_final, sr, float(lcfg.get("target_lufs", -14.0))
            )
            audio_final = run_isp_verification(audio_final, sr, ceiling_dbtp=ceiling)
            self._emit_stage_progress(97)

        # (6) Normalize tail — quick-260621-gfq FINAL stage. Applied per
        # keeper AFTER limiter + LUFS makeup + ISP verification + matchering
        # via the pure normalize_array (DC-remove + peak→target_db). When
        # disabled (or the key is absent) this is a no-op, so the mastered
        # output stays array-identical to the pre-change pipeline. Default
        # target is 0.0 dBFS (locked decision #6).
        # quick-260623-l7l — when loudness is ON it OWNS the final level, so
        # the trailing peak-Normalize is BYPASSED (the loudness target wins
        # even if normalize.enabled=True @ 0 dBFS).
        ncfg = self._cfg.get("normalize", {})
        if ncfg.get("enabled", False) and not loudness_enabled:
            _check_cancel(cancel_check)
            audio_final = normalize_array(
                audio_final, float(ncfg.get("target_db", 0.0))
            )
            self._emit_stage_progress(98)

        # Final dtype guard — pedalboard returns float32 but the
        # post-multiplication ops above could in principle widen if a
        # caller fed us a float64 input. Normalize to float32.
        if audio_final.dtype != np.float32:
            audio_final = audio_final.astype(np.float32, copy=False)
        return audio_final


__all__ = [
    "_STAGE_ORDER",
    "_FX_STAGES",
    "_SESSION_DEFAULTS",
    "load_session_chain_snapshot",
    "config_hash",
    "run_pre_limiter_stages",
    "run_dsp_chain",
    "MasteringChain",
]
