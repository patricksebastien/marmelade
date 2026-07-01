"""On-disk JSON sidecar for region marks (Phase 3 — REG-04).

Structural 1:1 mirror of :mod:`marmelade.audio.heatmap_cache`. Pure
Python + stdlib — no GUI-toolkit imports (N-3 invariant). The audio
package stays Qt-free so its logic is unit-testable without a graphical
event loop.

Cache key is reused from :mod:`proxy_cache.cache_key` so the sidecar,
heatmap, and audio-proxy caches share invalidation when source-file
size/mtime/head/tail change (D-A3-1).

Security (mirrors heatmap_cache discipline):
    * ``sidecar_path`` requires the key to match ``^[0-9a-f]{16}$`` BEFORE
      any filesystem use (traversal guard — re-used from heatmap_cache).
    * ``load_sidecar`` validates ALL absolute bounds (region count,
      start_sec/end_sec range, note length, state enum) BEFORE returning;
      on ANY failure the file is renamed to
      ``{key}.json.corrupt-{ISO-timestamp}`` and an empty in-memory list
      is returned (D-A3-5).
    * ``save_sidecar`` writes to a ``.tmp`` sibling and ``os.replace``s
      into place atomically (D-A3-4) — mirrors heatmap_cache lines 213-225.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Re-export so the sidecar, heatmap, audio-proxy, and waveform-proxy
# caches all share invalidation when source size/mtime/head/tail change.
# Identity preserved (not wrapped) so a future tweak to the hash inputs
# propagates to all caches in lockstep (D-A3-1).
from marmelade.audio.proxy_cache import cache_key  # noqa: F401 — re-export


# Cache key shape: 16 lowercase hex chars — IDENTICAL regex to
# proxy_cache._KEY_RE / heatmap_cache._KEY_RE. The literal pattern below
# is the source-grep gate (Phase 1 §"Inline-literal grep gates").
_KEY_RE = re.compile(r"^[0-9a-f]{16}$")

# Phase 3 sidecar schema. Bump on a breaking change; readers quarantine
# on mismatch unless a migrator lands (D-A3-5).
SCHEMA_VERSION = 1

# Absolute bounds — applied BEFORE returning a parsed sidecar (mirror of
# heatmap_cache CR-03 absolute-bounds discipline).
_MAX_REGIONS = 4096  # Defensive — single source rarely > 50; loose cap.
_MAX_NOTE_LEN = 200  # D-A3-3 Claude's discretion (200-char note limit).
_VALID_STATES = frozenset({"untouched", "trash", "keeper"})

# quick-260701-jc5 (MARK-05 / T-jc5-02 DoS) — defensive cap on the number of
# markers a single sidecar may carry. A single jam rarely has more than a few
# dozen point-in-time marks; the loose cap mirrors ``_MAX_REGIONS`` and is
# enforced BEFORE constructing Marker objects so a hostile sidecar cannot
# balloon memory. The marker label reuses ``_MAX_NOTE_LEN`` (200 chars).
_MAX_MARKERS = 4096

# quick-260621-gfq — the per-keeper normalize dB target now lives inside
# ``mastering['normalize']`` and its [-60, 0] range is enforced by the
# NormalizeStage Param min/max via ``_validate_stage_param_ranges`` (no
# standalone bounds constants any more — the standalone fields were removed).


@dataclass(slots=True)
class Region:
    """A single region mark.

    Attributes:
        id: Stable identifier (UUID4 hex, 32 chars in practice — but the
            schema accepts any non-empty string for forward compat).
        start_sec: Region start as seconds from source start. Must be
            finite and ``>= 0``.
        end_sec: Region end as seconds from source start. Must be finite
            and ``> start_sec``.
        state: One of ``_VALID_STATES`` (``"untouched"``, ``"trash"``,
            ``"keeper"``). Defaults to ``"untouched"`` — newly drawn
            regions enter the untouched state and are marked explicitly
            after.
        created_at: ISO 8601 timestamp (informational; not parsed).
        note: Free-text note. Bounded to ``_MAX_NOTE_LEN`` chars at the
            schema level.
        mastering: Phase 7 D-19 — optional per-keeper mastering chain
            configuration. ``None`` (the default) means "no mastering;
            export uses source proxy". A dict means "mastered cache
            applies". Keys are a subset of
            ``marmelade.audio.mastering.chain._STAGE_ORDER``; each
            value is a per-stage config dict with at least
            ``enabled: bool`` plus stage-specific parameter overrides.
            The serializer OMITS the key when ``mastering is None`` so
            pre-Phase-7 readers never see it.
        youtube_video_id: Phase 8 D-30 — optional YouTube video ID set
            after a successful upload. Typically the 11-char public ID
            YouTube assigns (e.g. ``"dQw4w9WgXcQ"``). ``None`` (the
            default) means "never uploaded". Additive — pre-Phase-8
            sidecars deserialize with ``youtube_video_id=None``; the
            serializer omits the key when ``None`` so older readers
            never see it. Identical pattern to the Phase 7 ``mastering``
            additive field.
    """

    id: str
    start_sec: float
    end_sec: float
    state: str = "untouched"
    created_at: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )
    note: str = ""
    # Phase 7 Plan 07-02 Task 1 — additive (D-19). Slots-compatible
    # because it's appended at end of the dataclass with a safe default.
    # The serializer omits this key from JSON when None; the validator
    # accepts both missing and ``None``.
    mastering: dict | None = None
    # Phase 8 Plan 08-01 Task 2 — additive (D-30). Slots-compatible
    # because it's appended at end of the dataclass with a safe default.
    # The serializer omits this key from JSON when None; the validator
    # accepts both missing and ``None``. Set by
    # ``YouTubeUploadRunnable.signals.finished`` →
    # ``MainWindow._on_youtube_upload_finished`` (wired in Plan 08-04).
    youtube_video_id: str | None = None
    # quick-260621-gfq — the standalone ``normalize_enabled`` /
    # ``normalize_target_db`` fields were REMOVED. Per-keeper normalize is now
    # the FINAL mastering-chain stage; its single source of truth is
    # ``mastering['normalize'] = {'enabled': bool, 'target_db': float}``
    # (default 0.0 dBFS). Legacy sidecars carrying the old top-level keys are
    # migrated into ``mastering['normalize']`` at load time (see
    # ``_validate_payload``).


@dataclass(slots=True)
class Marker:
    """A single point-in-time marker (quick-260701-jc5 — MARK-01).

    Mirrors :class:`Region`'s style (``slots=True``, ISO ``created_at`` via a
    ``default_factory``). Markers are a lightweight point-in-time complement to
    regions: a user drops one at the current playhead position via the "m" key
    or the Markers-panel [+] button.

    Attributes:
        id: Stable identifier (UUID4 hex in practice; schema accepts any
            non-empty string for forward compat — same discipline as Region).
        time_sec: Marker position as seconds from source start. Must be finite
            and ``>= 0`` (T-jc5-04 rejects NaN/inf; T-jc5-01 rejects negatives).
        label: Free-text label shown on the panel row + the waveform line.
            Bounded to ``_MAX_NOTE_LEN`` chars at the schema level (T-jc5-03).
        created_at: ISO 8601 timestamp (informational; not parsed).
    """

    id: str
    time_sec: float
    label: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )


class SidecarValidationError(ValueError):
    """Raised internally by :func:`_validate_payload` and caught by
    :func:`load_sidecar` to trigger quarantine.

    Callers never see this — :func:`load_sidecar` never raises.
    """


def sidecar_path(cache_root: Path, key: str) -> Path:
    """Return ``cache_root / 'sidecars' / f'{key}.json'``.

    Traversal-guard mitigation (T-03-01-01): ``key`` MUST match
    ``^[0-9a-f]{16}$`` BEFORE any filesystem use. Anything else raises
    :class:`ValueError`. Cache filenames can ONLY be 16-char lowercase
    hex digests; user-supplied path components never reach the file
    system.

    Does NOT create the directory — :func:`save_sidecar` does that
    immediately before writing.
    """
    if not _KEY_RE.match(key):
        raise ValueError(f"Invalid cache key: {key!r}")
    return Path(cache_root) / "sidecars" / f"{key}.json"


def save_sidecar(
    path: str | os.PathLike,
    regions: list[Region],
    markers: list[Marker] | None = None,
) -> None:
    """Write the sidecar JSON file atomically.

    Atomic write (D-A3-4 — mirrors heatmap_cache.write_heatmap lines
    197-225): build at ``<path>.tmp`` and ``os.replace()`` into place. A
    reader of the cache directory never observes a half-written
    ``{key}.json``. Exception handler unlinks any partial ``.tmp`` before
    re-raising so a crashed write does not leak the sibling.

    Creates the parent directory (typically ``cache_root / 'sidecars'``)
    lazily on first save — no caller has to mkdir first.

    quick-260701-jc5 (MARK-05) — ``markers`` is additive exactly like the
    existing ``mastering`` / ``youtube_video_id`` region fields, so
    ``SCHEMA_VERSION`` is NOT bumped. It defaults to an empty list so legacy
    region-only callers keep working during migration; MainWindow passes the
    live marker list explicitly. The ``"markers"`` array is written alongside
    ``"regions"`` in the same atomic payload.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")

    # Phase 7 Plan 07-02 Task 1 (D-19) — omit ``mastering`` key from
    # serialized JSON when the in-memory value is None. Pre-Phase-7
    # readers must not see a new key; the field is purely additive.
    # Phase 8 Plan 08-01 Task 2 (D-30) — same omit-when-None discipline
    # for ``youtube_video_id``; pre-Phase-8 readers never see the key
    # when no upload has happened.
    serialized_regions: list[dict] = []
    for r in regions:
        d = asdict(r)
        if d.get("mastering") is None:
            d.pop("mastering", None)
        if d.get("youtube_video_id") is None:
            d.pop("youtube_video_id", None)
        # quick-260621-gfq — no standalone normalize keys to omit any more;
        # normalize lives inside the already-serialized ``mastering`` dict.
        serialized_regions.append(d)
    # quick-260701-jc5 (MARK-05) — serialize markers into a "markers" array.
    # asdict on the slots dataclass yields {id, time_sec, label, created_at}.
    serialized_markers = [asdict(m) for m in (markers or [])]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "regions": serialized_regions,
        "markers": serialized_markers,
    }
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            # WR-01 — fsync BEFORE close so the kernel commits user
            # buffers to the disk's write cache before os.replace runs.
            # Without this, a power loss between f.close() and the
            # journal commit of the rename can leave a 0-byte file at
            # `p`, which then quarantines on next load — losing region
            # marks. The user's region marks are the only durable data
            # Phase 3 ships; the ~1 ms sync cost is invisible since saves
            # are debounced via editingFinished / dragFinished.
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
    except Exception:
        # BL-02 — narrow from BaseException to Exception so a
        # KeyboardInterrupt landing inside os.remove (or anywhere mid-
        # handler) cannot re-enter the cleanup path and corrupt the
        # cache directory. Also catch OSError on cleanup (not just
        # FileNotFoundError) so a Windows PermissionError on an
        # unclosed handle, or an ENOTDIR from a permission-failed
        # parent.mkdir, does not surface as a misleading secondary
        # error masking the original write failure.
        try:
            os.remove(str(tmp))
        except OSError:
            pass
        raise


def load_sidecar(
    path: str | os.PathLike,
) -> tuple[list[Region], list[Marker]]:
    """Return ``(regions, markers)``, or quarantine + return ``([], [])`` on
    any failure.

    quick-260701-jc5 (MARK-05) — the return shape is now a 2-tuple. An OLD
    regions-only sidecar (no ``"markers"`` key) loads with an empty markers
    list and its regions round-trip unchanged (backward compat).

    D-A3-5 — quarantine on load failure. Failure modes that trigger
    quarantine:
      * JSON parse error
      * schema_version > SCHEMA_VERSION (no forward-compat migrator yet)
      * Missing required fields on the top-level object
      * Any region failing bounds validation (start/end ordering, state
        enum, note length, region count, finite floats)
      * Underlying ``OSError`` (e.g., permission error reading)

    On quarantine: rename ``{key}.json`` to
    ``{key}.json.corrupt-{YYYYMMDD}T{HHMMSS}{microseconds}`` (preserves
    forensic data per D-A3-5 with microsecond precision so two
    corruptions in the same second produce distinct filenames per W-8),
    and return an empty list. NEVER raises — sidecar load must not block
    file-open.

    A missing file is NOT a failure — it's the "no sidecar yet" case;
    returns ``[]`` and does not create a quarantine file.
    """
    p = Path(path)
    if not p.exists():
        return [], []
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        regions, markers = _validate_payload(data)
        return regions, markers
    except (json.JSONDecodeError, SidecarValidationError, OSError):
        # Quarantine — preserve for forensic recovery (D-A3-5).
        # Microsecond precision so two corruptions in the same second
        # do not collide (W-8). ``%Y%m%dT%H%M%S%f`` is Windows-safe
        # (no colons) — full ISO format with colons is not.
        ts = datetime.now().strftime("%Y%m%dT%H%M%S%f")
        try:
            p.rename(p.with_name(f"{p.name}.corrupt-{ts}"))
        except OSError:
            pass  # best-effort — never block file open
        return [], []


def _validate_mastering_dict(mastering: Any) -> None:
    """Validate a region's ``mastering`` payload (Phase 7 Plan 07-02 Task 1 / D-19).

    Accepts ``None`` (no mastering applied) or a dict whose keys are a
    subset of :data:`marmelade.audio.mastering.chain._STAGE_ORDER`
    and whose values are dicts with at least ``enabled: bool`` plus
    optional per-stage parameter overrides. Range checks on the
    parameter values are NOT performed here — the chain orchestrator
    validates them at apply time. The job of this validator is the
    defense-in-depth inbound trust check: no attacker-controllable key
    names propagate downstream into ``config_hash`` or the cache
    filename composition (T-7-02 mitigation).

    WR-01 (Phase 7 review) — strips ``mastering.matchering.is_one_off``
    from incoming sidecars. The flag is set by the picker UI on
    in-session Browse selection but it is also persisted (by the
    MasteringDialog gear handler) into the sidecar, which weakens the
    library-containment defense in
    :func:`audio.mastering.chain._validate_reference_path`. Stripping
    it on load forces the chain to re-validate the persisted
    ``reference_path`` against the library directory; outside-library
    paths now raise ``ValueError`` at apply time and the user is
    prompted to re-Browse, exactly as the original "transient flag"
    docstring intended.

    Lazy-imports ``_STAGE_ORDER`` from ``audio.mastering.chain`` to
    avoid pulling pedalboard / numpy at sidecar-load time when the user
    does not use mastering (defensive — most existing keepers have
    ``mastering=None``).
    """
    if mastering is None:
        return
    if not isinstance(mastering, dict):
        raise SidecarValidationError(
            f"mastering must be a dict or None, got {type(mastering).__name__}"
        )
    # Lazy import to keep sidecar load cheap when not using mastering.
    from marmelade.audio.mastering.chain import _STAGE_ORDER

    allowed_stages = frozenset(_STAGE_ORDER)
    for stage_name, stage_cfg in mastering.items():
        if stage_name not in allowed_stages:
            raise SidecarValidationError(
                f"invalid mastering stage name: {stage_name!r} "
                f"(allowed: {sorted(allowed_stages)})"
            )
        if not isinstance(stage_cfg, dict):
            raise SidecarValidationError(
                f"mastering[{stage_name!r}] must be a dict, got "
                f"{type(stage_cfg).__name__}"
            )
        if "enabled" not in stage_cfg or not isinstance(
            stage_cfg["enabled"], bool
        ):
            raise SidecarValidationError(
                f"mastering[{stage_name!r}].enabled must be a bool"
            )

    # WR-01 sanitation — strip persisted is_one_off from matchering.
    # The flag is in-session UI state; a sidecar carrying it True
    # would bypass the library-containment check in
    # _validate_reference_path. Re-validation against the library
    # happens at chain apply time.
    matchering_cfg = mastering.get("matchering")
    if isinstance(matchering_cfg, dict) and "is_one_off" in matchering_cfg:
        matchering_cfg.pop("is_one_off", None)

    # WR-03 (Phase 7 review) — value-range validation per stage Param
    # descriptor (min/max declared on each MasteringStage subclass's
    # parameters()). Previously, _validate_mastering_dict only checked
    # `enabled: bool` and deferred range checks to "the chain
    # orchestrator at apply time" — but the chain just does
    # float(cfg["x"]) without range bounds, so a hostile sidecar with
    # e.g. {"compressor": {"enabled": true, "ratio": -1e100}} reached
    # pedalboard uncaught. Range checks happen here so the user gets
    # a friendly SidecarValidationError diagnostic at load instead of
    # a pedalboard crash in a worker thread.
    _validate_stage_param_ranges(mastering)


def _validate_stage_param_ranges(mastering: dict) -> None:
    """WR-03 — per-stage Param.min/max range checks.

    For each enabled stage in the mastering dict, look up the
    corresponding MasteringStage subclass, instantiate it to enumerate
    its Param descriptors, and check that every override value sits in
    ``[Param.min, Param.max]``. Keys that don't correspond to a known
    Param on the stage are tolerated (unknown keys don't affect
    pedalboard behavior — they pass through without being read).

    Lazy-imports the stage classes so the heavy pedalboard / numpy
    dependency tree is not pulled in at sidecar-load time when the
    user is not using mastering.

    The 'matchering' stage has no Param descriptors here — its
    `reference_path` validation is path-shape (not value-range) and
    happens in :func:`audio.mastering.chain._validate_reference_path`
    at apply time. We skip it.

    Phase 07.1 (07.1-03 / T-07.1-05) — the 'ending_fx' stage is included
    so its numeric params (tail_sec / wet / primary) are range-checked
    against the EndingFxStage Param min/max (and NaN/inf rejected) at LOAD
    time, before any value reaches pedalboard in the render worker. Its
    'effect_type' Param is kind="choice" (not numeric) so the numeric loop
    below skips it — the choice domain is enforced at the UI/builder layer
    (EFFECT_BUILDERS keys + the dialog combobox), and an unknown effect_type
    at render time is a graceful dry-with-tail no-op in apply_ending_fx
    (T-07.1-06 accept). The sidecar layer does NOT gate the choice domain.
    """
    # Lazy import — keeps cheap-path users out of pedalboard.
    from marmelade.audio.mastering.stages import (
        CompressorStage,
        DelayStage,
        DistortionStage,
        EndingFxStage,
        EqStage,
        FadeStage,
        HighPassStage,
        LimiterStage,
        LoudnessStage,
        LowPassStage,
        NormalizeStage,
        ReverbStage,
    )

    stage_classes = {
        "highpass": HighPassStage,
        "lowpass": LowPassStage,
        "eq": EqStage,
        "compressor": CompressorStage,
        # quick-260629 — whole-clip color stages. Range-checked at sidecar load
        # so a hostile out-of-range drive/feedback/room value never reaches
        # pedalboard in the render worker (defense-in-depth, like ending_fx).
        "distortion": DistortionStage,
        "delay": DelayStage,
        "reverb": ReverbStage,
        "limiter": LimiterStage,
        # quick-260621-gfq — target_db range [-60, 0] enforced via this map so
        # a hostile mastering.normalize.target_db can never reach 10**(db/20).
        "normalize": NormalizeStage,
        # quick-260623-l7l (T-l7l-01) — target_lufs range [-30, -6] enforced
        # here so a hostile mastering.loudness.target_lufs can never reach
        # 10**(gain_db/20) amplification.
        "loudness": LoudnessStage,
        # Phase 07.1 (T-07.1-05) — tail_sec [0.5, 12], wet [0, 1], primary
        # [0, 1] range-checked + NaN/inf rejected here so a hostile
        # mastering.ending_fx value never reaches pedalboard. effect_type is
        # kind="choice" → skipped by the numeric loop (T-07.1-06 accept).
        "ending_fx": EndingFxStage,
        # quick-260626-o9y (T-o9y-01) — fade.duration_sec range-checked to
        # [0.0, 10.0] + NaN/inf rejected here so a hostile sidecar can never
        # push a huge fade duration into the export fade math.
        "fade": FadeStage,
    }
    for stage_name, stage_cls in stage_classes.items():
        stage_cfg = mastering.get(stage_name)
        if not isinstance(stage_cfg, dict):
            continue
        # Range checks fire regardless of enabled/disabled — a hostile
        # sidecar could ship an out-of-range value with enabled=False,
        # the user could later toggle enabled=True at runtime, and the
        # bad value would still reach pedalboard. Validate eagerly.
        try:
            params = stage_cls().parameters()
        except Exception:
            # If a stage class fails to instantiate (shouldn't, given
            # they're concrete and ABC-checked), skip rather than
            # quarantine the sidecar over an internal bug.
            continue
        for param_name, param_desc in params.items():
            if param_name not in stage_cfg:
                continue
            raw_val = stage_cfg[param_name]
            # Param.kind in {"float", "int", "bool", "choice"} per
            # heatmaps.base.Param. We range-check numeric kinds; non-
            # numeric kinds have their own validity domain.
            if param_desc.kind in ("float", "int"):
                try:
                    num_val = float(raw_val)
                except (TypeError, ValueError):
                    raise SidecarValidationError(
                        f"mastering[{stage_name!r}][{param_name!r}] must "
                        f"be numeric, got {type(raw_val).__name__}"
                    )
                # quick-260621-gfq (T-gfq-01) — a NaN/inf value passes
                # ``float()`` but slips past the < / > comparisons below
                # (NaN comparisons are always False). Reject non-finite
                # values explicitly so e.g. NaN target_db can never reach
                # ``10**(db/20)`` in normalize_array.
                if not math.isfinite(num_val):
                    raise SidecarValidationError(
                        f"mastering[{stage_name!r}][{param_name!r}] must "
                        f"be finite, got {num_val}"
                    )
                lo = param_desc.min
                hi = param_desc.max
                if lo is not None and num_val < float(lo):
                    raise SidecarValidationError(
                        f"mastering[{stage_name!r}][{param_name!r}]={num_val} "
                        f"is below the allowed minimum ({lo})"
                    )
                if hi is not None and num_val > float(hi):
                    raise SidecarValidationError(
                        f"mastering[{stage_name!r}][{param_name!r}]={num_val} "
                        f"exceeds the allowed maximum ({hi})"
                    )


def _validate_markers(data: Any) -> list[Marker]:
    """Validate the ``"markers"`` array of the sidecar payload.

    quick-260701-jc5 (MARK-05 + STRIDE mitigations). Missing key → empty list
    (backward compat with pre-marker sidecars). Any violation raises
    :class:`SidecarValidationError` so :func:`load_sidecar` quarantines the
    file — mirrors the exact per-field discipline used for regions.

    Per-marker rules:
        - ``id`` is a non-empty str (T-jc5-01).
        - ``time_sec`` is a finite number ``>= 0.0`` (T-jc5-01 / T-jc5-04 —
          rejects negatives, NaN, inf, and non-numeric).
        - ``label`` is a str with ``len <= _MAX_NOTE_LEN`` (T-jc5-03 DoS).
        - ``created_at`` is a str.
        - total count ``<= _MAX_MARKERS`` (T-jc5-02 DoS) — enforced BEFORE
          constructing Marker objects.
    """
    markers_raw = data.get("markers", [])
    if not isinstance(markers_raw, list):
        raise SidecarValidationError(
            f"'markers' must be a list, got {type(markers_raw).__name__}"
        )
    if len(markers_raw) > _MAX_MARKERS:
        raise SidecarValidationError(
            f"too many markers: {len(markers_raw)} > {_MAX_MARKERS}"
        )
    out: list[Marker] = []
    for m in markers_raw:
        if not isinstance(m, dict):
            raise SidecarValidationError(
                f"marker must be a dict, got {type(m).__name__}"
            )
        mid = m.get("id")
        if not isinstance(mid, str) or not mid:
            raise SidecarValidationError(f"invalid marker id: {mid!r}")
        time_sec = m.get("time_sec")
        # bool is an int subclass — reject it explicitly so a JSON `true`
        # cannot masquerade as a numeric time.
        if isinstance(time_sec, bool) or not isinstance(
            time_sec, (int, float)
        ):
            raise SidecarValidationError(
                f"marker time_sec must be a number, got "
                f"{type(time_sec).__name__}"
            )
        time_f = float(time_sec)
        if not math.isfinite(time_f):
            raise SidecarValidationError(
                f"marker time_sec must be finite, got {time_f}"
            )
        if time_f < 0.0:
            raise SidecarValidationError(
                f"marker time_sec must be >= 0.0, got {time_f}"
            )
        label = m.get("label", "")
        if not isinstance(label, str):
            raise SidecarValidationError(
                f"marker label must be a string, got {type(label).__name__}"
            )
        if len(label) > _MAX_NOTE_LEN:
            raise SidecarValidationError(
                f"marker label too long: len={len(label)} > {_MAX_NOTE_LEN}"
            )
        created_at = m.get("created_at", "")
        if not isinstance(created_at, str):
            raise SidecarValidationError(
                f"marker created_at must be a string, got "
                f"{type(created_at).__name__}"
            )
        out.append(
            Marker(
                id=mid,
                time_sec=time_f,
                label=label,
                created_at=created_at,
            )
        )
    return out


def _validate_payload(data: Any) -> tuple[list[Region], list[Marker]]:
    """Validate the parsed JSON payload and return ``(regions, markers)``.

    Raises :class:`SidecarValidationError` on any violation — the caller
    (:func:`load_sidecar`) catches and triggers quarantine.

    Schema (D-A3-3):
        * Top-level is a dict.
        * ``schema_version == SCHEMA_VERSION`` (newer → reject;
          older → reject — no migrator in Phase 3).
        * ``regions`` is a list of length ``<= _MAX_REGIONS``.
        * Per-region:
            - ``id`` is a non-empty str.
            - ``start_sec`` and ``end_sec`` are finite floats with
              ``0.0 <= start_sec < end_sec``.
            - ``state`` is in ``_VALID_STATES``.
            - ``note`` is a str with ``len <= _MAX_NOTE_LEN``.
            - ``created_at`` is a str.
        * ``markers`` (quick-260701-jc5, additive) is a list validated by
          :func:`_validate_markers`; a missing key deserializes as ``[]``.

    Does NOT clamp against source duration — the schema is
    source-agnostic; the caller's gate decides whether to reject
    regions outside the source's range.
    """
    if not isinstance(data, dict):
        raise SidecarValidationError(
            f"sidecar root must be a dict, got {type(data).__name__}"
        )
    schema_version = data.get("schema_version")
    if schema_version != SCHEMA_VERSION:
        raise SidecarValidationError(
            f"unsupported schema_version: {schema_version!r} "
            f"(expected {SCHEMA_VERSION})"
        )
    regions_raw = data.get("regions")
    if not isinstance(regions_raw, list):
        raise SidecarValidationError(
            f"'regions' must be a list, got {type(regions_raw).__name__}"
        )
    if len(regions_raw) > _MAX_REGIONS:
        raise SidecarValidationError(
            f"too many regions: {len(regions_raw)} > {_MAX_REGIONS}"
        )
    out: list[Region] = []
    for r in regions_raw:
        if not isinstance(r, dict):
            raise SidecarValidationError(
                f"region must be a dict, got {type(r).__name__}"
            )
        rid = r.get("id")
        if not isinstance(rid, str) or not rid:
            raise SidecarValidationError(f"invalid region id: {rid!r}")
        start = r.get("start_sec")
        end = r.get("end_sec")
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            raise SidecarValidationError(
                f"start_sec/end_sec must be numbers, got "
                f"{type(start).__name__}/{type(end).__name__}"
            )
        start_f = float(start)
        end_f = float(end)
        if not math.isfinite(start_f) or not math.isfinite(end_f):
            raise SidecarValidationError(
                f"start_sec/end_sec must be finite, got "
                f"start={start_f}, end={end_f}"
            )
        if not (0.0 <= start_f < end_f):
            raise SidecarValidationError(
                f"invalid range: start={start_f}, end={end_f} "
                f"(require 0.0 <= start < end)"
            )
        state = r.get("state")
        if state not in _VALID_STATES:
            raise SidecarValidationError(f"invalid state: {state!r}")
        note = r.get("note", "")
        if not isinstance(note, str):
            raise SidecarValidationError(
                f"note must be a string, got {type(note).__name__}"
            )
        if len(note) > _MAX_NOTE_LEN:
            raise SidecarValidationError(
                f"note too long: len={len(note)} > {_MAX_NOTE_LEN}"
            )
        created_at = r.get("created_at", "")
        if not isinstance(created_at, str):
            raise SidecarValidationError(
                f"created_at must be a string, got {type(created_at).__name__}"
            )
        # Phase 7 Plan 07-02 Task 1 (D-19) — optional additive field.
        # Missing key → None; present-but-invalid → SidecarValidationError
        # → load_sidecar quarantines the file. Old Phase-3-era sidecars
        # (no "mastering" key in JSON) deserialize cleanly with None.
        mastering = r.get("mastering")
        # quick-260621-gfq — migrate legacy top-level normalize keys (the
        # quick-260620-mgu shape) into ``mastering['normalize']`` BEFORE
        # validation so the migrated entry is range/bool-checked by
        # ``_validate_mastering_dict`` (T-gfq-03). The fold fires only when
        # a truthy ``normalize_enabled`` OR an explicit ``normalize_target_db``
        # is present; absent → ``mastering`` stays as-is (None for legacy
        # keepers). Default target is 0.0 (locked decision #6); a legacy
        # explicit target is preserved verbatim. The legacy raw values are
        # folded as-is so a hostile non-bool ``enabled`` / out-of-range
        # ``target_db`` still quarantines through the shared validators.
        legacy_enabled = r.get("normalize_enabled", False)
        has_legacy_target = "normalize_target_db" in r
        if legacy_enabled or has_legacy_target:
            if not isinstance(mastering, dict):
                mastering = {}
            norm_entry = mastering.setdefault("normalize", {})
            if isinstance(norm_entry, dict):
                norm_entry.setdefault("enabled", legacy_enabled)
                if has_legacy_target:
                    norm_entry.setdefault(
                        "target_db", r.get("normalize_target_db")
                    )
                else:
                    norm_entry.setdefault("target_db", 0.0)
        _validate_mastering_dict(mastering)
        # Phase 8 Plan 08-01 Task 2 (D-30) — optional additive field.
        # Missing key → None (pre-Phase-8 sidecars deserialize cleanly).
        # Present-but-not-str → SidecarValidationError → quarantine.
        # Range / shape checks on the YouTube video ID itself are
        # intentionally skipped here (YouTube may change ID format;
        # softer than the mastering range checks). T-08-01-04
        # mitigation — defense-in-depth type check.
        youtube_video_id = r.get("youtube_video_id")
        if youtube_video_id is not None and not isinstance(youtube_video_id, str):
            raise SidecarValidationError(
                f"youtube_video_id must be str or None, got "
                f"{type(youtube_video_id).__name__}"
            )
        out.append(
            Region(
                id=rid,
                start_sec=start_f,
                end_sec=end_f,
                state=state,
                created_at=created_at,
                note=note,
                mastering=mastering,
                youtube_video_id=youtube_video_id,
            )
        )
    # quick-260701-jc5 (MARK-05) — parse+validate the additive markers array
    # AFTER the regions so a marker violation quarantines the whole file too.
    markers = _validate_markers(data)
    return out, markers
