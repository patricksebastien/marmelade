"""Frozen :class:`Param` descriptor — the mastering/params-dialog parameter type.

Relocated from ``marmelade.heatmaps.base`` during quick-260701-muv (the
removal of the dormant TensorFlow/Essentia AI-DSP heatmap backend). ``Param``
is a generic, Qt-free declarative descriptor of a tunable parameter's UX
surface; it merely lived in the heatmap module historically. It has NOTHING
to do with the deleted AI/DSP heatmaps — it is consumed by the entire
mastering subsystem (``mastering.base``, the 14 stage classes) and by the
``ParamsDialog`` / Matchering reference picker in the UI tier. This module is
its new canonical, stdlib-only home.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Param:
    """Declarative descriptor for one tunable parameter on a Heatmap subclass.

    Phase 6 — D-01/D-02/D-03. The :class:`HeatmapParamsDialog` consumer in
    ``ui/heatmap_params_dialog.py`` introspects this descriptor to build
    the corresponding widget (``QDoubleSpinBox`` + ``QSlider`` for float,
    ``QSpinBox`` for int, ``QCheckBox`` for bool, ``QComboBox`` for
    choice). The dataclass is frozen because it is a static declaration
    of the parameter's UX surface — the runtime VALUE lives in
    QSettings, not on this object.

    Attributes:
        name: Stable identifier (used as the dict key returned by
            :meth:`Heatmap.parameters` AND as the QSettings persistence
            key under ``heatmaps/<heatmap_name>/<param_name>``).
        label: Human-readable label shown next to the widget.
        kind: Discriminator over widget type. The dialog raises
            ``ValueError`` on unknown kinds.
        default: Initial value AND the target the Reset button restores.
        requires_recompute: D-04/D-06 Smart Apply dispatch flag.
            ``False`` (Energy thresholds only) — re-banding uses cached
            values + new thresholds.
            ``True`` (every other tunable) — change feeds into the
            algorithm body; D-08 reuse of the per-row Recompute path.
        min: Required for ``kind in ("float", "int")``; omitted otherwise.
        max: Required for ``kind in ("float", "int")``; omitted otherwise.
        step: Optional slider/spinbox step. Drives the float widget's
            slider-tick count via ``(max - min) / step``.
        choices: Required for ``kind == "choice"``; omitted otherwise.
        unit: Optional display label suffix (``"dB"``, ``"%"``, ``"s"``).
        description: Optional tooltip text shown on hover.
        browse_filter: Optional file dialog filter (e.g.
            ``"Audio files (*.wav *.flac);;All files (*)"``). When set,
            :class:`ParamsDialog` renders a Browse button next to the
            choice combobox; clicking it opens a QFileDialog at the
            Matchering reference library directory. Phase 7 Plan 07-05
            added this for the Matchering reference picker; Phase 6
            callers leave it at ``None`` and get the unchanged
            combobox-only widget.

    Validation rules in :meth:`__post_init__`:
        * Non-empty ``name`` AND non-empty ``label`` → ``ValueError`` otherwise.
        * ``kind in ("float", "int")`` requires both ``min`` and ``max``
          AND ``min < max`` → ``ValueError`` otherwise.
        * ``kind == "choice"`` requires non-empty ``choices`` AND
          ``default in choices`` → ``ValueError`` otherwise.
        * ``browse_filter`` is NOT validated — any string is accepted.
          The dataclass doesn't know about Qt and can't usefully check
          the filter syntax; that's QFileDialog's job.
    """

    name: str
    label: str
    kind: Literal["float", "int", "bool", "choice"]
    default: float | int | bool | str
    requires_recompute: bool
    min: float | int | None = None
    max: float | int | None = None
    step: float | int | None = None
    choices: tuple[str, ...] | None = None
    unit: str = ""
    description: str = ""
    # Phase 7 Plan 07-05 — Matchering reference picker plumbing. Optional
    # Qt file-filter string for choice-kind Params. None = no Browse
    # button (Phase 6 callers all use this default).
    browse_filter: str | None = None

    def __post_init__(self) -> None:
        # Non-empty identifiers — mirrors HeatmapResult discipline.
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(
                f"Param.name must be a non-empty str, got {self.name!r}"
            )
        if not isinstance(self.label, str) or not self.label:
            raise ValueError(
                f"Param.label must be a non-empty str, got {self.label!r}"
            )
        # Kind-specific bounds invariants.
        if self.kind in ("float", "int"):
            if self.min is None or self.max is None:
                raise ValueError(
                    f"Param {self.name!r}: kind={self.kind!r} requires "
                    f"min and max (got min={self.min!r}, max={self.max!r})"
                )
            if self.min >= self.max:
                raise ValueError(
                    f"Param {self.name!r}: min ({self.min}) must be < "
                    f"max ({self.max})"
                )
        if self.kind == "choice":
            if not self.choices:
                raise ValueError(
                    f"Param {self.name!r}: kind='choice' requires a "
                    "non-empty choices tuple"
                )
            if not isinstance(self.default, str) or self.default not in self.choices:
                raise ValueError(
                    f"Param {self.name!r}: default {self.default!r} not in "
                    f"choices {self.choices!r}"
                )
