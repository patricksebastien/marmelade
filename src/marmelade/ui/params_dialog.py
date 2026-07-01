"""Generic modal QDialog that introspects a ``parameters()`` dict to build a UI.

Phase 7 D-02 — refactored from the Phase 6
``src/marmelade/ui/heatmap_params_dialog.py`` module so the same widget
code path can be reused by any caller exposing a ``dict[str, Param]``
(heatmaps + mastering stages). Phase 6's :class:`HeatmapParamsDialog`
becomes a 3-line backward-compat alias for this class.

Public surface (D-02 contract):

    * Constructor signature ``(title, params, current_values, parent)`` is
      preserved verbatim from Phase 6 so existing callers — the
      MainWindow's ``_on_heatmap_parameters_requested`` slot and the
      Phase 6 test suite — continue to work without modification.

    * **New extension** (Phase 7 Plan 07-02 Task 1): optional
      ``add_left_button`` keyword-only argument — a
      ``(label, callback)`` tuple. When provided, a left-aligned
      QPushButton is inserted next to the QDialogButtonBox with the
      given label, wired to the given callback. Phase 6 callers do not
      pass this — behavior is unchanged for them.

    * **New extension** (Phase 7 Plan 07-02 Task 1): when a choice-kind
      :class:`Param` has a non-None ``browse_filter`` attribute (a Qt
      file-filter string, e.g. ``"Audio (*.wav *.flac)"``), the choice
      widget gets a companion "Browse..." QPushButton that opens
      :func:`QFileDialog.getOpenFileName` rooted at
      ``~/Music/Marmelade/References/``. The picked file is appended
      to the combobox + selected. The reader uses ``getattr(p,
      "browse_filter", None)`` so old Param descriptors (without the
      Phase-7-Plan-5 Matchering extension) keep working — combobox
      renders with no Browse button.

Apply / Cancel / Reset come from the existing QDialogButtonBox shape:

    * **Apply** — closes with ``QDialog.Accepted``. The caller reads
      :meth:`accepted_values`, diffs vs. the values at open time, and
      dispatches (Phase 6 — Smart Apply per D-06; Phase 7 — sidecar
      write + per-keeper MasteringRunnable).

    * **Cancel** — closes with ``QDialog.Rejected``. No persistence.

    * **Reset** — in-dialog only. Sets every widget to its
      ``Param.default`` (HM-07h — does NOT close, does NOT persist).

The dialog is Qt-bound (D-09 / D-15 N-3 tier separation). The Qt-free
:class:`Param` descriptor lives at
:mod:`marmelade.heatmaps.base` so heatmap + mastering-stage
algorithms remain unit-testable without a ``QApplication``.

The float-widget bidirectional sync uses a ``step / 10`` distance guard
to break the slider/spinbox feedback loop (RESEARCH Risk #7 / T-06-04).
"""

from __future__ import annotations

import os
from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from marmelade.audio.mastering.params import Param


class ParamsDialog(QDialog):
    """Modal popup that introspects a parameters() dict to build a UI.

    D-02 generalization — accepts any ``dict[str, Param]`` regardless of
    source. Phase 6 callers (Heatmap.parameters()) and Phase 7 callers
    (MasteringStage.parameters()) share the same widget code path.

    Args:
        title: Window title — typically ``f"{algo.display_name} parameters"``.
        params: The algorithm's ``parameters()`` dict (keyed by Param.name).
        current_values: Pre-populated values (typically from QSettings or a
            keeper.mastering[stage] dict). Missing keys fall back to
            ``Param.default``.
        parent: Optional parent QWidget.
        add_left_button: Optional ``(label, callback)`` tuple. When
            provided, a left-aligned QPushButton is inserted next to the
            QDialogButtonBox with the given label, wired to the given
            callback. Phase 7 Plan 07-02 Task 1 — used by
            MasteringDialog for the "Reset to session chain" button.

    Public API:
        :meth:`accepted_values` — read widget values into a dict after
        :data:`QDialog.Accepted`.
    """

    def __init__(
        self,
        title: str,
        params: dict[str, Param],
        current_values: dict[str, object],
        parent: QWidget | None = None,
        *,
        add_left_button: tuple[str, Callable[[], None]] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self._params: dict[str, Param] = params
        # Name -> widget for value extraction. For float kind the widget is
        # a QWidget container holding a QSlider + QDoubleSpinBox pair; the
        # QDoubleSpinBox is the canonical value source (slider is decorative).
        self._widgets: dict[str, QWidget] = {}

        form = QFormLayout()
        for name, p in params.items():
            current = current_values.get(name, p.default)
            widget = self._build_widget_for(p, current)
            self._widgets[name] = widget
            label_text = f"{p.label}{f' ({p.unit})' if p.unit else ''}"
            label = QLabel(label_text)
            if p.description:
                label.setToolTip(p.description)
            form.addRow(label, widget)

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Apply
            | QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.Reset
        )
        button_box.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(
            self.accept
        )
        button_box.button(QDialogButtonBox.StandardButton.Cancel).clicked.connect(
            self.reject
        )
        button_box.button(QDialogButtonBox.StandardButton.Reset).clicked.connect(
            self._reset_to_defaults
        )

        outer = QVBoxLayout(self)
        outer.addLayout(form)

        # Phase 7 Plan 07-02 Task 1 — bottom button row composes an
        # optional left-aligned auxiliary button (e.g. MasteringDialog's
        # "Reset to session chain") followed by the standard QDialogButtonBox
        # on the right. Wrapped in a QHBoxLayout so we can place widgets
        # at either edge.
        if add_left_button is not None:
            label, callback = add_left_button
            bottom = QHBoxLayout()
            self._left_button = QPushButton(label)
            self._left_button.clicked.connect(callback)
            bottom.addWidget(self._left_button, 0)
            bottom.addStretch(1)
            bottom.addWidget(button_box, 0)
            outer.addLayout(bottom)
        else:
            outer.addWidget(button_box)

    # ----------------------------------------------------------- builders

    def _build_widget_for(self, p: Param, current: object) -> QWidget:
        """Dispatch by ``Param.kind``. Raises ValueError on unknown kinds."""
        if p.kind == "float":
            return self._build_float_widget(p, float(current))
        if p.kind == "int":
            sb = QSpinBox()
            sb.setRange(int(p.min), int(p.max))
            if p.step:
                sb.setSingleStep(int(p.step))
            sb.setValue(int(current))
            return sb
        if p.kind == "bool":
            cb = QCheckBox()
            cb.setChecked(bool(current))
            return cb
        if p.kind == "choice":
            return self._build_choice_widget(p, str(current))
        raise ValueError(f"Unknown Param.kind: {p.kind!r}")

    def _build_choice_widget(self, p: Param, current: str) -> QWidget:
        """Build a QComboBox; if ``Param.browse_filter`` is set, add Browse button.

        Phase 7 Plan 07-02 Task 1 — Matchering reference-picker plumbing
        is owned by Plan 07-05 (the ``browse_filter`` attribute on Param
        is added there). This dialog reads via ``getattr(p,
        "browse_filter", None)`` so the absence of the attribute is
        harmless. The Browse... button opens QFileDialog at the
        Matchering reference library directory.

        Phase 7 Plan 07-05 — when ``browse_filter`` is set AND the
        ``choices`` tuple is empty / placeholder-only (i.e.
        ``len(p.choices) <= 1``), an inline empty-state guidance label
        is rendered below the combobox per UI-SPEC §"Matchering
        reference picker" line 392, and the combobox is disabled.
        Generic — activates for any choice-kind Param with a
        ``browse_filter`` and a placeholder-only choices tuple (the
        Matchering picker is the first user but the pattern is reusable).
        """
        cb = QComboBox()
        cb.addItems(list(p.choices))  # type: ignore[arg-type]
        cb.setCurrentText(current)

        browse_filter = getattr(p, "browse_filter", None)
        if browse_filter is None:
            return cb

        # Composite widget: combobox + Browse... button in an HBox + the
        # optional empty-state guidance label stacked below in a VBox.
        container = QWidget()
        outer = QVBoxLayout(container)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(cb, 1)
        browse_btn = QPushButton("Browse...")

        def _on_browse() -> None:
            ref_dir = os.path.expanduser("~/Music/Marmelade/References/")
            picked, _ = QFileDialog.getOpenFileName(
                self,
                "Pick reference",
                ref_dir,
                browse_filter,
            )
            if picked:
                cb.addItem(picked)
                cb.setCurrentText(picked)

        browse_btn.clicked.connect(_on_browse)
        row.addWidget(browse_btn, 0)
        outer.addLayout(row)

        # Phase 7 Plan 07-05 — inline empty-state guidance label.
        # Activates when choices is the placeholder-only ``("",)``
        # tuple (Matchering picker case when the library dir is empty).
        # The label wraps so it fits the dialog's min-width gracefully.
        choices = p.choices or ()
        if len(choices) <= 1:
            guidance = QLabel(
                "Drop pro-mastered reference tracks (WAV or FLAC) into "
                "~/Music/Marmelade/References/ — reload to refresh."
            )
            guidance.setWordWrap(True)
            guidance.setStyleSheet("color: #9CA3AF;")  # secondary color
            outer.addWidget(guidance)
            cb.setEnabled(False)

        # Expose the combobox as the canonical value source — accepted_values()
        # reads through container._combo when the container shape is used.
        container._combo = cb  # type: ignore[attr-defined]
        return container

    def _build_float_widget(self, p: Param, current: float) -> QWidget:
        """QSlider + QDoubleSpinBox synced pair — ``Param.step`` drives both.

        The slider operates in integer "ticks"; the spinbox is the float
        value source. Bidirectional sync uses a ``step / 10`` distance
        guard to break the feedback loop (RESEARCH Risk #7 / T-06-04 —
        without the guard the round-trip slider->spin->slider would
        ping-pong endlessly).
        """
        container = QWidget()
        h = QHBoxLayout(container)
        h.setContentsMargins(0, 0, 0, 0)

        slider = QSlider(Qt.Orientation.Horizontal)
        spin = QDoubleSpinBox()

        step = float(p.step) if p.step else (float(p.max) - float(p.min)) / 100.0
        n_ticks = int(round((float(p.max) - float(p.min)) / step))
        slider.setRange(0, n_ticks)
        spin.setRange(float(p.min), float(p.max))
        spin.setSingleStep(step)
        spin.setDecimals(3)
        spin.setValue(current)
        slider.setValue(int(round((current - float(p.min)) / step)))

        # Bi-directional sync — `step / 10` distance guard breaks the
        # feedback loop. Without it slider.valueChanged -> _slider_to_spin
        # -> spin.setValue -> spin.valueChanged -> _spin_to_slider ->
        # slider.setValue would recurse on every drag.
        def _slider_to_spin(v: int) -> None:
            new = float(p.min) + v * step
            if abs(new - spin.value()) > step / 10:
                spin.setValue(new)

        def _spin_to_slider(v: float) -> None:
            tick = int(round((v - float(p.min)) / step))
            if slider.value() != tick:
                slider.setValue(tick)

        slider.valueChanged.connect(_slider_to_spin)
        spin.valueChanged.connect(_spin_to_slider)
        h.addWidget(slider, 1)
        h.addWidget(spin, 0)

        # Stash the spin as the value-source — slider is decorative.
        container._spin = spin  # type: ignore[attr-defined]
        return container

    # ----------------------------------------------------------- reset

    def _reset_to_defaults(self) -> None:
        """Reset every widget to its ``Param.default``. In-dialog only — HM-07h.

        Does NOT call ``self.accept()`` or ``self.reject()`` (the dialog
        stays open). Does NOT touch QSettings (the caller is responsible
        for persistence on Apply).
        """
        for name, p in self._params.items():
            self._set_widget_value(self._widgets[name], p, p.default)

    def _set_widget_value(self, w: QWidget, p: Param, v: object) -> None:
        """Write ``v`` into the widget per its kind."""
        if p.kind == "float":
            w._spin.setValue(float(v))  # type: ignore[attr-defined]
        elif p.kind == "int":
            w.setValue(int(v))  # type: ignore[union-attr]
        elif p.kind == "bool":
            w.setChecked(bool(v))  # type: ignore[union-attr]
        elif p.kind == "choice":
            # Container-with-combo (browse_filter set) vs bare combobox.
            combo = getattr(w, "_combo", None)
            if combo is not None:
                combo.setCurrentText(str(v))
            else:
                w.setCurrentText(str(v))  # type: ignore[union-attr]

    # ----------------------------------------------------------- read-back

    def accepted_values(self) -> dict[str, object]:
        """Read widget values into a dict. Call after :data:`QDialog.Accepted`."""
        out: dict[str, object] = {}
        for name, p in self._params.items():
            w = self._widgets[name]
            if p.kind == "float":
                out[name] = float(w._spin.value())  # type: ignore[attr-defined]
            elif p.kind == "int":
                out[name] = int(w.value())  # type: ignore[union-attr]
            elif p.kind == "bool":
                out[name] = bool(w.isChecked())  # type: ignore[union-attr]
            elif p.kind == "choice":
                # Container-with-combo (browse_filter set) vs bare combobox.
                combo = getattr(w, "_combo", None)
                if combo is not None:
                    out[name] = str(combo.currentText())
                else:
                    out[name] = str(w.currentText())  # type: ignore[union-attr]
        return out


__all__ = ["ParamsDialog"]
