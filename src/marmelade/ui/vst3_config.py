"""Interactive VST3 configuration flow (quick-260625; quick-260626-fw2).

Thin Qt glue shared by any surface hosting a VST3 mastering-stage row: pick a
``.vst3`` file (if none chosen yet), then open the plugin's NATIVE editor so
the user dials it in (oXygen's Master Assistant etc. are GUI-driven), and
capture the plugin's opaque state into the stage config so the render is
deterministic.

The native editor is hosted OUT-OF-PROCESS (quick-260626-fw2): the editor's
native call blocks its host thread's event loop until the window closes, so
we run it in a separate worker process
(:mod:`marmelade.ui.vst3_editor_worker`) launched via a non-blocking
``QProcess``. :func:`configure_vst3` returns immediately after starting the
process; the cfg is mutated and the optional ``on_done`` callback fires later
from the QProcess ``finished`` handler, keeping Marmelade's Qt loop responsive
while the editor is open.

The pure, testable parts (loading + state capture) live in
:mod:`marmelade.audio.mastering.stages.vst3`; the worker's load/capture seam is
unit-tested in ``tests/unit/ui/test_vst3_editor_worker.py``. This module is the
Qt glue (file dialog + QProcess lifecycle) and is not unit tested headlessly.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QProcess
from PySide6.QtWidgets import (
    QFileDialog,
    QMessageBox,
    QWidget,
)


def _pick_vst3_path(parent: QWidget) -> str:
    """Open a platform-correct picker for a ``.vst3`` and return the path.

    A ``.vst3`` is a DIRECTORY bundle on Linux/macOS (the user must select the
    folder) but a single file on Windows. ``getOpenFileName`` cannot select a
    folder, so on the bundle platforms we use ``getExistingDirectory``. Returns
    "" if the user cancels.
    """
    if sys.platform.startswith("win"):
        chosen, _ = QFileDialog.getOpenFileName(
            parent,
            "Choose a VST3 plugin",
            "",
            "VST3 plugins (*.vst3);;All files (*)",
        )
        return chosen
    # Linux / macOS: the .vst3 bundle is a directory — pick the folder itself.
    return QFileDialog.getExistingDirectory(
        parent,
        "Choose a VST3 plugin bundle (the .vst3 folder)",
        "",
        QFileDialog.Option.ShowDirsOnly,
    )


def configure_vst3(
    parent: QWidget,
    cfg: dict,
    on_done: Callable[[bool], None] | None = None,
    on_started: Callable[[], None] | None = None,
) -> bool:
    """Pick a ``.vst3`` (if needed) and host its editor out-of-process.

    Launches :mod:`marmelade.ui.vst3_editor_worker` via a non-blocking
    ``QProcess`` and returns ``True`` as soon as the process is STARTED (or
    ``False`` immediately on cancel / a missing-file precondition). The actual
    cfg mutation (``plugin_path`` / ``plugin_name`` / ``state_b64`` /
    ``enabled``) and the optional ``on_done(changed)`` callback happen LATER,
    from the QProcess ``finished`` handler — so the caller must not assume the
    cfg has changed when this function returns.

    ``on_done`` receives ``True`` when the editor captured new state, ``False``
    when the worker exited with an error (a QMessageBox explains the failure).

    ``on_started`` (quick-260626 close-to-commit hardening) fires synchronously
    once the editor QProcess has been started — the caller uses it to DISABLE
    the Apply / gear surfaces so the user cannot Apply (committing a stale,
    not-yet-captured cfg) while the editor is still open. It is paired with
    ``on_done``, which re-enables them after the cfg has been mutated. It is
    NOT called on the cancel / missing-file early-return paths (no process is
    started, so nothing was disabled).
    """
    path = str(cfg.get("plugin_path", "") or "")
    if not path:
        chosen = _pick_vst3_path(parent)
        if not chosen:
            return False
        path = chosen

    if not Path(path).expanduser().exists():
        QMessageBox.warning(
            parent, "VST3 not found", f"Plugin file not found:\n{path}"
        )
        return False

    # Process-private temp files for the worker's in/out state + name. mkstemp
    # gives unguessable names; the finished handler unlinks them.
    in_fd, in_state_path = tempfile.mkstemp(suffix=".vst3state.in")
    with os.fdopen(in_fd, "w", encoding="utf-8") as f:
        f.write(str(cfg.get("state_b64", "") or ""))
    out_state_fd, out_state_path = tempfile.mkstemp(suffix=".vst3state.out")
    os.close(out_state_fd)
    out_name_fd, out_name_path = tempfile.mkstemp(suffix=".vst3name.out")
    os.close(out_name_fd)

    # Parent the process to `parent` so it is not GC'd while running. QProcess
    # inherits this process's environment, so pedalboard + marmelade import.
    proc = QProcess(parent)
    proc.setProgram(sys.executable)
    proc.setArguments(
        [
            "-m",
            "marmelade.ui.vst3_editor_worker",
            path,
            str(cfg.get("plugin_name", "") or ""),
            in_state_path,
            out_state_path,
            out_name_path,
        ]
    )

    def _cleanup() -> None:
        for p in (in_state_path, out_state_path, out_name_path):
            try:
                os.unlink(p)
            except OSError:
                pass

    def _on_finished(exit_code: int, _status: QProcess.ExitStatus) -> None:
        if exit_code == 0:
            try:
                state_b64 = Path(out_state_path).read_text(encoding="utf-8")
                name = Path(out_name_path).read_text(encoding="utf-8")
            except OSError:
                state_b64 = ""
                name = ""
            cfg["plugin_path"] = path
            cfg["plugin_name"] = name or str(cfg.get("plugin_name", "") or "")
            cfg["state_b64"] = state_b64
            cfg["enabled"] = True
            _cleanup()
            if on_done is not None:
                on_done(True)
        else:
            stderr = bytes(proc.readAllStandardError()).decode("utf-8", "replace")
            QMessageBox.warning(
                parent,
                "VST3 editor failed",
                stderr or "The plugin editor exited with an error.",
            )
            _cleanup()
            if on_done is not None:
                on_done(False)

    proc.finished.connect(_on_finished)
    proc.start()
    # Close-to-commit hardening (quick-260626): now that the editor process is
    # live, tell the caller to lock the Apply / gear surfaces. The matching
    # unlock happens in _on_finished's on_done call (the cfg is captured by
    # then). Done AFTER start() so a failed start (which would emit finished
    # with a non-zero code) still pairs lock→unlock correctly.
    if on_started is not None:
        on_started()
    return True
