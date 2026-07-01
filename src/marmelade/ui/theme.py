"""Theme bootstrap — Fusion style, QSS, and PyQtGraph config options.

UI-SPEC §Cross-Platform Notes mandates ``Fusion`` style as base widget style on
all three platforms. RESEARCH Pitfalls #7 and #8 mandate the explicit
``leftButtonPan=True`` and ``antialias=False`` PyQtGraph config options. The
single source of truth for both lives here.

Plan 02-01: the PyQtGraph ``setConfigOption`` calls now run at MODULE IMPORT
TIME (not inside ``apply_theme``) because RESEARCH §Pitfall #9 requires the
``imageAxisOrder='row-major'`` flip to be in effect BEFORE any ``ImageItem``
is constructed anywhere in the app — including in tests that build widgets
directly without going through ``app.py``'s ``apply_theme()`` bootstrap.
``app.setStyle("Fusion")`` and ``app.setStyleSheet(...)`` still live inside
``apply_theme`` because they require a live ``QApplication`` instance.
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

import pyqtgraph as pg
from PySide6.QtWidgets import QApplication

# Public — packaged QSS resource path (read-only; T-01-02 mitigation).
QSS_PATH: Path = Path(str(files("marmelade.resources").joinpath("app.qss")))


# ============================================================================
# Module-load-time pyqtgraph config (Plan 02-01).
#
# These five setConfigOption calls run as a side-effect of importing this
# module. The Phase 2 imageAxisOrder pin MUST land before any ImageItem is
# constructed anywhere — including tests that build a WaveformView directly
# without invoking ``apply_theme(QApplication.instance())`` from app.py.
# theme.py is imported transitively by every entry point that needs Qt UI
# (app.py, every test fixture, MainWindow), so the import-time side effect
# is the load-bearing ordering guarantee.
# ============================================================================
pg.setConfigOption("background", "#1E1E1E")
pg.setConfigOption("foreground", "#9CA3AF")
pg.setConfigOption("antialias", False)  # Pitfall #8 — keep frame budget tight.
pg.setConfigOption("leftButtonPan", True)  # Pitfall #7 — UI-SPEC mandates pan-on-drag.
pg.setConfigOption("useOpenGL", False)  # Default; explicit for clarity.
# Phase 2 RESEARCH §Pitfall #9: imageAxisOrder must be 'row-major' BEFORE any
# ImageItem is constructed. The axis convention is baked at construction time;
# changing it later has no effect on already-built ImageItems. Row-major matches
# numpy's default memory layout and is the faster ingestion path.
# [PyQtGraph 0.13.1 ImageItem docs]
pg.setConfigOption("imageAxisOrder", "row-major")


def apply_theme(app: QApplication) -> None:
    """Apply Fusion style and the dark QSS to the given ``QApplication``.

    The pyqtgraph ``setConfigOption`` calls already ran at module-import time
    (see the module docstring) so widgets constructed after this import will
    pick up the pinned configuration. ``apply_theme`` is still called from
    ``app.py`` to apply the Qt widget style and stylesheet which require a
    live ``QApplication``.

    Calling this function is idempotent — subsequent calls re-apply the same
    style and stylesheet without side effects.
    """
    # 1. Widget style baseline — Fusion is the only style that renders
    #    pixel-similar across Linux / macOS / Windows. UI-SPEC §Cross-Platform.
    app.setStyle("Fusion")

    # 2. Single dark QSS resource — colors and typography per UI-SPEC.
    qss_text = QSS_PATH.read_text(encoding="utf-8")
    app.setStyleSheet(qss_text)

    # 3. pyqtgraph config options were already pinned at module-import time
    #    above. We re-apply them here as a belt-and-suspenders against any
    #    upstream code (or test) that flipped a flag between import and now.
    pg.setConfigOption("background", "#1E1E1E")
    pg.setConfigOption("foreground", "#9CA3AF")
    pg.setConfigOption("antialias", False)  # Pitfall #8 — keep frame budget tight.
    pg.setConfigOption("leftButtonPan", True)  # Pitfall #7 — UI-SPEC mandates pan-on-drag.
    pg.setConfigOption("useOpenGL", False)  # Default; explicit for clarity.
    pg.setConfigOption("imageAxisOrder", "row-major")  # Pitfall #9.
