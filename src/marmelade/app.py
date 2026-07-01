"""QApplication bootstrap — Fusion style, QSS, MainWindow.

`python -m marmelade` (via ``__main__.py``) and the console script
``marmelade`` (via ``[project.scripts]``) both land in ``main()``.

Threat mitigation T-01-05: ``setOrganizationName`` / ``setApplicationName`` are
called BEFORE any ``QSettings`` or ``QStandardPaths`` access so per-user
settings always land in the Marmelade namespace.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Sequence

from PySide6.QtCore import QCoreApplication
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from marmelade.ui import theme
from marmelade.ui.main_window import MainWindow


def _apply_app_icon(app: QApplication) -> None:
    """Set the application/window icon from a bundled resource, if present.

    Looks for ``app_icon.png`` in ``marmelade.resources`` (packaged into the
    wheel like ``app.qss`` / ``baldapprouved.png``). ``QApplication.
    setWindowIcon`` sets the default icon for every top-level window — the
    title bar and the OS taskbar/dock. It is a graceful no-op until the asset
    exists, so dropping the file in later needs no code change.
    """
    try:
        from importlib.resources import files

        path = files("marmelade.resources").joinpath("app_icon.png")
        if path.is_file():
            icon = QIcon(str(path))
            if not icon.isNull():
                app.setWindowIcon(icon)
    except Exception:  # never let a cosmetic icon failure block startup
        logger.debug("app icon not applied", exc_info=True)

logger = logging.getLogger(__name__)


def main(argv: Sequence[str] | None = None) -> int:
    """Launch Marmelade. Returns the QApplication exit code.

    Args:
        argv: Argument vector. Defaults to ``sys.argv``. Tests can pass a
            stub list to avoid leaking pytest args into Qt.
    """
    args = list(argv) if argv is not None else list(sys.argv)

    # OBS-MASTER-LOG: configure root logging EARLY so logger.* calls across
    # the app (including the mastering worker's traceback logging) reach
    # stderr. Level honors JAMX_LOG_LEVEL (default INFO); an invalid value
    # falls back to INFO. basicConfig is a no-op if a host/test harness
    # already installed root handlers (force=False), so we never clobber them.
    _log_level_name = os.environ.get("JAMX_LOG_LEVEL", "INFO")
    _log_level = logging.getLevelName(_log_level_name.upper())
    if not isinstance(_log_level, int):
        _log_level = logging.INFO
    logging.basicConfig(
        level=_log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # T-01-05: identify the app BEFORE any QSettings / QStandardPaths use.
    QCoreApplication.setOrganizationName("Marmelade")
    QCoreApplication.setApplicationName("Marmelade")

    app = QApplication.instance() or QApplication(args)

    # App/window icon (title bar + taskbar/dock). No-op until the asset exists.
    _apply_app_icon(app)

    # Fusion + QSS + PyQtGraph config flags — must happen before MainWindow().
    theme.apply_theme(app)  # type: ignore[arg-type]

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
