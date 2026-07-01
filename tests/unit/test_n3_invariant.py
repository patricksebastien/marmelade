"""N-3 invariant — ``src/marmelade/audio/mastering/`` is Qt-free.

The mastering subpackage is intentionally toolkit-free so the audio tier
(DSP, LUFS, cache resolution) stays unit-testable without a QApplication
event loop. The only deliberate boundary import is the
``QSettings`` import in ``chain.py``'s session-snapshot helper — that
is excluded from the grep below per RESEARCH §Pattern 2 Reuse
Discipline 5 option (b).

Phase 7 — Plan 01 Wave 0 (07-01-PLAN.md Task 1, GREEN after Task 2).
"""

from __future__ import annotations

import re
from pathlib import Path


_MASTERING_ROOT = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "marmelade"
    / "audio"
    / "mastering"
)

# Match any *toolkit* (Widget/Gui) Qt import. The ``QtCore.QSettings``
# import in ``chain.py`` is the documented deliberate boundary crossing
# (PATTERNS Reuse Discipline 5 option (b)) — we explicitly forbid the
# higher-level widget/GUI surface only.
_BAD_IMPORT_RE = re.compile(
    r"^\s*(?:from\s+PySide6\.(?:QtWidgets|QtGui)|import\s+PySide6\.(?:QtWidgets|QtGui))",
    re.MULTILINE,
)


def test_audio_mastering_no_qt():
    """No ``PySide6.QtWidgets`` / ``PySide6.QtGui`` import under ``audio/mastering/``."""
    if not _MASTERING_ROOT.exists():
        # RED state — the package does not exist yet. Task 2 lands the GREEN.
        # Skip rather than fail at collection time so the rest of the suite
        # runs deterministically. The Task 2 verify step calls this test
        # directly and expects it to PASS once the package exists.
        import pytest

        pytest.skip("audio/mastering/ does not exist yet (Task 2 lands the package)")

    offenders: list[str] = []
    for py in _MASTERING_ROOT.rglob("*.py"):
        if "__pycache__" in py.parts:
            continue
        text = py.read_text(encoding="utf-8")
        if _BAD_IMPORT_RE.search(text):
            offenders.append(str(py))
    assert not offenders, (
        "audio/mastering/ must be Qt-toolkit-free (no QtWidgets / QtGui "
        f"imports). Offending files: {offenders}"
    )
