"""Matchering reference picker — scan + Param builder helpers.

Phase 7 Plan 07-05. Picker UX strategy A (per PLAN.md Task 3 ``<behavior>``):
NOT a standalone dialog — the picker is composed INSIDE
:class:`~marmelade.ui.params_dialog.ParamsDialog` via the choice-kind
widget + browse_filter render path (already wired by Plan 07-02 Task 1).

This module owns the directory-scan + Param-build helpers — pure functions
over filesystem state that are easy to unit-test without a QApplication
event loop. The MasteringDialog (per-keeper) and MasteringDock (session)
gear handlers call these helpers to dynamically populate the combobox
choices before constructing the ParamsDialog.

Empty-state guidance is rendered by ParamsDialog itself when a
choice-kind Param has ``browse_filter`` set AND ``choices`` is the
single ``("",)`` placeholder — see :meth:`ParamsDialog._build_choice_widget`.
"""

from __future__ import annotations

from pathlib import Path

from marmelade.audio.mastering.params import Param
from marmelade.paths import matchering_reference_dir


# File extensions surfaced by the picker. WAV + FLAC succeed via
# soundfile.read without spawning ffmpeg (RESEARCH §Pitfall 3 — pinned
# in tests/unit/audio/test_matchering_no_ffmpeg.py). Adding MP3 here
# would silently require ffmpeg on PATH; out-of-scope for Phase 7.
_REFERENCE_FILE_GLOBS: tuple[str, ...] = ("*.wav", "*.flac")


def scan_reference_dir() -> list[tuple[str, Path]]:
    """Return sorted ``(filename, absolute_path)`` tuples for WAV+FLAC files.

    Scans :func:`matchering_reference_dir` for ``*.wav`` and ``*.flac``
    files (case-insensitive). Returns ``[]`` when the directory is
    empty OR does not exist (the MainWindow auto-create-on-init hook
    should prevent the latter, but the empty-list return is a safe
    fallback for tests that monkeypatch the dir to a non-existent path).

    The returned list is sorted by filename (case-insensitive) so the
    picker combobox displays a stable, alphabetical order.
    """
    ref_dir = matchering_reference_dir()
    if not ref_dir.exists() or not ref_dir.is_dir():
        return []
    found: dict[str, Path] = {}
    for pattern in _REFERENCE_FILE_GLOBS:
        for p in ref_dir.glob(pattern):
            if p.is_file():
                found[p.name] = p
    return sorted(found.items(), key=lambda item: item[0].lower())


def build_reference_param(current_value: str = "") -> Param:
    """Build a ``reference_path`` choice :class:`Param` for the picker UI.

    Args:
        current_value: The current ``mastering.matchering.reference_path``
            value (typically a filename from the library dir, an
            absolute path for a Browse-picked file, or empty string).
            When non-empty AND not in the scan, an extra
            ``"(custom: <basename>)"`` choice is added so the combobox
            can display + persist the user's previous selection.

    Returns:
        A choice-kind :class:`Param` with:

        * ``name = "reference_path"``,
        * ``kind = "choice"``,
        * ``default = current_value`` (must be in ``choices`` —
          Param.__post_init__ invariant),
        * ``choices`` populated from :func:`scan_reference_dir` + the
          ``""`` empty placeholder + the optional custom entry,
        * ``browse_filter`` set so :class:`ParamsDialog` renders a
          Browse button and (when ``choices`` is empty/placeholder-only)
          the inline empty-state guidance label.
        * ``requires_recompute = True`` — changing the reference
          regenerates the matchered cache (Plan 01 Task 2
          ``is_mastered_cache_fresh`` checks config_hash equality).
    """
    scan = scan_reference_dir()
    # Always-present empty-string entry — lets the user explicitly
    # "no reference" and the chain pass-through happen.
    choices: list[str] = [""]
    choices.extend(name for name, _ in scan)

    # If current_value is set and NOT a known library filename, add a
    # "(custom: <basename>)" entry so the combobox can preserve the
    # selection (typical for Browse-picked absolute paths).
    default = current_value
    if current_value and current_value not in choices:
        # Display the basename to keep the dropdown narrow.
        custom_label = current_value
        choices.append(custom_label)
        default = custom_label

    return Param(
        name="reference_path",
        label="Reference track",
        kind="choice",
        default=default,
        requires_recompute=True,
        choices=tuple(choices),
        browse_filter="Audio files (*.wav *.flac);;All files (*)",
        description=(
            "Drop pro-mastered reference tracks (WAV or FLAC) into "
            "~/Music/Marmelade/References/ — reload to refresh."
        ),
    )


__all__ = ["scan_reference_dir", "build_reference_param"]
