"""Unit test for ``marmelade.paths.default_cache_root`` (N-3 — Qt-using helper).

This is the *single* file in the audio-related code stack that touches Qt;
keeping it at the top level lets ``src/marmelade/audio/*`` stay Qt-free.

The test consumes ``tmp_cache_dir`` from ``tests/conftest.py``. After plan
01-06's CR-06 closure, that fixture pins ``default_cache_root`` to the
per-test ``tmp_path / 'cache'`` directory via ``monkeypatch.setattr`` (not
via ``QStandardPaths.setTestModeEnabled`` alone), so the helper returns
the fixture's path while the test runs. The assertion checks the
post-pin equality — proving the helper goes through the patched binding
in BOTH the source module (``marmelade.paths``) and this test
module's local import.

Phase 7 Plan 07-05 — also pins :func:`matchering_reference_dir` at the
``~/Music/Marmelade/References/`` shape (D-12 in CONTEXT.md item 12).
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QCoreApplication

import marmelade.paths as paths_module
from marmelade.paths import default_cache_root, matchering_reference_dir


def test_default_cache_root_returns_per_test_cache_root(tmp_cache_dir: Path) -> None:
    """Inside ``tmp_cache_dir``, ``default_cache_root()`` returns the
    per-test path — proving the conftest fixture's CR-06 fix patches BOTH
    the source-module attribute AND the importer's local binding.

    Before CR-06 closure (plan 01-06), this test asserted only that the
    leaf segment was ``Marmelade`` — a weak invariant that passed even
    while real cache writes were leaking to ``~/.qttest/cache/Marmelade``.
    The current assertion is strong: it proves the helper has been
    redirected to the per-test directory, which is the load-bearing
    invariant that prevents cache leakage across pytest invocations.
    """
    # Both the locally-imported name and the source-module attribute must
    # resolve to the per-test path — that's what makes integration tests
    # actually isolated.
    assert isinstance(default_cache_root(), Path)
    assert default_cache_root() == tmp_cache_dir
    assert paths_module.default_cache_root() == tmp_cache_dir


def test_default_cache_root_not_triple_nested_when_org_app_set() -> None:
    """Regression-pin: ``default_cache_root()`` must NOT triple-nest even
    when ``QCoreApplication.organizationName`` and ``applicationName`` are
    both set to ``"Marmelade"`` (which is what the running app does in
    ``marmelade.app``).

    Pins the bug fixed by switching from ``QStandardPaths.CacheLocation``
    (app-specific — embeds ``<orgName>/<applicationName>``) to
    ``QStandardPaths.GenericCacheLocation`` (OS cache ROOT, no app prefix)
    in :func:`marmelade.paths.default_cache_root`. Before the fix, the
    running app saw ``~/.cache/Marmelade/Marmelade/Marmelade``
    (CacheLocation appended ``Marmelade/Marmelade``, then
    ``paths.py`` appended another ``Marmelade``), while a bare test
    process with org/app unset saw ``~/.cache/Marmelade`` — diverging
    caches that broke the danceability tests (looking for ``.pb`` model
    files the app had written to the triple-nested path).

    This test bypasses the ``tmp_cache_dir`` fixture deliberately: that
    fixture monkeypatches ``default_cache_root`` at the source module
    AND at every importer's local binding, so it can't exercise the
    real ``QStandardPaths.writableLocation`` call we are pinning here.
    The test sets the org/app names on the live ``QCoreApplication`` and
    restores them in ``finally`` so it doesn't pollute the rest of the
    pytest run.
    """
    saved_org_name = QCoreApplication.organizationName()
    saved_app_name = QCoreApplication.applicationName()
    try:
        QCoreApplication.setOrganizationName("Marmelade")
        QCoreApplication.setApplicationName("Marmelade")

        p = default_cache_root()

        # Leaf stable across platforms (existing contract).
        assert p.name == "Marmelade", f"leaf segment wrong: {p}"
        # Regression-pin: parent MUST NOT be another "Marmelade" segment.
        # Before the fix, parent.name == "Marmelade" (and grandparent too).
        assert p.parent.name != "Marmelade", (
            f"Triple-nesting regression: parent of {p} is also "
            f"'Marmelade' — default_cache_root() is double-counting "
            f"the app name."
        )
        # Belt-and-suspenders: pin the canonical Linux path shape.
        if sys.platform.startswith("linux"):
            assert p.parts[-2:] == (".cache", "Marmelade"), (
                f"Linux path shape regression: expected "
                f"(..., '.cache', 'Marmelade'), got {p.parts[-2:]} "
                f"(full path: {p})"
            )
    finally:
        QCoreApplication.setOrganizationName(saved_org_name)
        QCoreApplication.setApplicationName(saved_app_name)


# ---------------------------------------------------------------------------
# Phase 7 Plan 07-05 — matchering_reference_dir().
# ---------------------------------------------------------------------------


def test_matchering_reference_dir_returns_music_subdir() -> None:
    """``matchering_reference_dir()`` returns ``~/Music/Marmelade/References/``.

    Phase 7 — D-12 (CONTEXT.md domain item 12). The references library is
    user-curated content (NOT derived cache), so it lives under
    ``~/Music`` rather than the OS cache root. The exact path shape is the
    UI-SPEC contract — the picker UI also encodes it in its empty-state
    guidance label.

    No fixture: we want to see the REAL path the helper produces
    regardless of test mode (the helper has no Qt dependency — it uses
    ``Path.home()`` directly).
    """
    p = matchering_reference_dir()
    assert isinstance(p, Path), type(p)
    assert p == Path.home() / "Music" / "Marmelade" / "References", p
