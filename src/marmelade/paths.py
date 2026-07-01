"""OS-cache-directory derivation — the *only* Qt-using helper in the audio path stack.

N-3 resolution (option 2 — relocate helper out of ``audio/``):
    Living at the top level of the package keeps the architectural invariant
    "no Qt imports anywhere under ``src/marmelade/audio/``" intact. The
    alternatives — relaxing the gate, or accepting a Qt import inside the
    audio layer — were rejected because:

    * the audio layer's Qt-freeness is what lets Plans 02-03's unit tests run
      without a ``QApplication`` event loop,
    * a future CLI front-end (research §Open Question 6) can reuse the audio
      backbone unchanged, and
    * ``QStandardPaths`` is a non-GUI Qt utility — it does *not* require a
      ``QApplication`` instance, it works in worker threads, and it costs
      nothing to colocate at the top level next to ``app.py``.

This helper is the single source of truth for the cache root. Plans 01-03
and 01-04 import it via ``from marmelade.paths import default_cache_root``.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QStandardPaths


def default_cache_root() -> Path:
    """Return ``<OS cache root>/Marmelade`` as a :class:`pathlib.Path`.

    Uses ``GenericCacheLocation`` (the OS cache ROOT — ``~/.cache`` on
    Linux, ``~/Library/Caches`` on macOS, ``%LOCALAPPDATA%/cache`` on
    Windows) and appends ``Marmelade`` ourselves. This makes the
    function deterministic regardless of whether
    ``QCoreApplication.organizationName``/``applicationName`` are set on
    the running process — production app, tests without the org/app
    monkeypatch, and standalone scripts (e.g., ``fetch_models.sh``) all
    resolve to the same path.

    Why ``GenericCacheLocation`` over ``CacheLocation``: Qt's
    ``CacheLocation`` is app-specific by design — it embeds
    ``<orgName>/<applicationName>`` when both are set on
    ``QCoreApplication``. Combined with our explicit ``/Marmelade``
    append, that produced triple-nesting in the running app (org/app
    set: ``~/.cache/Marmelade/Marmelade/Marmelade``) but
    single-nesting in the test process (org/app unset:
    ``~/.cache/Marmelade``). ``GenericCacheLocation`` is the OS root
    without that prefix, so our explicit append is the ONLY namespacing
    layer.

    With the test-mode toggle (``QStandardPaths.setTestModeEnabled(True)``)
    the root is redirected under a private prefix (``~/.qttest`` on
    Linux) — test mode applies to ``GenericCacheLocation`` too, so test
    runs never write to the user's real cache directory.

    The leaf segment ``Marmelade`` is contractually stable across
    platforms and is what tests assert against (the parent varies).
    """
    cache_root = QStandardPaths.writableLocation(
        QStandardPaths.StandardLocation.GenericCacheLocation
    )
    return Path(cache_root) / "Marmelade"


def default_open_dir() -> str:
    """Return the OS-conventional initial directory for the file-open dialog.

    Used by MainWindow's QFileDialog for first-time opens (subsequent opens
    use QSettings 'last_dir'). On Linux that's ``~/Music`` (or
    ``$XDG_MUSIC_DIR``); macOS ``~/Music``; Windows ``%USERPROFILE%\\Music``.

    Lives at the package top level next to :func:`default_cache_root` so
    the GUI tier never imports Qt-toolkit path lookups directly (W-5 /
    N-3: single source of truth for all writable-location queries).
    """
    return QStandardPaths.writableLocation(
        QStandardPaths.StandardLocation.MusicLocation
    )


def mastered_cache_dir() -> Path:
    """Return ``<cache_root>/mastered`` for the Phase 7 mastered-WAV cache.

    Phase 7 — RESEARCH §Pattern 3: each mastered output is written at
    ``<cache_root>/mastered/<src_key>-<keeper_id>-<config_hash>.wav``.
    The directory itself is created lazily by the writer
    (``MasteringRunnable`` calls ``parent.mkdir(parents=True,
    exist_ok=True)`` immediately before the atomic write), so this
    helper does NOT create the directory — same discipline as
    :func:`default_cache_root` + ``heatmap_cache`` use.
    """
    return default_cache_root() / "mastered"


def matchering_reference_dir() -> Path:
    """Return ``~/Music/Marmelade/References/`` — the Matchering library.

    Phase 7 — D-12 (CONTEXT.md domain item 12) + UI-SPEC §"Matchering
    reference picker — D-03 + D-12". This is the directory users drop
    pro-mastered reference tracks into; the Plan 07-05 reference-picker
    UI scans it for ``*.wav`` and ``*.flac`` files to populate the
    combobox.

    NOT under :func:`default_cache_root`: the references are
    user-curated source material, not derived cache. Living under
    ``~/Music`` matches the convention of every other audio-management
    app on Linux/macOS/Windows.

    Does NOT use ``QStandardPaths`` — :func:`Path.home` is sufficient
    and Qt-free (the helper can be reached from non-Qt code paths
    safely). The directory is auto-created by the MainWindow on first
    launch (``mkdir(parents=True, exist_ok=True)`` in ``__init__``,
    wrapped in ``try/except OSError`` so a read-only home cannot crash
    app startup).
    """
    return Path.home() / "Music" / "Marmelade" / "References"
