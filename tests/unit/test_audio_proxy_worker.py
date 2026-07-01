"""Unit-level pins for :class:`marmelade.audio.audio_proxy_worker.AudioProxyRunnable`.

Constructor-time invariants ONLY — no thread pool, no audio I/O, no Qt event
loop (the worker-test shape per PATTERNS.md §12). End-to-end coverage
(start → exactly-one terminal signal) lives in Plan 04's MainWindow
integration tests.

The load-bearing invariants pinned here are:

* **CR-02** — ``runnable.autoDelete()`` is ``False``. MainWindow stores the
  runnable on ``self._current_proxy_runnable`` and later dereferences
  ``runnable.signals`` via stored ``_mw_proxy_conn_*`` connection tokens; if
  Qt's default ``autoDelete=True`` were in effect, the C++ QRunnable would be
  freed as soon as ``run()`` returns and that signal-access path would be a
  use-after-free.
* **D-16** — ``runnable.signals`` is EXACTLY a
  :class:`marmelade.concurrency.worker.WorkerSignals` instance, not a
  subclass. The 4-signal contract (``progress`` / ``finished`` / ``error`` /
  ``cancelled``) is a cross-worker invariant shared with the peak-builder and
  heatmap workers. Subclassing would break that contract.
* **D-17** — ``BuildCancelled`` is the shared cancellation type imported from
  :mod:`marmelade.audio.peak_builder` (NOT a new ``AudioProxyCancelled``).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import QRunnable

from marmelade.audio import audio_proxy_worker
from marmelade.audio import peak_builder
from marmelade.audio.audio_proxy_worker import AudioProxyRunnable
from marmelade.concurrency.worker import WorkerSignals


# A QCoreApplication is required for QObject construction (WorkerSignals is a
# QObject). pytest-qt's ``qapp`` fixture provides the singleton; autouse so
# every test in this module gets it without repeating the parameter.
@pytest.fixture(autouse=True)
def _qapp(qapp):
    return qapp


def test_audio_proxy_runnable_disables_auto_delete(tmp_path: Path) -> None:
    """CR-02 — AudioProxyRunnable.autoDelete() must be False at construction.

    Python refcount owns the lifetime; the C++ runnable must NOT be freed by
    QThreadPool when ``run()`` returns, because MainWindow keeps a reference
    in ``self._current_proxy_runnable`` and later dereferences
    ``runnable.signals`` through that reference (the cancel-preamble
    targeted-disconnect path).
    """
    src = tmp_path / "src.wav"
    dst = tmp_path / "dst.wav"
    runnable = AudioProxyRunnable(src, dst)
    assert runnable.autoDelete() is False, (
        "AudioProxyRunnable must call setAutoDelete(False) — otherwise "
        "QThreadPool deletes the C++ object after run() and MainWindow's "
        "stored reference becomes a use-after-free hazard when the "
        "cancel-preamble disconnect path dereferences `runnable.signals`."
    )


def test_audio_proxy_runnable_uses_worker_signals_verbatim(tmp_path: Path) -> None:
    """D-16 — ``runnable.signals`` MUST be a vanilla ``WorkerSignals``.

    The exact-type check (``type(...) is WorkerSignals``) is intentional: an
    ``isinstance`` check would pass for a subclass, which is exactly what
    D-16 forbids. The 4-signal contract is the cross-worker invariant; we do
    NOT define an ``AudioProxySignals`` class.
    """
    src = tmp_path / "src.wav"
    dst = tmp_path / "dst.wav"
    runnable = AudioProxyRunnable(src, dst)
    assert type(runnable.signals) is WorkerSignals, (
        "AudioProxyRunnable.signals must be exactly WorkerSignals (D-16); "
        f"got {type(runnable.signals).__name__}. Do NOT subclass — the "
        "4-signal contract is cross-worker."
    )


def test_audio_proxy_runnable_stores_paths(tmp_path: Path) -> None:
    """src_path / dst_path must be ``Path`` instances equal to the inputs."""
    src = tmp_path / "src.wav"
    dst = tmp_path / "dst.wav"
    runnable = AudioProxyRunnable(src, dst)
    assert isinstance(runnable.src_path, Path)
    assert isinstance(runnable.dst_path, Path)
    assert runnable.src_path == src
    assert runnable.dst_path == dst


def test_audio_proxy_runnable_accepts_str_paths(tmp_path: Path) -> None:
    """Constructor accepts ``str`` paths and coerces them to ``Path``."""
    src = tmp_path / "src.wav"
    dst = tmp_path / "dst.wav"
    runnable = AudioProxyRunnable(str(src), str(dst))
    assert isinstance(runnable.src_path, Path)
    assert isinstance(runnable.dst_path, Path)
    assert runnable.src_path == src
    assert runnable.dst_path == dst


def test_audio_proxy_runnable_cancel_is_idempotent(tmp_path: Path) -> None:
    """``cancel()`` is a single-line ``Event.set()`` — safe to call twice.

    Once set, ``_is_cancelled()`` returns True; calling ``cancel()`` again is
    a no-op (``Event.set`` is idempotent). Pins the cancel contract that
    MainWindow's cancel-preamble depends on.
    """
    src = tmp_path / "src.wav"
    dst = tmp_path / "dst.wav"
    runnable = AudioProxyRunnable(src, dst)
    assert runnable._is_cancelled() is False
    runnable.cancel()
    assert runnable._is_cancelled() is True
    # second call must not raise; state stays True
    runnable.cancel()
    assert runnable._is_cancelled() is True


def test_audio_proxy_runnable_has_four_signals(tmp_path: Path) -> None:
    """All four WorkerSignals attributes are reachable through ``runnable.signals``."""
    src = tmp_path / "src.wav"
    dst = tmp_path / "dst.wav"
    runnable = AudioProxyRunnable(src, dst)
    # Attribute access only — no emission, no connection. This pins the
    # 4-signal contract surface at construction time.
    assert hasattr(runnable.signals, "progress")
    assert hasattr(runnable.signals, "finished")
    assert hasattr(runnable.signals, "error")
    assert hasattr(runnable.signals, "cancelled")


def test_audio_proxy_runnable_isinstance_qrunnable(tmp_path: Path) -> None:
    """Direct QRunnable subclass — no intermediate base."""
    src = tmp_path / "src.wav"
    dst = tmp_path / "dst.wav"
    runnable = AudioProxyRunnable(src, dst)
    assert isinstance(runnable, QRunnable)


def test_audio_proxy_runnable_reuses_buildcancelled() -> None:
    """D-17 — ``BuildCancelled`` is reused from :mod:`peak_builder` (no new class).

    The worker module imports ``BuildCancelled`` from
    :mod:`marmelade.audio.peak_builder`. The load-bearing assertion is the
    identity check ``worker_module.BuildCancelled is peak_builder.BuildCancelled``
    — there must be no fresh ``AudioProxyCancelled`` class hiding under any
    alias.
    """
    assert audio_proxy_worker.BuildCancelled is peak_builder.BuildCancelled, (
        "audio_proxy_worker.BuildCancelled must be the SAME class object as "
        "peak_builder.BuildCancelled (D-17). Defining a new exception type "
        "here would break the cross-worker `except BuildCancelled` machinery "
        "shared with PeakBuilderRunnable and EnergyHeatmapRunnable."
    )
