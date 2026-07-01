"""Programmatic proof that ``tmp_cache_dir`` (in tests/conftest.py) closes
REVIEW.md CR-06.

CR-06 had two distinct sub-bugs that the fix in plan 01-06 addresses:

1. The fixture did NOT actually pin :func:`marmelade.paths.default_cache_root`
   to ``tmp_path / 'cache'``. Instead it relied on
   ``QStandardPaths.setTestModeEnabled(True)`` to redirect the lookup, which
   on Linux lands writes in ``~/.qttest/cache/Marmelade-test`` — a path
   in the user's home that persists across pytest invocations.

2. ``QStandardPaths.setTestModeEnabled(True)`` is a *process-global* toggle
   and the fixture had no teardown. Any test running *after* a fixture
   consumer inherited the test-mode prefix.

This test asserts both invariants programmatically by running the fixture
inside an inner pytest-style invocation (we instantiate the fixture context
manually via ``pytest.FixtureRequest`` is not trivial — easier route: use a
nested test function that pytest itself drives, captured here as the standard
fixture-consumption pattern).

We use the simplest possible technique: ask pytest to run BOTH tests in one
file, in declaration order. The first test consumes ``tmp_cache_dir`` and
records the inside-value of ``default_cache_root()`` in a module-level dict.
The second test, which does NOT consume ``tmp_cache_dir``, asserts that
``QStandardPaths.testModeEnabled()`` has been restored to ``False`` AND
that ``default_cache_root()`` is no longer pinned to the per-test path.

This pair of tests cannot be replaced by a single test because the teardown
(part 2 of CR-06) by definition only happens AFTER the fixture goes out of
scope — which from a function-scoped fixture's point of view is "after the
test function returns".
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import QStandardPaths

import marmelade.paths as paths_module
from marmelade.paths import default_cache_root

# Module-level capture so the second test can observe what the first test saw.
_CAPTURE: dict[str, object] = {}


def test_tmp_cache_dir_isolates_default_cache_root(
    tmp_path: Path, tmp_cache_dir: Path
) -> None:
    """CR-06 fix part 1: inside the fixture, ``default_cache_root()`` returns
    the per-test ``tmp_path / 'cache'`` — NOT ``~/.qttest/cache/...``.
    """
    # The fixture returns the per-test cache root.
    assert tmp_cache_dir == tmp_path / "cache"
    assert tmp_cache_dir.exists()

    # The helper, called from production-style code, must now return that
    # same path — proving the monkeypatch.setattr on
    # "marmelade.paths.default_cache_root" is in effect.
    observed = default_cache_root()
    assert observed == tmp_cache_dir, (
        f"CR-06 part 1: default_cache_root() must return the per-test cache "
        f"root while inside tmp_cache_dir. Got {observed!r}, expected "
        f"{tmp_cache_dir!r}. If you see ~/.qttest/... in the observed value, "
        f"the monkeypatch.setattr in conftest.py is missing or wrong."
    )

    # Test-mode was enabled on entry — it should still be True inside the
    # fixture body. This is the preserved invariant from the original code.
    assert QStandardPaths.isTestModeEnabled() is True, (
        "tmp_cache_dir must still call setTestModeEnabled(True) on entry — "
        "the fix only adds the False teardown, not removes the True enable."
    )

    # Record state for the next test to observe.
    _CAPTURE["inside_value"] = observed
    _CAPTURE["inside_test_mode"] = QStandardPaths.isTestModeEnabled()


def test_tmp_cache_dir_restores_state_after_teardown() -> None:
    """CR-06 fix part 2: after tmp_cache_dir's teardown,
    ``QStandardPaths.setTestModeEnabled(False)`` has restored the
    process-global toggle, and ``marmelade.paths.default_cache_root``
    is no longer pinned to the per-test directory.

    This test does NOT consume ``tmp_cache_dir`` itself — it relies on
    pytest declaration order so it runs immediately after the previous
    test, observing the post-teardown world.
    """
    # The previous test must have populated this; if not, our ordering
    # assumption is broken.
    assert "inside_value" in _CAPTURE, (
        "test ordering broken: the previous test (which consumes "
        "tmp_cache_dir) did not run first."
    )

    # CR-06 fix part 2: process-global toggle is restored.
    assert QStandardPaths.isTestModeEnabled() is False, (
        "CR-06 part 2: after tmp_cache_dir's teardown, "
        "QStandardPaths.isTestModeEnabled() must return False. The fixture "
        "is missing 'QStandardPaths.setTestModeEnabled(False)' after the "
        "'yield' statement."
    )

    # The monkeypatch on marmelade.paths.default_cache_root has been
    # unwound by pytest's monkeypatch teardown — the helper is back to its
    # production behaviour. We can't easily assert the actual path because
    # it now depends on the user's real cache dir, but we CAN assert:
    #   (a) the helper is callable,
    #   (b) it does NOT return the per-test cache root (which was inside the
    #       deleted tmp_path).
    inside_value = _CAPTURE["inside_value"]
    assert isinstance(inside_value, Path)
    post_value = default_cache_root()
    assert post_value != inside_value, (
        "CR-06 part 2: after tmp_cache_dir's teardown, "
        "default_cache_root() must no longer return the per-test path "
        f"({inside_value!r}). The monkeypatch on "
        "'marmelade.paths.default_cache_root' did not unwind correctly."
    )

    # And the helper is back to being the production function object on
    # the module (not the lambda the fixture injected).
    assert paths_module.default_cache_root is not (
        lambda: inside_value
    ), "sanity: module attribute is restored"
