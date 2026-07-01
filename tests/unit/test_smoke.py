"""Wave-0 smoke: every Plan-01 module must import without side effects."""

from __future__ import annotations


def test_package_imports() -> None:
    """All Plan-01 modules import cleanly and __version__ is pinned at 0.1.0."""
    import marmelade
    import marmelade.app  # noqa: F401
    import marmelade.ui.main_window  # noqa: F401
    import marmelade.ui.waveform_view  # noqa: F401
    import marmelade.ui.theme  # noqa: F401

    assert marmelade.__version__ == "0.1.0", marmelade.__version__
