"""Phase 8 Plan 08-06 Task 3 — _PhaseABMixin contract tests (revision iter 1 W6).

Pins the mixin-extraction contract: both :class:`UploadDialog` and
:class:`BundleDialog` inherit from :class:`_PhaseABMixin`, AND the
mixin's ``set_phase_b`` actually swaps the QStackedWidget index on
both dialog types.

Placebo audit (Phase 7 LEARNINGS):
    PRE-FIX expected failure signal — before Plan 08-06 Task 3 the
    mixin did not exist; the dialogs each carried duplicated
    ``set_phase_b`` / ``set_progress`` / ``show_error`` /
    ``update_thumbnail`` method bodies. The MRO check would fail
    ImportError-style (no _PhaseABMixin module to import) AND the
    duplicated method assertion would fail (same line counts in both
    files).

Three tests:

1. ``test_upload_dialog_uses_phase_ab_mixin`` — _PhaseABMixin in
   UploadDialog.__mro__.
2. ``test_bundle_dialog_uses_phase_ab_mixin`` — same for BundleDialog.
3. ``test_set_phase_b_swaps_stack_in_both_dialogs`` — parametrized
   over the two dialog classes; instantiate, call set_phase_b, assert
   stack.currentIndex() == 1.
"""

from __future__ import annotations

from io import BytesIO

import pytest
from PIL import Image

from marmelade.ui._phase_ab_mixin import _PhaseABMixin
from marmelade.ui.bundle_dialog import BundleDialog
from marmelade.ui.upload_dialog import UploadDialog


REGION_ID = "0123456789abcdef0123456789abcdef"
RID_A = "0123456789abcdef0123456789abcde0"
RID_B = "0123456789abcdef0123456789abcde1"


def _make_jpeg() -> bytes:
    img = Image.new("RGB", (1280, 720), color=(64, 96, 192))
    buf = BytesIO()
    img.save(buf, "JPEG", quality=90)
    return buf.getvalue()


def _make_upload_dialog(qtbot) -> UploadDialog:
    dlg = UploadDialog(
        keeper_id=REGION_ID,
        keeper_range="00:14:32 – 00:18:07",
        initial_title="distant ferries hum quietly",
        initial_description="2026-05-22 — exported from Marmelade",
        initial_privacy="private",
        initial_thumbnail_bytes=_make_jpeg(),
    )
    qtbot.addWidget(dlg)
    return dlg


def _make_bundle_dialog(qtbot) -> BundleDialog:
    dlg = BundleDialog(
        keepers=[
            (RID_A, "00:00:10 – 00:00:20"),
            (RID_B, "00:00:30 – 00:00:40"),
        ],
        initial_title="bundle test",
        initial_description="2026-05-22",
        initial_privacy="private",
        initial_thumbnail_bytes=_make_jpeg(),
        initial_spacer_sec=2.0,
    )
    qtbot.addWidget(dlg)
    return dlg


def test_upload_dialog_uses_phase_ab_mixin(qtbot) -> None:
    """UploadDialog inherits from _PhaseABMixin (revision iter 1 W6 contract)."""
    assert _PhaseABMixin in UploadDialog.__mro__, (
        f"_PhaseABMixin not in UploadDialog.__mro__; got {UploadDialog.__mro__}"
    )
    # Smoke: can construct + the mixin methods are callable on an instance.
    dlg = _make_upload_dialog(qtbot)
    assert callable(dlg.set_phase_b)
    assert callable(dlg.set_progress)
    assert callable(dlg.show_error)
    assert callable(dlg.update_thumbnail)


def test_bundle_dialog_uses_phase_ab_mixin(qtbot) -> None:
    """BundleDialog inherits from _PhaseABMixin (revision iter 1 W6 contract)."""
    assert _PhaseABMixin in BundleDialog.__mro__, (
        f"_PhaseABMixin not in BundleDialog.__mro__; got {BundleDialog.__mro__}"
    )
    dlg = _make_bundle_dialog(qtbot)
    assert callable(dlg.set_phase_b)
    assert callable(dlg.set_progress)
    assert callable(dlg.show_error)
    assert callable(dlg.update_thumbnail)


@pytest.mark.parametrize(
    "dialog_factory",
    [
        pytest.param(_make_upload_dialog, id="UploadDialog"),
        pytest.param(_make_bundle_dialog, id="BundleDialog"),
    ],
)
def test_set_phase_b_swaps_stack_in_both_dialogs(qtbot, dialog_factory) -> None:
    """set_phase_b() swaps the QStackedWidget to page 1 on both dialog types."""
    dlg = dialog_factory(qtbot)
    # Page 0 (Phase A — Setup) is the default visible page.
    assert dlg._stack.currentIndex() == 0
    # Invoke the mixin-provided helper.
    dlg.set_phase_b()
    # Page 1 (Phase B — Progress) is now active.
    assert dlg._stack.currentIndex() == 1
