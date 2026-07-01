"""Phase 7 Plan 07-05 Task 3 — Matchering reference picker integration.

Pins:

* :func:`scan_reference_dir` returns sorted WAV + FLAC entries; MP3 +
  other extensions are NOT surfaced.
* :func:`scan_reference_dir` returns ``[]`` for an empty directory.
* :func:`build_reference_param` constructs a choice :class:`Param` with
  ``browse_filter`` set + the WAV/FLAC filter string.
* :func:`build_reference_param` exposes scanned filenames as choices.
* :func:`build_reference_param` preserves a custom ``current_value`` that
  is NOT in the scan (Browse-picked absolute path) via a "(custom: ...)"
  trick.
* ParamsDialog renders the inline empty-state guidance label when the
  Param's choices are the ``("",)`` placeholder only.
* The MasteringDialog's matchering gear opens a ParamsDialog with the
  reference combobox populated from the (monkeypatched) reference dir.
* The Browse button is present in the ParamsDialog when
  ``browse_filter`` is set.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import soundfile as sf
from PySide6.QtWidgets import QComboBox, QDialog, QLabel, QPushButton

import numpy as np


def _write_silent_wav(p: Path, seconds: float = 0.2, sr: int = 44100) -> None:
    """Write a tiny silent stereo WAV to ``p`` (touch-equivalent that
    ``soundfile.info`` will accept). For empty-state guidance tests we
    don't care about content — only file presence + extension."""
    n = int(round(seconds * sr))
    audio = np.zeros((2, n), dtype=np.float32)
    sf.write(str(p), audio.T, sr, format="WAV", subtype="FLOAT")


def _write_silent_flac(p: Path, seconds: float = 0.2, sr: int = 44100) -> None:
    n = int(round(seconds * sr))
    audio = np.zeros((2, n), dtype=np.float32)
    sf.write(str(p), audio.T, sr, format="FLAC", subtype="PCM_16")


@pytest.fixture
def fake_ref_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Provide an empty tmp directory and pin :func:`matchering_reference_dir`
    to it. Yields the directory path.

    Patches BOTH ``marmelade.paths`` and the importer namespaces that
    consume the helper at module-level (the picker module imports it
    via ``from marmelade.paths import matchering_reference_dir``).
    """
    import marmelade.paths as paths_mod
    import marmelade.ui.matchering_reference_picker as picker_mod

    ref_dir = tmp_path / "ref_lib"
    ref_dir.mkdir(parents=True)

    monkeypatch.setattr(paths_mod, "matchering_reference_dir", lambda: ref_dir)
    monkeypatch.setattr(
        picker_mod, "matchering_reference_dir", lambda: ref_dir, raising=False
    )
    return ref_dir


# ---------------------------------------------------------------------------
# scan_reference_dir + build_reference_param — pure helpers.
# ---------------------------------------------------------------------------


def test_scan_reference_dir_lists_wav_and_flac_only(fake_ref_dir: Path) -> None:
    """Scan returns WAV + FLAC; MP3 / OGG / other extensions are NOT surfaced.

    Pitfall 3 — surfacing MP3 would silently require ffmpeg on PATH at
    matchering.load time. The picker's allow-list is the policy boundary
    pinned here.
    """
    from marmelade.ui.matchering_reference_picker import scan_reference_dir

    _write_silent_wav(fake_ref_dir / "a_song.wav")
    _write_silent_flac(fake_ref_dir / "b_song.flac")
    # MP3 — NOT a valid library entry. Write an empty file (extension only).
    (fake_ref_dir / "c_song.mp3").write_bytes(b"")

    out = scan_reference_dir()
    names = [name for name, _ in out]
    assert "a_song.wav" in names
    assert "b_song.flac" in names
    assert "c_song.mp3" not in names, "MP3 must not be surfaced (Pitfall 3)"


def test_scan_reference_dir_sorted_alphabetically(fake_ref_dir: Path) -> None:
    """Scan output is sorted by lowercase filename — stable combobox order."""
    from marmelade.ui.matchering_reference_picker import scan_reference_dir

    for name in ("ZebraJam.wav", "alphaJam.wav", "MidJam.flac"):
        _write_silent_wav(fake_ref_dir / name)

    out = scan_reference_dir()
    names = [name for name, _ in out]
    assert names == sorted(names, key=str.lower)


def test_scan_reference_dir_empty_returns_empty_list(fake_ref_dir: Path) -> None:
    """An empty library directory returns ``[]``."""
    from marmelade.ui.matchering_reference_picker import scan_reference_dir

    assert scan_reference_dir() == []


def test_scan_reference_dir_nonexistent_returns_empty_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pin: nonexistent ref dir returns ``[]`` (safety fallback).

    The MainWindow auto-create-on-init hook should prevent the
    nonexistent case, but a brittle read-only $HOME could mean the dir
    isn't there. Empty-list is the safe default.
    """
    import marmelade.paths as paths_mod
    import marmelade.ui.matchering_reference_picker as picker_mod

    missing = tmp_path / "does_not_exist"
    monkeypatch.setattr(paths_mod, "matchering_reference_dir", lambda: missing)
    monkeypatch.setattr(
        picker_mod, "matchering_reference_dir", lambda: missing, raising=False
    )

    from marmelade.ui.matchering_reference_picker import scan_reference_dir

    assert scan_reference_dir() == []


def test_build_reference_param_empty_dir_has_placeholder_only(
    fake_ref_dir: Path,
) -> None:
    """Empty library → ``choices == ("",)`` (placeholder only).

    The ParamsDialog renders the inline empty-state guidance label
    when ``len(choices) <= 1`` AND ``browse_filter is not None``.
    """
    from marmelade.ui.matchering_reference_picker import build_reference_param

    p = build_reference_param("")
    assert p.choices == ("",), p.choices
    assert p.kind == "choice"
    assert p.browse_filter is not None
    assert "*.wav" in p.browse_filter
    assert "*.flac" in p.browse_filter
    assert p.requires_recompute is True


def test_build_reference_param_dir_with_files_lists_them(
    fake_ref_dir: Path,
) -> None:
    """Non-empty dir → both filenames present as choices."""
    from marmelade.ui.matchering_reference_picker import build_reference_param

    _write_silent_wav(fake_ref_dir / "ref1.wav")
    _write_silent_flac(fake_ref_dir / "ref2.flac")

    p = build_reference_param("")
    assert "ref1.wav" in p.choices
    assert "ref2.flac" in p.choices
    # Empty-string placeholder is always present too.
    assert "" in p.choices


def test_build_reference_param_preserves_custom_current_value(
    fake_ref_dir: Path,
) -> None:
    """A Browse-picked absolute path (NOT in the scan) is preserved.

    The user previously picked a file via Browse → the keeper's
    ``mastering.matchering.reference_path`` is an absolute path. The
    picker must preserve that selection so re-opening the gear popup
    doesn't reset it to empty.
    """
    from marmelade.ui.matchering_reference_picker import build_reference_param

    _write_silent_wav(fake_ref_dir / "in_lib.wav")
    custom = "/home/user/elsewhere/picked.wav"

    p = build_reference_param(custom)
    assert custom in p.choices, p.choices
    assert p.default == custom


# ---------------------------------------------------------------------------
# ParamsDialog rendering — empty-state guidance + Browse button + combobox.
# ---------------------------------------------------------------------------


def test_params_dialog_renders_browse_button_when_browse_filter_set(
    qtbot, qapp, fake_ref_dir: Path
) -> None:
    """The Browse button is present whenever ``browse_filter`` is set.

    Phase 7 Plan 07-02 Task 1 wired this; Plan 07-05's RED test pins
    it for the matchering reference picker shape.
    """
    from marmelade.ui.matchering_reference_picker import build_reference_param
    from marmelade.ui.params_dialog import ParamsDialog

    _write_silent_wav(fake_ref_dir / "ref.wav")
    param = build_reference_param("")
    dlg = ParamsDialog(
        title="Matchering reference parameters",
        params={"reference_path": param},
        current_values={"reference_path": ""},
    )
    qtbot.add_widget(dlg)
    browse_buttons = [
        b
        for b in dlg.findChildren(QPushButton)
        if b.text() == "Browse..."
    ]
    assert browse_buttons, "Browse... button missing from ParamsDialog"


def test_params_dialog_combobox_populated_from_dir_when_non_empty(
    qtbot, qapp, fake_ref_dir: Path
) -> None:
    """Combobox lists WAV/FLAC filenames found in the library."""
    from marmelade.ui.matchering_reference_picker import build_reference_param
    from marmelade.ui.params_dialog import ParamsDialog

    _write_silent_wav(fake_ref_dir / "alpha.wav")
    _write_silent_flac(fake_ref_dir / "beta.flac")
    param = build_reference_param("")
    dlg = ParamsDialog(
        title="Matchering reference parameters",
        params={"reference_path": param},
        current_values={"reference_path": ""},
    )
    qtbot.add_widget(dlg)
    combos = dlg.findChildren(QComboBox)
    assert combos, "No QComboBox found in ParamsDialog"
    items = [combos[0].itemText(i) for i in range(combos[0].count())]
    assert "alpha.wav" in items, items
    assert "beta.flac" in items, items


def test_params_dialog_empty_state_renders_inline_guidance_label(
    qtbot, qapp, fake_ref_dir: Path
) -> None:
    """Empty library → an inline guidance QLabel is rendered + visible.

    The label's text contains "Drop pro-mastered reference tracks" per
    UI-SPEC §"Matchering reference picker" line 392. The combobox itself
    must be disabled when only the placeholder choice is available.
    """
    from marmelade.ui.matchering_reference_picker import build_reference_param
    from marmelade.ui.params_dialog import ParamsDialog

    param = build_reference_param("")
    assert param.choices == ("",), "precondition: empty dir → placeholder only"
    dlg = ParamsDialog(
        title="Matchering reference parameters",
        params={"reference_path": param},
        current_values={"reference_path": ""},
    )
    qtbot.add_widget(dlg)

    labels = [l.text() for l in dlg.findChildren(QLabel)]
    matched = [t for t in labels if "Drop pro-mastered reference" in t]
    assert matched, (
        f"empty-state guidance label missing — labels found: {labels}"
    )

    # Combobox disabled when nothing to select.
    combos = dlg.findChildren(QComboBox)
    assert combos
    assert not combos[0].isEnabled(), (
        "combobox must be disabled when only placeholder choice available"
    )


# ---------------------------------------------------------------------------
# MasteringDialog matchering gear handler — opens picker with populated combobox.
# ---------------------------------------------------------------------------


def test_mastering_dialog_matchering_gear_opens_picker_with_populated_choices(
    qtbot, qapp, fake_ref_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Click matchering gear → ParamsDialog opens with library filenames.

    The picker UI lives INSIDE ParamsDialog (strategy A — Plan 07-05
    Task 3 <behavior>). The MasteringDialog's gear handler dynamically
    populates choices via :func:`build_reference_param`, so the dialog
    that opens has the WAV/FLAC entries pre-populated.

    We monkeypatch ParamsDialog to capture the params dict it's
    constructed with, so the test does not block on a real modal.
    """
    import copy

    from marmelade.audio.mastering.chain import _SESSION_DEFAULTS
    from marmelade.ui import mastering_dialog as md_module
    from marmelade.ui.mastering_dialog import MasteringDialog

    _write_silent_wav(fake_ref_dir / "ref_a.wav")
    _write_silent_flac(fake_ref_dir / "ref_b.flac")

    captured: dict[str, Any] = {}

    class _StubParamsDialog:
        def __init__(self, *args, **kwargs):
            captured["title"] = kwargs.get("title", "")
            captured["params"] = kwargs.get("params", {})
            captured["current_values"] = kwargs.get("current_values", {})

        def exec(self):
            return QDialog.DialogCode.Rejected

        def accepted_values(self):
            return {}

    monkeypatch.setattr(md_module, "ParamsDialog", _StubParamsDialog)

    keeper_cfg = copy.deepcopy(_SESSION_DEFAULTS)
    keeper_cfg["matchering"] = {
        "enabled": True,
        "reference_path": "",
    }
    dlg = MasteringDialog(
        keeper_id="kid_matchering_gear_open",
        keeper_mastering=keeper_cfg,
        keeper_range="00:00:10 – 00:00:20",
    )
    qtbot.add_widget(dlg)

    dlg._on_stage_gear_clicked("matchering")

    assert "Matchering" in captured["title"], captured["title"]
    params = captured["params"]
    assert "reference_path" in params, list(params.keys())
    p = params["reference_path"]
    assert "ref_a.wav" in p.choices, p.choices
    assert "ref_b.flac" in p.choices, p.choices
    assert p.browse_filter is not None


def test_mastering_dialog_matchering_apply_resolves_filename_to_abs_path(
    qtbot, qapp, fake_ref_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On Apply, selected filename resolves to absolute path inside the dir.

    is_one_off must be False (the file is in the library — not Browse-picked).
    """
    import copy

    from marmelade.audio.mastering.chain import _SESSION_DEFAULTS
    from marmelade.ui import mastering_dialog as md_module
    from marmelade.ui.mastering_dialog import MasteringDialog

    _write_silent_wav(fake_ref_dir / "ref_a.wav")

    class _AcceptStubParamsDialog:
        def __init__(self, *args, **kwargs):
            pass

        def exec(self):
            return QDialog.DialogCode.Accepted

        def accepted_values(self):
            return {"reference_path": "ref_a.wav"}

    monkeypatch.setattr(md_module, "ParamsDialog", _AcceptStubParamsDialog)

    keeper_cfg = copy.deepcopy(_SESSION_DEFAULTS)
    keeper_cfg["matchering"] = {"enabled": True, "reference_path": ""}
    dlg = MasteringDialog(
        keeper_id="kid_resolve_filename",
        keeper_mastering=keeper_cfg,
        keeper_range="00:00:10 – 00:00:20",
    )
    qtbot.add_widget(dlg)

    dlg._on_stage_gear_clicked("matchering")

    cfg = dlg._cfg["matchering"]
    assert cfg["reference_path"] == str(fake_ref_dir / "ref_a.wav"), (
        cfg["reference_path"]
    )
    assert cfg.get("is_one_off", False) is False
