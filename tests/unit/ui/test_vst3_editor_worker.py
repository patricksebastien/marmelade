"""quick-260626-fw2 — out-of-process VST3 editor worker (load/capture seam).

Exercises the worker's testable seam WITHOUT a real ``.vst3`` and WITHOUT
calling ``show_editor()``: ``open_editor`` is injected as a no-op recorder and
``vst3_editor_worker.load_vst3`` is monkeypatched to return a fake plugin (or
None). No QApplication, no native window — the GUI-blocking call is isolated
in ``_open_editor`` so it never runs here.
"""

from __future__ import annotations

import base64

import marmelade.ui.vst3_editor_worker as worker


class _FakePlugin:
    """Minimal pedalboard-plugin stand-in with raw_state + .name."""

    def __init__(self, name: str = "FakeVST") -> None:
        self.name = name
        self._raw = b""

    @property
    def raw_state(self) -> bytes:
        return self._raw

    @raw_state.setter
    def raw_state(self, value: bytes) -> None:
        self._raw = bytes(value)


def test_happy_path_writes_state_and_name(tmp_path, monkeypatch) -> None:
    fake = _FakePlugin(name="oXygen")
    fake.raw_state = b"PLUGIN-STATE"
    monkeypatch.setattr(worker, "load_vst3", lambda cfg: fake)

    out_state = tmp_path / "out_state.txt"
    out_name = tmp_path / "out_name.txt"
    opened: list = []

    rc = worker.run_editor(
        str(tmp_path / "Plugin.vst3"),
        "oXygen",
        "",
        str(out_state),
        str(out_name),
        open_editor=opened.append,
    )

    assert rc == 0
    assert len(opened) == 1 and opened[0] is fake
    assert base64.b64decode(out_state.read_text(encoding="utf-8")) == b"PLUGIN-STATE"
    assert out_name.read_text(encoding="utf-8") == "oXygen"


def test_unloadable_returns_2_and_writes_nothing(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(worker, "load_vst3", lambda cfg: None)

    out_state = tmp_path / "out_state.txt"
    out_name = tmp_path / "out_name.txt"
    opened: list = []

    rc = worker.run_editor(
        "/missing/Plugin.vst3",
        "",
        "",
        str(out_state),
        str(out_name),
        open_editor=opened.append,
    )

    assert rc == 2
    assert not opened
    assert not out_state.exists()
    assert not out_name.exists()
    captured = capsys.readouterr()
    assert "not loadable" in captured.err.lower()
    assert "/missing/Plugin.vst3" in captured.err


def test_missing_in_state_restores_empty(tmp_path, monkeypatch) -> None:
    fake = _FakePlugin()
    seen: dict = {}

    def _loader(cfg):
        seen["cfg"] = dict(cfg)
        return fake

    monkeypatch.setattr(worker, "load_vst3", _loader)

    out_state = tmp_path / "out_state.txt"
    out_name = tmp_path / "out_name.txt"
    # in_state_path points at a file that does not exist.
    rc = worker.run_editor(
        str(tmp_path / "Plugin.vst3"),
        "",
        str(tmp_path / "does_not_exist.txt"),
        str(out_state),
        str(out_name),
        open_editor=lambda p: None,
    )

    assert rc == 0
    assert seen["cfg"]["state_b64"] == ""


def test_empty_in_state_path_restores_empty(tmp_path, monkeypatch) -> None:
    fake = _FakePlugin()
    seen: dict = {}
    monkeypatch.setattr(
        worker, "load_vst3", lambda cfg: seen.update(cfg=dict(cfg)) or fake
    )

    rc = worker.run_editor(
        str(tmp_path / "Plugin.vst3"),
        "",
        "",  # empty in_state_path
        str(tmp_path / "out_state.txt"),
        str(tmp_path / "out_name.txt"),
        open_editor=lambda p: None,
    )

    assert rc == 0
    assert seen["cfg"]["state_b64"] == ""


def test_existing_in_state_passes_through(tmp_path, monkeypatch) -> None:
    fake = _FakePlugin()
    seen: dict = {}

    def _loader(cfg):
        seen["cfg"] = dict(cfg)
        return fake

    monkeypatch.setattr(worker, "load_vst3", _loader)

    in_state = tmp_path / "in_state.txt"
    state_text = base64.b64encode(b"SAVED").decode("ascii")
    in_state.write_text(state_text, encoding="utf-8")

    rc = worker.run_editor(
        str(tmp_path / "Plugin.vst3"),
        "",
        str(in_state),
        str(tmp_path / "out_state.txt"),
        str(tmp_path / "out_name.txt"),
        open_editor=lambda p: None,
    )

    assert rc == 0
    assert seen["cfg"]["state_b64"] == state_text


def test_main_parses_five_positional_args(tmp_path, monkeypatch) -> None:
    captured: dict = {}

    def _fake_run_editor(*args, **kwargs):
        captured["args"] = args
        return 0

    monkeypatch.setattr(worker, "run_editor", _fake_run_editor)
    rc = worker.main(
        ["/p.vst3", "", "/in.txt", "/out_state.txt", "/out_name.txt"]
    )
    assert rc == 0
    assert captured["args"] == (
        "/p.vst3",
        "",
        "/in.txt",
        "/out_state.txt",
        "/out_name.txt",
    )
