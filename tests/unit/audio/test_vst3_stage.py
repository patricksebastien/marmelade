"""quick-260625 — external VST3 mastering stage (load/apply/hash/persist).

Covers the backend contract without a real ``.vst3`` on disk: a fake loader
returns a fake plugin so we can pin parameter passing, state restore/capture,
passthrough semantics, config_hash behavior, chain integration, and sidecar
acceptance. The interactive file-picker + native editor flow
(:mod:`marmelade.ui.vst3_config`) is GUI-only and not exercised here.
"""

from __future__ import annotations

import base64
import copy

import numpy as np
import pytest

import marmelade.audio.mastering.chain as cmod
import marmelade.audio.mastering.stages.vst3 as vmod
from marmelade.audio.mastering.chain import (
    _SESSION_DEFAULTS,
    _STAGE_ORDER,
    config_hash,
)


class _FakePlugin:
    """Minimal pedalboard-plugin stand-in with raw_state + callable render."""

    def __init__(self, name: str = "FakeVST") -> None:
        self.name = name
        self._raw = b""
        self.calls: list[tuple] = []

    @property
    def raw_state(self) -> bytes:
        return self._raw

    @raw_state.setter
    def raw_state(self, value: bytes) -> None:
        self._raw = bytes(value)

    def __call__(self, audio: np.ndarray, sr: int) -> np.ndarray:
        self.calls.append((audio.shape, sr))
        return audio * 0.5


# --------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------
def test_vst3_registered_in_order_and_defaults() -> None:
    assert "vst3" in _STAGE_ORDER
    d = _SESSION_DEFAULTS["vst3"]
    assert d == {
        "enabled": False,
        "plugin_path": "",
        "plugin_name": "",
        "state_b64": "",
    }


# --------------------------------------------------------------------------
# load_vst3
# --------------------------------------------------------------------------
def test_load_vst3_missing_path_returns_none() -> None:
    assert vmod.load_vst3({"plugin_path": ""}) is None
    assert vmod.load_vst3({"plugin_path": "/nonexistent/x.vst3"}) is None


def test_load_vst3_accepts_directory_bundle(tmp_path) -> None:
    """A .vst3 is a DIRECTORY bundle on Linux/macOS — the gate must accept it."""
    bundle = tmp_path / "Plugin.vst3"
    bundle.mkdir()
    fake = _FakePlugin()
    plugin = vmod.load_vst3(
        {"enabled": True, "plugin_path": str(bundle)},
        loader=lambda p, plugin_name=None: fake,
    )
    assert plugin is fake


def test_load_vst3_restores_state_and_passes_name(tmp_path) -> None:
    f = tmp_path / "p.vst3"
    f.write_bytes(b"x")
    fake = _FakePlugin()
    captured: dict = {}

    def loader(path, plugin_name=None):
        captured["path"] = path
        captured["name"] = plugin_name
        return fake

    cfg = {
        "enabled": True,
        "plugin_path": str(f),
        "plugin_name": "SubPlugin",
        "state_b64": base64.b64encode(b"STATE").decode("ascii"),
    }
    plugin = vmod.load_vst3(cfg, loader=loader)
    assert plugin is fake
    assert captured["path"] == str(f)
    assert captured["name"] == "SubPlugin"
    assert fake.raw_state == b"STATE"


def test_load_vst3_tolerates_corrupt_state(tmp_path) -> None:
    f = tmp_path / "p.vst3"
    f.write_bytes(b"x")
    fake = _FakePlugin()
    cfg = {"enabled": True, "plugin_path": str(f), "state_b64": "@@not-base64@@"}
    # Corrupt blob must not raise — falls back to the plugin's default state.
    plugin = vmod.load_vst3(cfg, loader=lambda p, plugin_name=None: fake)
    assert plugin is fake
    assert fake.raw_state == b""


def test_capture_state_b64_round_trips() -> None:
    fake = _FakePlugin()
    fake.raw_state = b"HELLO-STATE"
    assert base64.b64decode(vmod.capture_state_b64(fake)) == b"HELLO-STATE"


# --------------------------------------------------------------------------
# apply_vst3
# --------------------------------------------------------------------------
def test_apply_vst3_disabled_is_passthrough() -> None:
    audio = np.ones((2, 100), dtype=np.float32)
    out = vmod.apply_vst3(audio, 48000, {"enabled": False})
    assert out is audio


def test_apply_vst3_missing_file_is_passthrough() -> None:
    audio = np.ones((2, 100), dtype=np.float32)
    out = vmod.apply_vst3(
        audio, 48000, {"enabled": True, "plugin_path": "/nope.vst3"}
    )
    assert out is audio


def test_apply_vst3_enabled_runs_plugin(tmp_path) -> None:
    f = tmp_path / "p.vst3"
    f.write_bytes(b"x")
    fake = _FakePlugin()
    audio = np.ones((2, 100), dtype=np.float32)
    out = vmod.apply_vst3(
        audio,
        48000,
        {"enabled": True, "plugin_path": str(f)},
        loader=lambda p, plugin_name=None: fake,
    )
    assert np.allclose(out, audio * 0.5)
    assert fake.calls == [((2, 100), 48000)]


# --------------------------------------------------------------------------
# Vst3Stage
# --------------------------------------------------------------------------
def test_stage_has_no_auto_params() -> None:
    assert vmod.Vst3Stage().parameters() == {}


def test_stage_build_plugin_uses_load(monkeypatch) -> None:
    fake = _FakePlugin()
    monkeypatch.setattr(vmod, "load_vst3", lambda cfg, loader=None: fake)
    stage = vmod.Vst3Stage()
    stage._param_overrides = {"plugin_path": "/p.vst3", "state_b64": "AAAA"}
    assert stage.build_plugin() is fake


def test_stage_build_plugin_raises_when_unloadable(monkeypatch) -> None:
    monkeypatch.setattr(vmod, "load_vst3", lambda cfg, loader=None: None)
    with pytest.raises(FileNotFoundError):
        vmod.Vst3Stage().build_plugin()


# --------------------------------------------------------------------------
# config_hash
# --------------------------------------------------------------------------
def test_disabled_vst3_drops_from_hash() -> None:
    base = {"limiter": {"enabled": True, "ceiling_dbtp": -1.0, "release_ms": 100.0}}
    cfg_no_key = dict(base)
    cfg_disabled = dict(
        base,
        vst3={
            "enabled": False,
            "plugin_path": "/p.vst3",
            "plugin_name": "x",
            "state_b64": "AAAA",
        },
    )
    # A disabled vst3 hashes EQUAL to a config without the key (preset match).
    assert config_hash(cfg_no_key) == config_hash(cfg_disabled)


def test_enabled_vst3_state_changes_hash() -> None:
    base = {"limiter": {"enabled": True, "ceiling_dbtp": -1.0, "release_ms": 100.0}}
    a = dict(base, vst3={"enabled": True, "plugin_path": "/p.vst3", "state_b64": "AAAA"})
    b = dict(base, vst3={"enabled": True, "plugin_path": "/p.vst3", "state_b64": "BBBB"})
    assert config_hash(a) != config_hash(b)


# --------------------------------------------------------------------------
# Chain integration
# --------------------------------------------------------------------------
def test_chain_process_applies_vst3_before_limiter(monkeypatch) -> None:
    calls: list[dict] = []

    def spy(audio, sr, cfg, cancel_check=None):
        calls.append(cfg)
        return audio * 0.25

    monkeypatch.setattr(cmod, "apply_vst3", spy)
    cfg = copy.deepcopy(_SESSION_DEFAULTS)
    cfg["limiter"]["enabled"] = False  # keep the chain a pure passthrough
    cfg["vst3"] = {
        "enabled": True,
        "plugin_path": "/p.vst3",
        "plugin_name": "",
        "state_b64": "",
    }
    audio = np.full((2, 48000), 0.5, dtype=np.float32)
    out = cmod.MasteringChain(cfg).process(audio, 48000)
    assert calls and calls[0]["enabled"] is True
    # vst3 quartered the only-active stage's output; nothing else touched it.
    assert np.allclose(out, 0.5 * 0.25)


# --------------------------------------------------------------------------
# End-to-end cfg-by-reference persistence (configure → keeper.mastering →
# MasteringChain). Proves the whole apply→master data path the GUI relies on,
# WITHOUT a real plugin/editor — the failure the user reports is NOT here.
# --------------------------------------------------------------------------
def test_configure_by_reference_persists_into_keeper_mastering(tmp_path) -> None:
    """Simulate the GUI seam: configure_vst3's finished handler mutates the
    SAME dict the dialog holds (by reference); that dict becomes
    keeper.mastering and must drive MasteringChain to run the plugin.

    This mirrors vst3_config._on_finished setting enabled/plugin_path/state_b64
    on the dict the MasteringDialog passed in, then Apply emitting that dict.
    """
    bundle = tmp_path / "oXygen.vst3"
    bundle.mkdir()

    # keeper.mastering as the dialog holds it: a per-stage dict, vst3 starts
    # as the disabled default (the dock/dialog seed).
    keeper_mastering = copy.deepcopy(_SESSION_DEFAULTS)
    keeper_mastering["limiter"]["enabled"] = False  # isolate the vst3 effect
    vst3_cfg = keeper_mastering["vst3"]  # the SAME ref the gear handler mutates

    # Before configuring: chain is a pure passthrough (vst3 disabled).
    audio = np.full((2, 48000), 0.5, dtype=np.float32)
    fake = _FakePlugin()
    out_before = cmod.MasteringChain(
        copy.deepcopy(keeper_mastering)
    ).process(audio.copy(), 48000)
    assert np.allclose(out_before, 0.5)  # untouched

    # configure_vst3 finished handler mutates vst3_cfg BY REFERENCE.
    vst3_cfg["plugin_path"] = str(bundle)
    vst3_cfg["plugin_name"] = "oXygen"
    vst3_cfg["state_b64"] = base64.b64encode(b"OXYGEN-STATE").decode("ascii")
    vst3_cfg["enabled"] = True

    # Apply emits keeper_mastering; the chain must now run the plugin.
    import marmelade.audio.mastering.chain as _c

    def _loader(path, plugin_name=None):
        assert path == str(bundle)
        assert plugin_name == "oXygen"
        return fake

    # Route the chain's apply_vst3 through the injectable loader (the real
    # production path uses the default pedalboard loader; here we fake it).
    orig_apply = vmod.apply_vst3
    try:
        _c.apply_vst3 = lambda a, sr, cfg, cc=None: orig_apply(
            a, sr, cfg, cc, loader=_loader
        )
        out_after = _c.MasteringChain(
            copy.deepcopy(keeper_mastering)
        ).process(audio.copy(), 48000)
    finally:
        _c.apply_vst3 = orig_apply

    assert np.allclose(out_after, 0.5 * 0.5)  # plugin halved the audio
    assert fake.calls == [((2, 48000), 48000)]
    assert fake.raw_state == b"OXYGEN-STATE"  # editor state restored at master


# --------------------------------------------------------------------------
# Sidecar acceptance
# --------------------------------------------------------------------------
def test_sidecar_accepts_vst3_stage() -> None:
    from marmelade.audio.sidecar_cache import _validate_mastering_dict

    # Must not raise — "vst3" is an allowed _STAGE_ORDER stage with enabled bool.
    _validate_mastering_dict(
        {
            "vst3": {
                "enabled": True,
                "plugin_path": "/p.vst3",
                "plugin_name": "",
                "state_b64": "AAAA",
            }
        }
    )
