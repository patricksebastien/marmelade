"""VST3 plugin mastering stage (quick-260625) — load an external VST3.

Lets the user drop any compiled VST3 mastering plugin (e.g. the open-source
oXygen, https://github.com/Wamphyre/oXygen) into the per-keeper chain.
``pedalboard.load_plugin`` loads the ``.vst3`` headlessly; the plugin's
opaque state (its full settings, captured from the native editor) is
persisted as base64 in the keeper's mastering config so renders are
deterministic and the mastered cache keys on it via ``config_hash``.

Config dict shape (one entry in the keeper ``mastering`` dict)::

    "vst3": {
        "enabled": bool,
        "plugin_path": str,   # absolute path to a .vst3 file
        "plugin_name": str,   # sub-plugin name (multi-plugin .vst3 bundles)
        "state_b64": str,     # base64 of plugin.raw_state (native editor output)
    }

The chain applies this stage via :func:`apply_vst3` (mirroring how
``ending_fx`` is applied directly), NOT through the pedalboard-list factory,
so it can run between the built-in pre-limiter chain and the true-peak
limiter.

Security note (single-user desktop threat model): :func:`load_vst3` loads a
shared library from ``plugin_path``, which executes plugin code. A sidecar
from a hostile party could point ``plugin_path`` anywhere. We only load when
the stage is ENABLED and the file EXISTS; the broader app already opens
user-chosen files and the local filesystem is treated as user-owned (the
same model as the matchering reference path). Do not start auto-loading
plugin paths from untrusted sidecars without revisiting this.

N-3 invariant: zero PySide6 imports — the interactive file-picker + native
editor flow lives in :mod:`marmelade.ui.vst3_config`.
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pedalboard

from marmelade.audio.mastering.base import MasteringStage
from marmelade.audio.mastering.params import Param

logger = logging.getLogger(__name__)

# Injectable loader signature so tests can substitute a fake without a real
# ``.vst3`` on disk. The default delegates to pedalboard.load_plugin.
PluginLoader = Callable[..., Any]


def _rms(audio: np.ndarray) -> float:
    """Root-mean-square level of a sample buffer (for passthrough diagnostics)."""
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))


def _default_loader(path: str, plugin_name: Optional[str] = None) -> Any:
    """Default loader — pedalboard.load_plugin with an optional sub-name."""
    if plugin_name:
        return pedalboard.load_plugin(path, plugin_name=plugin_name)
    return pedalboard.load_plugin(path)


def load_vst3(
    cfg: dict[str, Any], loader: PluginLoader = _default_loader
) -> Any | None:
    """Load the configured VST3 and restore its saved state.

    Returns the constructed pedalboard plugin, or ``None`` when the stage is
    not loadable (empty ``plugin_path`` or the path is missing). Loading is
    intentionally tolerant: a corrupt/incompatible ``state_b64`` blob falls
    back to the plugin's own default state rather than failing the render.

    Note: a ``.vst3`` is a DIRECTORY bundle on Linux/macOS and a single file
    on Windows, so the existence gate accepts EITHER (``exists()``, not
    ``is_file()``) — pedalboard.load_plugin takes the bundle path directly.
    """
    path = str(cfg.get("plugin_path", "") or "")
    if not path:
        logger.warning(
            "VST3 stage: no plugin_path configured (cfg keys=%s) — "
            "the plugin slot is enabled but never had a .vst3 picked; "
            "audio passes through unchanged.",
            sorted(cfg.keys()),
        )
        return None
    if not Path(path).expanduser().exists():
        logger.warning(
            "VST3 stage: plugin_path does not exist on disk: %r — "
            "audio passes through unchanged.",
            path,
        )
        return None
    plugin_name = str(cfg.get("plugin_name", "") or "") or None
    try:
        plugin = loader(path, plugin_name=plugin_name)
    except TypeError:
        # A loader (or fake) that does not accept the plugin_name kwarg.
        plugin = loader(path)
    state_b64 = str(cfg.get("state_b64", "") or "")
    if state_b64:
        try:
            plugin.raw_state = base64.b64decode(state_b64)
        except Exception:
            # Corrupt or version-incompatible blob — keep the plugin's
            # default state rather than aborting the whole render.
            logger.warning(
                "VST3 stage: failed to restore saved plugin state "
                "(%d base64 chars) for %r — using the plugin's DEFAULT "
                "state, so your editor settings will NOT be applied.",
                len(state_b64),
                path,
            )
    else:
        logger.info(
            "VST3 stage: loaded %r with EMPTY saved state (state_b64='') — "
            "running the plugin at its default settings.",
            path,
        )
    return plugin


def capture_state_b64(plugin: Any) -> str:
    """Return the plugin's ``raw_state`` as base64 text (for persistence)."""
    raw = getattr(plugin, "raw_state", b"") or b""
    return base64.b64encode(bytes(raw)).decode("ascii")


def apply_vst3(
    audio: np.ndarray,
    sr: int,
    cfg: dict[str, Any],
    cancel_check: Callable[[], bool] | None = None,
    loader: PluginLoader = _default_loader,
) -> np.ndarray:
    """Run ``audio`` through the configured VST3 when enabled; else passthrough.

    Disabled / unconfigured / missing-file all return ``audio`` unchanged so
    a keeper that never loaded a plugin is a byte-identical no-op.

    Args:
        audio: ``(num_channels, num_samples)`` float32.
        sr: Sample rate (the chain enforces 48000).
        cfg: The keeper's ``mastering["vst3"]`` dict.
        cancel_check: Optional cancel poll (checked before the load — the
            pedalboard call itself is not interruptible, RESEARCH §Pitfall 5).
        loader: Injectable plugin loader (tests pass a fake).
    """
    if not cfg.get("enabled", False):
        logger.info(
            "VST3 stage: disabled (enabled=%r) — audio passes through "
            "unchanged. If you configured a plugin but it is not applied, "
            "make sure you closed the plugin editor BEFORE clicking Apply.",
            cfg.get("enabled", False),
        )
        return audio
    if cancel_check is not None and cancel_check():
        # Mirror chain.BuildCancelled semantics without importing it here:
        # the orchestrator polls again right after this call, so a plain
        # passthrough is safe — but prefer to stop work early.
        logger.info("VST3 stage: cancelled before plugin load — passthrough.")
        return audio
    plugin = load_vst3(cfg, loader=loader)
    if plugin is None:
        # load_vst3 already logged the specific reason (no path / missing file).
        return audio
    rms_in = _rms(audio)
    out = plugin(audio, sr)
    rms_out = _rms(out)
    if abs(rms_out - rms_in) < 1e-9:
        logger.warning(
            "VST3 stage: plugin %r ran but did NOT change the audio "
            "(RMS in=%.6f == out=%.6f). The plugin may be bypassed, or its "
            "restored state has no audible effect on this material.",
            cfg.get("plugin_path", ""),
            rms_in,
            rms_out,
        )
    else:
        logger.info(
            "VST3 stage: plugin %r applied (RMS in=%.6f -> out=%.6f).",
            cfg.get("plugin_path", ""),
            rms_in,
            rms_out,
        )
    return out


class Vst3Stage(MasteringStage):
    """External VST3 plugin slot in the per-keeper mastering chain.

    Configured via the plugin's NATIVE editor (see
    :func:`marmelade.ui.vst3_config.configure_vst3`), not the auto-rendered
    ParamsDialog — so :meth:`parameters` returns an empty dict (the gear
    button opens the bespoke picker + editor flow instead). The chain applies
    this stage through :func:`apply_vst3`; :meth:`build_plugin` is provided
    for ABC completeness and direct callers.
    """

    name = "vst3"
    display_name = "VST3 plugin"

    def parameters(self) -> dict[str, Param]:
        # No auto-rendered params — the native editor owns configuration.
        return {}

    def build_plugin(self) -> pedalboard.Plugin:
        cfg = {
            "enabled": True,
            "plugin_path": self._get("plugin_path", ""),
            "plugin_name": self._get("plugin_name", ""),
            "state_b64": self._get("state_b64", ""),
        }
        plugin = load_vst3(cfg)
        if plugin is None:
            raise FileNotFoundError(
                f"VST3 plugin not loadable: {cfg.get('plugin_path')!r}"
            )
        return plugin
