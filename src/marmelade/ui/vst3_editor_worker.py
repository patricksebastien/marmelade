"""Out-of-process VST3 native-editor host (quick-260626-fw2).

WHY a separate process: ``pedalboard.VST3Plugin.show_editor()`` is
main-thread-only and BLOCKS the calling thread's event loop until the editor
window is closed (pedalboard 0.9.22). Calling it in-process on Marmelade's Qt
main thread freezes the whole app (and leaves the QFileDialog lingering on
screen) — that is the bug this module fixes. Here the editor runs on THIS
process's own main thread; Marmelade launches the worker via QProcess
(signal-based, non-blocking) so its Qt loop keeps running, then reads back the
captured state + name when this process exits.

This module imports no GUI-toolkit bindings: pedalboard manages its own native
editor window, and spinning up a second application object here would be wrong.
Keep the import surface light.

argv protocol (5 positional args)::

    [plugin_path, plugin_name, in_state_path, out_state_path, out_name_path]

* ``plugin_name`` may be an empty string (single-plugin ``.vst3`` bundle).
* ``in_state_path`` may be an empty string OR a missing file => no saved state
  is restored (``state_b64`` becomes ``""``); if it exists, its UTF-8 text is
  the base64 state blob to restore into the plugin.

On success the worker writes ``base64(plugin.raw_state)`` to ``out_state_path``
and ``plugin.name`` to ``out_name_path`` (both UTF-8 text) and exits 0. If the
plugin is not loadable it writes an error to stderr and exits 2 WITHOUT writing
the out files.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Callable

from marmelade.audio.mastering.stages.vst3 import capture_state_b64, load_vst3

logger = logging.getLogger(__name__)


def _open_editor(plugin: Any) -> None:
    """Open the plugin's native editor (BLOCKS until the window closes).

    Isolated as a thin wrapper so tests can monkeypatch / inject a no-op in
    place of the real GUI-blocking ``show_editor()`` call.
    """
    plugin.show_editor()


def run_editor(
    plugin_path: str,
    plugin_name: str,
    in_state_path: str,
    out_state_path: str,
    out_name_path: str,
    *,
    open_editor: Callable[[Any], None] = _open_editor,
) -> int:
    """Load the plugin, open its editor, capture + persist state and name.

    Returns 0 on success, 2 when the plugin is not loadable. On the unloadable
    path the out files are NOT written.
    """
    state_b64 = ""
    if in_state_path:
        p = Path(in_state_path)
        if p.exists():
            state_b64 = p.read_text(encoding="utf-8")

    cfg = {
        "enabled": True,
        "plugin_path": plugin_path,
        "plugin_name": plugin_name,
        "state_b64": state_b64,
    }

    plugin = load_vst3(cfg)
    if plugin is None:
        print(f"VST3 not loadable: {plugin_path}", file=sys.stderr)
        return 2

    open_editor(plugin)

    captured = capture_state_b64(plugin)
    logger.info(
        "VST3 editor worker: captured %d base64 chars of plugin state for %r "
        "(name=%r) after the editor closed.",
        len(captured),
        plugin_path,
        str(getattr(plugin, "name", "") or ""),
    )
    if not captured:
        logger.warning(
            "VST3 editor worker: plugin %r returned EMPTY raw_state — the "
            "saved config will not carry any editor settings.",
            plugin_path,
        )
    Path(out_state_path).write_text(captured, encoding="utf-8")
    Path(out_name_path).write_text(
        str(getattr(plugin, "name", "") or ""), encoding="utf-8"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse exactly 5 positional args and run the editor."""
    args = list(argv if argv is not None else sys.argv[1:])
    if len(args) != 5:
        print(
            "usage: python -m marmelade.ui.vst3_editor_worker "
            "<plugin_path> <plugin_name> <in_state_path> "
            "<out_state_path> <out_name_path>",
            file=sys.stderr,
        )
        return 2
    plugin_path, plugin_name, in_state_path, out_state_path, out_name_path = args
    return run_editor(
        plugin_path,
        plugin_name,
        in_state_path,
        out_state_path,
        out_name_path,
    )


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
