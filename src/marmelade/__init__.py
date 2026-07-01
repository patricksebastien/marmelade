"""Marmelade — desktop DAW-style waveform viewer for long jam recordings."""

from __future__ import annotations

import os
import subprocess
from typing import Optional

__version__ = "0.1.0"


def _compute_build_sha() -> str:
    """Best-effort short git SHA of the running source tree.

    Returns 'unknown' if anything fails (no git, detached install, no
    .git directory). Used by MainWindow's status-bar footer so the user
    can verify which build is actually running — surfaced in response to
    Phase 2.1 HUMAN-UAT request: "maybe adding Version: XYZ in the
    footer would help" (so the user can confirm a restart picked up
    the latest commits).
    """
    repo_root: Optional[str] = None
    here = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.dirname(os.path.dirname(here))
    if os.path.isdir(os.path.join(candidate, ".git")):
        repo_root = candidate
    if repo_root is None:
        return "unknown"
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            timeout=2.0,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return "unknown"
    return out.decode("utf-8", errors="replace").strip() or "unknown"


__build__ = _compute_build_sha()
