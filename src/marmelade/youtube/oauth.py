"""OAuth 2.0 + keyring persistence for YouTube Data API v3 (Phase 8 YT-01).

Drives Google's installed-app loopback flow against the bundled Marmelade
OAuth client (D-07), persists the resulting refresh-token bearing
:class:`google.oauth2.credentials.Credentials` JSON to the OS keychain via
:mod:`keyring` (D-08), and falls back to a ``platformdirs``-resolved
plaintext file (mode ``0o600``) when no keyring backend is available
(Pitfall 3 — RESEARCH lines 654-675).

Public surface (consumed by :mod:`marmelade.ui.settings_dialog` +
Plan 08-04 upload runnable):

* :func:`first_time_connect` — blocking, opens the system browser.
* :func:`load_or_refresh` — returns ``None`` when no credentials stored,
  otherwise re-raises :class:`google.auth.exceptions.RefreshError` on a
  hard-refresh failure (the dialog distinguishes "not connected" from
  "refresh failed" at upload time per D-25).
* :func:`disconnect` — best-effort POST to ``oauth2.googleapis.com/revoke``
  followed by ``keyring.delete_password`` + plaintext-fallback file removal
  (D-09 channel-switch UX).
* :func:`channel_info` — wraps ``youtube.channels().list(mine=True)`` and
  returns ``{"title": ..., "id": ...}``. Used by SettingsDialog to render
  ``"Connected as <name>"``.
* :func:`is_connected` — display-only convenience; swallows
  :class:`RefreshError` as ``False`` (Plan 08-04 distinguishes at upload).

Threat model anchors (08-02-PLAN.md ``<threat_model>``):

* T-08-02-01 (refresh token at rest) → mitigated by ``keyring`` →
  OS-keychain, fallback file mode ``0o600`` + ``log.warning``.
* T-08-02-02 (loopback redirect URI hijack) → mitigated by
  ``run_local_server(port=0)`` — random free port.
* T-08-02-03 (scope creep) → :data:`SCOPES` is the frozen module-level
  constant; the plan grep-pins its value.
* T-08-02-04 (plaintext fallback world-readable) → ``os.umask(0o077)``
  before ``open`` + defensive ``os.chmod(0o600)`` after write.
* T-08-02-05 (client_secret extraction) → ACCEPTED — per Google's
  installed-app threat model, the embedded ``client_secret`` is NOT
  confidential; auth boundary is the per-user refresh token in keyring.
* T-08-02-06 (credentials in stack trace) → never log
  ``creds.to_json()`` / ``creds.token``; surface ``str(e)`` only.

N-3 (Qt-free package) — this module imports zero ``PySide6.*`` symbols.

Re-export discipline — ``InstalledAppFlow``, ``Credentials``, ``Request``,
``urlopen`` are imported (not re-exported) at module scope so tests can
``patch.object(oauth, ...)`` cleanly without touching the third-party
package's import tree (mirrors :mod:`marmelade.audio.mastering_cache`
test-friendly import pattern).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request as _UrllibRequest  # noqa: F401 — kept for caller monkeypatch parity
from urllib.request import urlopen

import keyring
import keyring.errors
import platformdirs
from google.auth.exceptions import RefreshError  # noqa: F401 — re-exported for callers
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level constants — frozen contract for downstream plans + grep gates.
# ---------------------------------------------------------------------------

# T-08-02-03 mitigation: minimum scopes only.
# - ``youtube.upload`` (sensitive): required for ``videos().insert`` +
#   ``thumbnails().set`` — the actual upload path.
# - ``youtube.readonly`` (sensitive, NOT restricted): required for
#   ``channels().list(part='snippet', mine=True)`` which powers the
#   ``Connected as <channel name>`` UI in the Settings panel (D-10).
#
# Both scopes sit in Google's "sensitive" tier (developers.google.com/
# identity/protocols/oauth2/scopes#youtube). Neither is in the stricter
# "restricted" tier that mandates a security assessment. Adding readonly
# does NOT meaningfully change the verification posture beyond what
# upload already requires.
#
# Adding broader scopes (``youtube`` / ``youtube.force-ssl``) WOULD
# expand surface area — those grant write/delete on the channel and are
# intentionally not requested.
SCOPES: list[str] = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.readonly",
]

KEYRING_SERVICE: str = "Marmelade"
KEYRING_USER: str = "youtube-oauth"

_REVOKE_URL: str = "https://oauth2.googleapis.com/revoke"

# D-07 — Marmelade app-level OAuth client (installed-app flow).
#
# The client_id / client_secret are NOT embedded in source — they are read
# from the environment (``MARMELADE_YT_CLIENT_ID`` / ``MARMELADE_YT_CLIENT_SECRET``)
# so no credential is committed to the repo. Set both env vars to enable the
# YouTube "Connect" flow; with them unset the flow raises ``invalid_client``
# at Google (the code is otherwise exercised by mocks in
# tests/youtube/test_oauth.py).
#
# T-08-02-05 (ACCEPTED): per Google's installed-app threat model
# (developers.google.com/identity/protocols/oauth2/native-app), the client
# secret is not a true confidentiality boundary — the per-user refresh_token
# in keyring is. We still keep it out of source control.
CLIENT_CONFIG: dict[str, dict[str, Any]] = {
    "installed": {
        "client_id": os.environ.get("MARMELADE_YT_CLIENT_ID", ""),
        "client_secret": os.environ.get("MARMELADE_YT_CLIENT_SECRET", ""),
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "redirect_uris": ["http://127.0.0.1"],
    }
}

# Cross-platform plaintext fallback path (N-1 reconciliation, revision
# iter 1 — pyproject.toml pins ``platformdirs>=4,<5`` explicitly).
# Linux:   ~/.config/Marmelade/youtube_oauth.json
# macOS:   ~/Library/Application Support/Marmelade/youtube_oauth.json
# Windows: %LOCALAPPDATA%\Marmelade\youtube_oauth.json
_FALLBACK_PATH: Path = (
    Path(platformdirs.user_config_dir("Marmelade")) / "youtube_oauth.json"
)

# Pitfall 3 — emit the "your token is in plaintext at rest" warning ONCE
# per session so the user sees it the first time the fallback engages and
# isn't spammed on every load_or_refresh call.
_warned_about_fallback: bool = False


# ---------------------------------------------------------------------------
# Plaintext fallback helpers (T-08-02-04 mitigation).
# ---------------------------------------------------------------------------


def _emit_fallback_warning(reason: type[BaseException]) -> None:
    """Surface a user-visible warning the first time the fallback engages.

    Pitfall 3 — distinguishes the keyring backend failure mode
    (``NoKeyringError`` vs ``InitError``) so a Linux user can tell whether
    the secret-service daemon is missing entirely or just failed to
    initialize this session.
    """
    global _warned_about_fallback
    if _warned_about_fallback:
        return
    _warned_about_fallback = True
    log.warning(
        "Keyring backend unavailable (%s); using plaintext fallback at %s. "
        "Token will NOT be encrypted at rest. Install libsecret on Linux to "
        "enable Secret Service.",
        reason.__name__,
        _FALLBACK_PATH,
    )


def _load_from_plaintext_fallback(path: Path) -> str | None:
    """Read the JSON blob from ``path``; return ``None`` if absent."""
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None


def _save_to_plaintext_fallback(path: Path, json_str: str) -> None:
    """Write ``json_str`` to ``path`` with mode 0600 (T-08-02-04).

    Uses ``os.umask(0o077)`` before opening + defensive ``os.chmod(0o600)``
    after write so the file is never world-readable, even momentarily.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # umask shifts so the create-mode of the new file masks group+world bits.
    previous_umask = os.umask(0o077)
    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(json_str)
    finally:
        os.umask(previous_umask)
    # Defensive — chmod after-write covers the case where umask is overridden
    # by the platform (e.g., NFS volumes) or by the user's process inheritance.
    try:
        os.chmod(path, 0o600)
    except OSError:
        # Best-effort: filesystems that don't support chmod (e.g., FAT) just
        # surface the umask-applied mode.
        pass


def _delete_plaintext_fallback(path: Path) -> None:
    """Remove ``path`` if it exists; silently no-op otherwise."""
    try:
        path.unlink()
    except FileNotFoundError:
        pass


# ---------------------------------------------------------------------------
# Keyring read / write / delete with fallback.
# ---------------------------------------------------------------------------


def _read_token_json() -> str | None:
    """Return the persisted credentials JSON or ``None`` if not stored.

    Engages :func:`_load_from_plaintext_fallback` when the keyring backend
    raises :class:`keyring.errors.NoKeyringError` or
    :class:`keyring.errors.InitError` (Pitfall 3).
    """
    try:
        raw = keyring.get_password(KEYRING_SERVICE, KEYRING_USER)
        return raw
    except (keyring.errors.NoKeyringError, keyring.errors.InitError) as e:
        _emit_fallback_warning(type(e))
        return _load_from_plaintext_fallback(_FALLBACK_PATH)


def _write_token_json(json_str: str) -> None:
    """Persist credentials JSON to keyring, falling back to plaintext."""
    try:
        keyring.set_password(KEYRING_SERVICE, KEYRING_USER, json_str)
    except (keyring.errors.NoKeyringError, keyring.errors.InitError) as e:
        _emit_fallback_warning(type(e))
        _save_to_plaintext_fallback(_FALLBACK_PATH, json_str)


def _clear_token_json() -> None:
    """Remove credentials from keyring + plaintext fallback (both paths)."""
    try:
        keyring.delete_password(KEYRING_SERVICE, KEYRING_USER)
    except (keyring.errors.NoKeyringError, keyring.errors.InitError) as e:
        _emit_fallback_warning(type(e))
    # Always try to clean up the fallback file too — D-09 channel-switch UX
    # requires every persistence path is empty after disconnect.
    _delete_plaintext_fallback(_FALLBACK_PATH)


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------


def first_time_connect() -> Credentials:
    """Drive Google's installed-app loopback flow + persist the result.

    Blocks while the system browser presents Google's consent screen and
    the user clicks Allow. The loopback listener binds to a random free
    port (``port=0``) so an attacker cannot pre-bind a known port
    (T-08-02-02). On success the returned :class:`Credentials` JSON
    (refresh_token included) is persisted to keyring (or the plaintext
    fallback) and the object is returned to the caller.
    """
    flow = InstalledAppFlow.from_client_config(CLIENT_CONFIG, SCOPES)
    creds = flow.run_local_server(port=0)
    _write_token_json(creds.to_json())
    return creds


def load_or_refresh() -> Credentials | None:
    """Return persisted credentials, silently refreshing the access token.

    Returns ``None`` when no credentials are stored (first-time-use case).
    Raises :class:`google.auth.exceptions.RefreshError` when the
    refresh-token-driven access-token refresh fails — callers MUST handle
    this (the SettingsDialog uses :func:`is_connected` which swallows it
    for display, but Plan 08-04 surfaces the "Reconnect YouTube" UX at
    upload time per D-25).
    """
    raw = _read_token_json()
    if raw is None:
        return None
    creds = Credentials.from_authorized_user_info(json.loads(raw), SCOPES)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())  # raises RefreshError on hard failure
        _write_token_json(creds.to_json())
    return creds


def disconnect() -> None:
    """Revoke the access token (best-effort) + clear local persistence.

    D-09 channel-switch UX: after disconnect the user is "Not connected"
    in the SettingsDialog and can immediately Connect again with a
    different account. The revoke POST to ``oauth2.googleapis.com/revoke``
    is best-effort — network failures are swallowed because local state is
    the source of truth (T-08-02-10 disposition).
    """
    raw = _read_token_json()
    if raw:
        try:
            creds = Credentials.from_authorized_user_info(json.loads(raw), SCOPES)
            if creds.token:
                # urlencode → bytes → POST body. ``urlopen`` with a ``data``
                # kwarg auto-selects POST.
                body = urlencode({"token": creds.token}).encode("utf-8")
                with urlopen(_REVOKE_URL, data=body, timeout=10):
                    pass
        except Exception:
            # T-08-02-10 — revoke is best-effort. Never block disconnect on
            # a network hiccup; the local state is still cleared below.
            log.info("YouTube revoke POST failed (ignored — local state cleared).")
    _clear_token_json()


def channel_info(creds: Credentials) -> dict[str, str]:
    """Return ``{"title": ..., "id": ...}`` for the authenticated channel.

    Used by SettingsDialog to render ``"Connected as <name>"``. Builds the
    YouTube discovery client via :func:`marmelade.youtube.client.build_youtube_client`
    + calls ``channels().list(part="snippet", mine=True).execute()``. The
    network call is wrapped at the caller level — any error here propagates.
    """
    # Local import to avoid a top-level cycle on first import of the
    # ``youtube`` package + to keep this module's import-time cost minimal.
    from marmelade.youtube.client import build_youtube_client

    client = build_youtube_client(creds)
    resp = client.channels().list(part="snippet", mine=True).execute()
    items = resp.get("items") or []
    if not items:
        return {"title": "", "id": ""}
    snippet = items[0].get("snippet") or {}
    return {
        "title": str(snippet.get("title", "")),
        "id": str(items[0].get("id", "")),
    }


def is_connected() -> bool:
    """Return ``True`` when persisted credentials load + refresh successfully.

    Display-only convenience for SettingsDialog. Swallows
    :class:`RefreshError` as ``False`` — Plan 08-04 distinguishes
    "refresh-failed-needs-reconnect" from "never-connected" at upload time.
    """
    try:
        return load_or_refresh() is not None
    except RefreshError:
        return False
