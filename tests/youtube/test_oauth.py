"""Phase 8 Plan 08-02 — OAuth + keychain (YT-01) GREEN tests.

Pin the contract published by :mod:`marmelade.youtube.oauth`:

* :func:`first_time_connect` — drives ``InstalledAppFlow.run_local_server``
  and persists the returned Credentials JSON to keyring under service
  ``"Marmelade"`` / username ``"youtube-oauth"`` (D-07 bundled
  installed-app client).
* :func:`load_or_refresh` — re-raises ``google.auth.exceptions.RefreshError``
  on a hard-refresh failure (D-25 caller-decides surface).
  :func:`is_connected` swallows the same error and returns False.
* :func:`disconnect` — POSTs the token to ``oauth2.googleapis.com/revoke``
  and clears keyring + plaintext fallback (D-09 channel-switch UX).
* Keyring fallback (D-08) — ``keyring.errors.InitError`` or
  ``NoKeyringError`` triggers ``_FALLBACK_PATH`` read with a ``log.warning``
  surfaced ONCE per session.

RESEARCH §"Phase Requirements -> Test Map" lines 982-1003 — YT-01 row.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import keyring.errors

from marmelade.youtube import oauth


# ---------------------------------------------------------------------------
# Test 1 — first_time_connect drives the InstalledAppFlow + writes to keyring.
# ---------------------------------------------------------------------------


def test_first_time_connect(mock_credentials_factory, fake_keyring_backend) -> None:
    """``first_time_connect`` opens the loopback flow + persists JSON to keyring."""
    fake_creds = mock_credentials_factory()
    # The two real moving parts of `first_time_connect` are
    # ``InstalledAppFlow.from_client_config`` (constructor) and the
    # returned flow object's ``run_local_server`` method. Mock both:
    fake_flow = MagicMock(name="InstalledAppFlow")
    fake_flow.run_local_server.return_value = fake_creds

    with patch.object(
        oauth.InstalledAppFlow, "from_client_config", return_value=fake_flow
    ) as ctor:
        result = oauth.first_time_connect()

    # Flow was constructed with CLIENT_CONFIG + SCOPES, NOT bare strings.
    ctor.assert_called_once_with(oauth.CLIENT_CONFIG, oauth.SCOPES)
    # ``run_local_server(port=0)`` so an attacker cannot pre-bind the port
    # (T-08-02-02 mitigation).
    fake_flow.run_local_server.assert_called_once_with(port=0)
    # Credentials are persisted to keyring (D-08).
    stored = fake_keyring_backend.get_password(
        oauth.KEYRING_SERVICE, oauth.KEYRING_USER
    )
    assert stored == fake_creds.to_json()
    # ``first_time_connect`` returns the Credentials object.
    assert result is fake_creds


# ---------------------------------------------------------------------------
# Test 2 — RefreshError surfaces vs is_connected swallowing it.
# ---------------------------------------------------------------------------


def test_refresh_error_surfaces(mock_credentials_factory, fake_keyring_backend) -> None:
    """Expired creds whose refresh fails re-raise from ``load_or_refresh``."""
    from google.auth.exceptions import RefreshError

    expired = mock_credentials_factory(expired=True)
    # Override the fixture's auto-success refresh side_effect with a raise.
    expired.refresh.side_effect = RefreshError("Token has been expired or revoked")
    # Seed keyring with the expired JSON.
    fake_keyring_backend.set_password(
        oauth.KEYRING_SERVICE, oauth.KEYRING_USER, expired.to_json()
    )
    # Patch ``Credentials.from_authorized_user_info`` to return the expired
    # MagicMock so the production code doesn't try to parse the (fake) JSON.
    with patch.object(
        oauth.Credentials,
        "from_authorized_user_info",
        return_value=expired,
    ):
        # 1. load_or_refresh RE-RAISES (D-25 caller-decides UX surface).
        with pytest.raises(RefreshError):
            oauth.load_or_refresh()
        # 2. is_connected SWALLOWS the same error and returns False (it's a
        #    display-only convenience used by SettingsDialog).
        assert oauth.is_connected() is False


# ---------------------------------------------------------------------------
# Test 3 — disconnect POSTs to revoke endpoint + clears keyring.
# ---------------------------------------------------------------------------


def test_disconnect(fake_keyring_backend, mock_credentials_factory) -> None:
    """``disconnect`` POSTs to oauth2.revoke + ``keyring.delete_password``."""
    creds = mock_credentials_factory()
    fake_keyring_backend.set_password(
        oauth.KEYRING_SERVICE, oauth.KEYRING_USER, creds.to_json()
    )
    # The disconnect path calls Credentials.from_authorized_user_info(...).token
    # to obtain the access token to revoke.
    with patch.object(
        oauth.Credentials, "from_authorized_user_info", return_value=creds
    ), patch.object(oauth, "urlopen") as fake_urlopen:
        fake_urlopen.return_value.__enter__ = lambda self: self
        fake_urlopen.return_value.__exit__ = lambda *a: None
        oauth.disconnect()

    # Revoke URL was POSTed with the token in the body.
    assert fake_urlopen.call_count == 1
    args, kwargs = fake_urlopen.call_args
    assert args[0] == oauth._REVOKE_URL
    posted_data = kwargs.get("data") or (args[1] if len(args) > 1 else b"")
    assert isinstance(posted_data, (bytes, bytearray))
    assert b"token=" in posted_data
    assert creds.token.encode() in posted_data
    # Keyring entry is gone.
    assert (
        fake_keyring_backend.get_password(
            oauth.KEYRING_SERVICE, oauth.KEYRING_USER
        )
        is None
    )


# ---------------------------------------------------------------------------
# Test 4 — keyring InitError/NoKeyringError triggers plaintext fallback.
# ---------------------------------------------------------------------------


def test_keyring_fallback(
    fake_keyring_backend,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    mock_credentials_factory,
) -> None:
    """InitError / NoKeyringError fall back to ``_FALLBACK_PATH`` + ``log.warning``."""
    import logging

    # Re-point the module-level fallback path at tmp_path so test writes
    # never touch the real user config dir.
    fallback_p = tmp_path / "youtube_oauth.json"
    monkeypatch.setattr(oauth, "_FALLBACK_PATH", fallback_p)
    # Force a fresh warning emission within this test.
    monkeypatch.setattr(oauth, "_warned_about_fallback", False)

    # Seed the plaintext fallback file with a JSON blob and mode 0600.
    fake_creds = mock_credentials_factory()
    fallback_p.write_text(fake_creds.to_json())
    os.chmod(fallback_p, 0o600)

    # Trip the keyring backend into InitError mode.
    fake_keyring_backend.failure_mode = "init"

    # Make from_authorized_user_info return the same MagicMock so production
    # code can finish the load path without parsing the dummy JSON.
    with patch.object(
        oauth.Credentials, "from_authorized_user_info", return_value=fake_creds
    ):
        with caplog.at_level(logging.WARNING, logger=oauth.log.name):
            result = oauth.load_or_refresh()

    assert result is fake_creds, "load_or_refresh must fall back to plaintext file"
    # Pitfall 3 — user-visible warning on fallback engagement.
    fallback_warnings = [
        r for r in caplog.records if "fallback" in r.message.lower()
    ]
    assert fallback_warnings, (
        f"expected log.warning about plaintext fallback, got {caplog.records!r}"
    )

    # NoKeyringError engages the same fallback path.
    fake_keyring_backend.failure_mode = "no_keyring"
    monkeypatch.setattr(oauth, "_warned_about_fallback", False)
    with patch.object(
        oauth.Credentials, "from_authorized_user_info", return_value=fake_creds
    ):
        result2 = oauth.load_or_refresh()
    assert result2 is fake_creds

    # T-08-02-04 mitigation — write path uses mode 0600 (defensive chmod).
    fake_keyring_backend.failure_mode = "init"
    fallback_p.unlink()  # Force a write of a fresh fallback file.
    monkeypatch.setattr(oauth, "_warned_about_fallback", False)
    with patch.object(
        oauth.Credentials, "from_authorized_user_info", return_value=fake_creds
    ):
        oauth._save_to_plaintext_fallback(fallback_p, fake_creds.to_json())
    assert fallback_p.exists()
    mode = oct(os.stat(fallback_p).st_mode & 0o777)
    assert mode == "0o600", f"plaintext fallback must be mode 0600, got {mode}"
