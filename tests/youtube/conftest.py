"""Phase 8 shared test fixtures (Wave 0 — Plan 08-01 Task 3).

Four shared fixtures used by every ``test_*.py`` under ``tests/youtube/``:

* :func:`mock_credentials_factory` — callable that builds a stub
  :class:`google.oauth2.credentials.Credentials`-shaped MagicMock with
  ``.token``, ``.refresh_token``, ``.expired``, ``.refresh(Request)``,
  and ``.to_json()`` attributes. Parametrize expired-vs-valid via a
  flag. Used by :mod:`tests.youtube.test_oauth` and
  :mod:`tests.youtube.test_upload_runnable`.

* :func:`fake_keyring_backend` — monkeypatches ``keyring.get_password``
  / ``set_password`` / ``delete_password`` to use a dict-backed
  in-memory store, with a ``failure_mode`` hook that swaps the backend
  into ``keyring.errors.NoKeyringError`` / ``InitError`` raising mode
  (Pitfall 3 — research lines 654-675).

* :func:`mock_youtube_client` — MagicMock whose ``videos().insert(...)``
  and ``.next_chunk()`` are scripted by a yielded list of
  ``(status, response)`` tuples. ``thumbnails().set(...)`` is a sibling
  mock. Used by :mod:`tests.youtube.test_upload_runnable`.

* :func:`fake_picsum_response` — monkeypatches ``urllib.request.urlopen``
  to return a ``BytesIO`` containing a valid 1280x720 JPEG built via
  ``PIL.Image.new``. Includes a ``failure_mode`` parameter for
  "always raise URLError" (Pitfall 1 retry-then-fallback testing).

T-08-01-05 disposition: fixtures monkeypatch keyring + urllib +
googleapiclient inside the test process ONLY; no production code path
uses these fakes. Standard pytest hygiene.
"""

from __future__ import annotations

from io import BytesIO
from typing import Any, Callable, Iterator
from unittest.mock import MagicMock

import pytest


# ----------------------------------------------------------------------
# Fixture 1: mock_credentials_factory
# ----------------------------------------------------------------------


@pytest.fixture
def mock_credentials_factory() -> Callable[..., MagicMock]:
    """Return a callable that builds a Credentials-shaped MagicMock.

    Usage in tests::

        creds = mock_credentials_factory()                          # valid
        expired = mock_credentials_factory(expired=True)            # expired
        no_refresh = mock_credentials_factory(refresh_token=None)   # cannot refresh

    The returned mock mirrors the public surface of
    :class:`google.oauth2.credentials.Credentials` used by Phase 8
    OAuth code: ``token``, ``refresh_token``, ``expired``,
    ``refresh(Request)``, ``to_json()``.
    """

    def _make(
        *,
        token: str = "ya29.fake-access-token",
        refresh_token: str | None = "1//fake-refresh-token",
        expired: bool = False,
        token_json: str | None = None,
    ) -> MagicMock:
        m = MagicMock(name="Credentials")
        m.token = token
        m.refresh_token = refresh_token
        m.expired = expired
        # ``refresh(Request)`` flips ``expired`` to False (the real
        # Google Credentials object does this in-place).
        def _refresh(_request: Any) -> None:
            m.expired = False
            m.token = token + "-refreshed"

        m.refresh.side_effect = _refresh
        m.to_json.return_value = token_json or (
            '{"token": "%s", "refresh_token": "%s"}'
            % (token, refresh_token or "")
        )
        return m

    return _make


# ----------------------------------------------------------------------
# Fixture 2: fake_keyring_backend
# ----------------------------------------------------------------------


class _DictKeyringBackend:
    """Dict-backed in-memory keyring backend for tests.

    Supports a ``failure_mode`` switch to simulate the headless-Linux
    no-keyring case (raises ``keyring.errors.NoKeyringError`` per
    Pitfall 3 / RESEARCH lines 654-675).
    """

    def __init__(self) -> None:
        self._store: dict[tuple[str, str], str] = {}
        self.failure_mode: str | None = None  # None | "no_keyring" | "init"

    def get_password(self, service: str, user: str) -> str | None:
        self._maybe_raise()
        return self._store.get((service, user))

    def set_password(self, service: str, user: str, password: str) -> None:
        self._maybe_raise()
        self._store[(service, user)] = password

    def delete_password(self, service: str, user: str) -> None:
        self._maybe_raise()
        self._store.pop((service, user), None)

    def _maybe_raise(self) -> None:
        if self.failure_mode is None:
            return
        # Lazy import — keyring may not be installed when stubs only
        # collect (it IS installed after Plan 08-01 Task 1 ships, but
        # this fixture is tolerant either way).
        import keyring.errors as kerr

        if self.failure_mode == "no_keyring":
            raise kerr.NoKeyringError("Test backend in no_keyring failure mode")
        if self.failure_mode == "init":
            raise kerr.InitError("Test backend in init failure mode")


@pytest.fixture
def fake_keyring_backend(monkeypatch: pytest.MonkeyPatch) -> _DictKeyringBackend:
    """Replace ``keyring.{get,set,delete}_password`` with a dict-backed fake.

    Returns the backend instance so tests can read state and toggle
    ``failure_mode``::

        def test_keyring_fallback(fake_keyring_backend):
            fake_keyring_backend.failure_mode = "no_keyring"
            # ... assert oauth.load_or_refresh falls back to plaintext ...
    """
    backend = _DictKeyringBackend()
    import keyring  # safe — Plan 08-01 Task 1 installs this

    monkeypatch.setattr(keyring, "get_password", backend.get_password)
    monkeypatch.setattr(keyring, "set_password", backend.set_password)
    monkeypatch.setattr(keyring, "delete_password", backend.delete_password)
    return backend


# ----------------------------------------------------------------------
# Fixture 3: mock_youtube_client
# ----------------------------------------------------------------------


@pytest.fixture
def mock_youtube_client() -> MagicMock:
    """Return a MagicMock shaped like ``googleapiclient.discovery.build(...).``

    The mock exposes::

        client.videos().insert(part=..., body=..., media_body=...).next_chunk()
        client.thumbnails().set(videoId=..., media_body=...).execute()

    Tests script ``next_chunk`` via ``side_effect=[(status1, None),
    (status2, None), (None, response_dict)]`` etc. The default
    behaviour is a single completed-on-first-chunk upload returning
    ``{"id": "fake-video-id-abc12345xyz"}``.
    """
    client = MagicMock(name="youtube_client")

    # videos().insert(...).next_chunk() — scriptable progress.
    insert_request = MagicMock(name="videos.insert.request")
    insert_request.next_chunk.side_effect = [
        (None, {"id": "fake-video-id-abc12345xyz"}),
    ]
    client.videos.return_value.insert.return_value = insert_request

    # thumbnails().set(...).execute()
    thumb_request = MagicMock(name="thumbnails.set.request")
    thumb_request.execute.return_value = {"items": []}
    client.thumbnails.return_value.set.return_value = thumb_request

    return client


# ----------------------------------------------------------------------
# Fixture 4: fake_picsum_response
# ----------------------------------------------------------------------


@pytest.fixture
def fake_picsum_response(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[..., None]:
    """Monkeypatch ``urllib.request.urlopen`` to return a fake Picsum JPEG.

    Returns a configurator callable::

        fake_picsum_response()                          # success — 1280x720 JPEG
        fake_picsum_response(failure_mode="urlerror")   # always raises URLError
        fake_picsum_response(failure_mode="timeout")    # always raises TimeoutError

    Pitfall 1 retry-then-fallback testing — the thumbnail_provider
    (Plan 08-04) retries 3 times with 2s backoff, then falls back to
    the Pillow plain-color fallback. Tests can verify both paths.
    """

    def _configure(*, failure_mode: str | None = None) -> None:
        import urllib.request as _urlreq
        import urllib.error as _urlerr

        if failure_mode == "urlerror":
            def _raise(*_a: Any, **_kw: Any) -> Any:
                raise _urlerr.URLError("simulated network failure")
            monkeypatch.setattr(_urlreq, "urlopen", _raise)
            return
        if failure_mode == "timeout":
            def _raise(*_a: Any, **_kw: Any) -> Any:
                raise TimeoutError("simulated timeout")
            monkeypatch.setattr(_urlreq, "urlopen", _raise)
            return

        # Success path — build a 1280x720 JPEG in memory.
        from PIL import Image

        img = Image.new("RGB", (1280, 720), color=(50, 50, 50))
        buf = BytesIO()
        img.save(buf, "JPEG", quality=85)
        jpeg_bytes = buf.getvalue()

        class _FakeResponse:
            def __init__(self, data: bytes) -> None:
                self._data = data

            def __enter__(self) -> "_FakeResponse":
                return self

            def __exit__(self, *_a: Any) -> None:
                return None

            def read(self) -> bytes:
                return self._data

        def _urlopen(*_a: Any, **_kw: Any) -> _FakeResponse:
            return _FakeResponse(jpeg_bytes)

        monkeypatch.setattr(_urlreq, "urlopen", _urlopen)

    return _configure
