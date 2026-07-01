"""YouTube Data API v3 discovery client builder (Phase 8 YT-01).

Single thin wrapper around :func:`googleapiclient.discovery.build` so all
upload code paths share one construction site. ``cache_discovery=False``
is intentional — avoids file-locking issues on PyInstaller bundles
(RESEARCH §B2 future-proofing) and removes a sometimes-broken file write
to the user's home directory.

Public surface (consumed by Plan 08-04 upload runnable + Plan 08-02
:func:`marmelade.youtube.oauth.channel_info`):

* :func:`build_youtube_client(creds)` — returns a discovery ``Resource``.

N-3 (Qt-free package) — this module imports zero ``PySide6.*`` symbols.
"""

from __future__ import annotations

from typing import Any

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build


def build_youtube_client(creds: Credentials) -> Any:
    """Return a YouTube Data API v3 discovery client bound to ``creds``.

    Args:
        creds: Authenticated :class:`google.oauth2.credentials.Credentials`,
            typically obtained from
            :func:`marmelade.youtube.oauth.load_or_refresh`.

    Returns:
        A :class:`googleapiclient.discovery.Resource` shaped like
        ``client.videos().insert(...).next_chunk()`` and
        ``client.channels().list(part=..., mine=True).execute()``. The
        precise return type is :any:`Any` because googleapiclient builds
        the resource shape dynamically from the v3 discovery document.

    Notes:
        ``cache_discovery=False`` skips the file-system discovery-document
        cache. On PyInstaller bundles the default cache directory can be
        read-only (or shared between users), surfacing as a confusing
        "permission denied" deep inside googleapiclient. Disabling the
        cache costs one extra HTTPS GET per client construction (~50 ms)
        — acceptable for an interactive upload flow.
    """
    return build("youtube", "v3", credentials=creds, cache_discovery=False)
