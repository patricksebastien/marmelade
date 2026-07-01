"""Audio backbone — Qt-free pure-Python audio I/O, proxy cache, and peak builder.

N-3 invariant: NO Qt imports in this package. The OS-cache-directory helper
that needs Qt's ``QStandardPaths`` lives at ``marmelade.paths`` (top-level),
keeping the audio layer unit-testable without a Qt event loop and reusable
from a future CLI front-end.
"""
