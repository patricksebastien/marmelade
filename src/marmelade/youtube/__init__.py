"""YouTube upload package (Phase 8) — Qt-free per D-27.

Houses OAuth (:mod:`marmelade.youtube.oauth`), video assembly
(:mod:`marmelade.youtube.video_builder`), thumbnail fetch
(:mod:`marmelade.youtube.thumbnail_provider`), and the resumable
upload QRunnable (:mod:`marmelade.youtube.upload_runnable`).

N-3 invariant (Phase 8 D-27): NO QtWidgets and NO QtGui imports in
this package. The upload QRunnable will import the QtCore submodule
for QRunnable + Slot ONLY, mirroring
:mod:`marmelade.audio.mastering_worker`. UI surfaces (upload
dialog, share buttons, settings panel) live under
:mod:`marmelade.ui`.

Wave 0 (Plan 08-01): this package is created as an empty importable
namespace with stub modules added by downstream plans:

* Plan 08-02 — :mod:`oauth`, :mod:`client`
* Plan 08-04 — :mod:`thumbnail_provider`, :mod:`video_builder`,
  :mod:`upload_runnable`
"""
