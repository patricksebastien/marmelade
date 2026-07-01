#!/usr/bin/env bash
#
# scripts/fetch_models.sh — one-shot dev helper to fetch the two essentia
# Danceability `.pb` weights from essentia.upf.edu into
# `$(default_cache_root)/models/essentia/`.
#
# Phase 4 only — Phase 6 supersedes this with the in-app auto-downloader
# (AI-SPEC §1 Failure Mode #5 + RESEARCH §A2). The two `.pb` files are
# CC-BY-NC-SA-4.0 licensed and cannot be auto-downloaded inside Phase 4
# (see plan-04-01 `<user_setup>`).
#
# Idempotent: `curl -z` skips files already present on disk. Hardcoded
# HTTPS URLs — no shell expansion on user input.

set -euo pipefail

# Resolve cache root by calling the production helper. `uv run` ensures we
# pick up the project venv (PySide6 is needed for QStandardPaths). After
# paths.py switched to GenericCacheLocation, no QCoreApplication org/app
# setup is needed in the sub-process — the helper is deterministic.
CACHE_ROOT="$(uv run python -c 'from jamextractor.paths import default_cache_root; print(default_cache_root())')"
DEST_DIR="$CACHE_ROOT/models/essentia"
mkdir -p "$DEST_DIR"

declare -a URLS=(
    "https://essentia.upf.edu/models/feature-extractors/discogs-effnet/discogs-effnet-bs64-1.pb"
    "https://essentia.upf.edu/models/classification-heads/danceability/danceability-discogs-effnet-1.pb"
)

for url in "${URLS[@]}"; do
    filename="$(basename "$url")"
    dst="$DEST_DIR/$filename"
    # -f: fail on HTTP errors; -L: follow redirects; -C -: resume partial
    # downloads; -z: skip if local file is newer-or-equal to remote mtime.
    curl -fL --continue-at - -z "$dst" -o "$dst" "$url"
done

echo "fetch_models.sh — summary:"
for url in "${URLS[@]}"; do
    filename="$(basename "$url")"
    dst="$DEST_DIR/$filename"
    if [[ -f "$dst" ]]; then
        size="$(stat -c%s "$dst" 2>/dev/null || stat -f%z "$dst")"
        echo "  ok  $dst ($size bytes)"
    else
        echo "  MISSING $dst"
    fi
done

# --- YAMNet (Plan 04.1-01) ------------------------------------------------
# YAMNet ships as a TF Hub SavedModel; we pre-warm the on-disk cache by
# running tensorflow_hub.load() through the venv's TF stack, which is the
# same code path the app uses at runtime. Idempotent — TF Hub's downloader
# hits the cache on subsequent runs.
YAMNET_DIR="$CACHE_ROOT/models/yamnet"
mkdir -p "$YAMNET_DIR"
uv run python - <<PY
import os
os.environ["TFHUB_CACHE_DIR"] = "$YAMNET_DIR"
import tensorflow_hub as hub
m = hub.load("https://tfhub.dev/google/yamnet/1")
# Resolve the on-disk class_map.csv path — proves the download landed.
print("yamnet class_map_path():", m.class_map_path().numpy().decode("utf-8"))
PY

# Print the SHA-256 of the cached SavedModel's saved_model.pb so the
# Phase 6 auto-downloader can pin the digest. This is informational —
# Plan 04.1-01 ships sha256='unverified' per R-04.
SAVED_PB="$(find "$YAMNET_DIR" -name saved_model.pb -print -quit 2>/dev/null || true)"
if [[ -n "$SAVED_PB" && -f "$SAVED_PB" ]]; then
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$SAVED_PB"
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$SAVED_PB"
    fi
fi
