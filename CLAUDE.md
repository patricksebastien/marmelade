<!-- GSD:project-start source:PROJECT.md -->
## Project

**Marmelade**

A desktop application for musicians who record long, unstructured jam sessions (up to 8 hours per file) and need to find and extract the good moments without listening through every minute. The UI is a DAW focused purely on analysis: a zoomable waveform with multiple render/view modes (amplitude, dB, energy, and spectral: spectrogram, spectral-centroid tint, RGB frequency bands) to spot musically active sections by eye. Users select promising regions, audition them, master them with a per-keeper chain, and export clean clips with auto-naming and fades (or upload to YouTube). Timeline markers annotate ideas along the way.

> **History note:** an earlier milestone shipped AI/DSP "heatmap" overlays (silence/speech/music/danceability/rhythm/harmony via Essentia + TensorFlow/YAMNet) and a goodness engine. That backend was removed (quick-260620-vrn / dt4 for the UI, quick-260701-muv for the deps + code) — it is no longer part of the app. Discovery is now visual (render modes), not model-driven.

**Core Value:** A musician can open an 8-hour recording, visually locate the musically valuable sections in minutes (not hours), and extract them as clean clips — fast navigation + region selection + per-keeper mastering + lossless extraction must all work together end-to-end.

### Constraints

- **Tech stack**: Python 3.10+ — chosen for its audio ecosystem (pedalboard, soundfile, librosa, PySide6/PyQtGraph)
- **GUI**: PyQt6 (or PySide6) + PyQtGraph — Matplotlib cannot pan/zoom millions of waveform samples smoothly; PyQtGraph is required, not optional
- **Audio I/O**: pedalboard + soundfile — fast, memory-efficient read/write; pedalboard also handles the export + mastering pipeline
- **Audio DSP**: librosa + numpy for spectral analysis (spectrogram / centroid / band renders); soxr for resampling; pyloudnorm for LUFS; matchering for reference mastering. CPU-only — no ML/GPU runtime.
- **Memory**: must never load a full 8-hour WAV into raw RAM; block-based processing, lazy loading, or downsampled proxy files only
- **Modularity**: separate files (e.g., `app.py`, `audio/`, `ui/`, `audio/mastering/stages/`) — no monolithic file
- **Extensibility**: pluggable registries so new behavior is added without touching core code — the mastering chain is a set of modular stages, and waveform render modes use a `RenderMode` registry
- **Performance**: proxy build + spectral render for an 8hr file must stay responsive; UI must never block while background jobs (proxy, spectral, mastering, export) run
<!-- GSD:project-end -->

<!-- GSD:stack-start source:STACK.md -->
## Technology Stack

Technology stack not yet documented. Will populate after codebase mapping or first phase.
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->
## Conventions

Conventions not yet established. Will populate as patterns emerge during development.
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->
## Architecture

Architecture not yet mapped. Follow existing patterns found in the codebase.
<!-- GSD:architecture-end -->

<!-- GSD:skills-start source:skills/ -->
## Project Skills

No project skills found. Add skills to any of: `.claude/skills/`, `.agents/skills/`, `.cursor/skills/`, `.github/skills/`, or `.codex/skills/` with a `SKILL.md` index file.
<!-- GSD:skills-end -->

<!-- GSD:workflow-start source:GSD defaults -->
## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:
- `/gsd-quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd-debug` for investigation and bug fixing
- `/gsd-execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->



<!-- GSD:profile-start -->
## Developer Profile

> Profile not yet configured. Run `/gsd-profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
