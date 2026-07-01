Project Prompt: "Marmelade" - AI-Powered Audio Analysis Interface
1. Project Overview & Rules
I need you to build a desktop application designed to process, analyze, and extract "good sections" from massive (up to 8 hours) audio recordings of music jam sessions.
Modularity: Write the code in separate files (e.g., main.py, audio_engine.py, ui_components.py, heatmap_extractors.py). Do not write everything in one file.
Extensibility: The "Heatmap" generation system must be an extensible base class so I can easily add new analysis algorithms later.
Memory Management: The software must never load the entire 8-hour WAV file into raw RAM at once. It must use block-based processing, lazy loading, or temporary downsampled proxy files for rendering.
2. Technology Stack
Language: Python 3.10+
Frontend / GUI: PyQt6 (or PySide6) combined with PyQtGraph. Note: PyQtGraph is mandatory for the UI as standard Matplotlib cannot handle zooming/panning millions of data points smoothly.
Audio Loading/Slicing: Spotify's pedalboard and soundfile for fast, memory-efficient read/write operations.
Audio AI & DSP Engine (GPU Accelerated where possible):
essentia (with tensorflow models)
transformers (Hugging Face YAMNet implementation)
torchaudio
librosa (only for non-intensive CPU tasks if PyTorch/Essentia can't do it)
3. Data Architecture: The "Heatmap" Concept
A "Heatmap" in this software is a 1-dimensional array of float values (0.0 to 1.0) synced to the audio timeline. For example, a 1-minute audio file analyzed at 1-second intervals produces an array of 60 values.
The UI will render these arrays as color-coded overlays underneath the main audio waveform.
The Backend will generate these arrays using different algorithms.
Required Heatmap Generators (To be implemented step-by-step)
Silence/Noise: Use pedalboard or torchaudio RMS energy detection.
Talking: Use YAMNet (Classifies "Speech" vs "Music").
Tuning / Fret Noise: Use YAMNet or Essentia to detect dissonant/non-musical transients.
Danceability: Use Essentia's pre-trained Tensorflow danceability model.
Sync BPM / Rhythm Stability: Use torchaudio or essentia beat tracking. Calculate the variance of the tempo; low variance = high heatmap score (they are playing in sync).
In-Tune (Harmonic Consonance): Extract pitch/chromagram. High consonance = high heatmap score.
Repetitive Patterns: Calculate the autocorrelation (tempogram) of the audio. Sustained high autocorrelation = high score.
Similar Tone/Instruments: Use Essentia's timbre extraction (MFCCs) to group similar sounding jams.
4. User Interface (GUI) Requirements
The UI should look like a modern DAW (Digital Audio Workstation) but focused purely on analysis.
Main Timeline: A large, horizontal PyQtGraph widget.
Must display a downsampled, static waveform of the audio file.
Must support smooth mouse-wheel zooming and drag-panning.
Heatmap Layers: Checkboxes on the left sidebar to toggle visibility of different heatmaps overlaying the timeline.
Region Selection Tool: Click and drag on the timeline to create a highlighted "Region".
5. Tools & Actions
When a user selects a region (or clicks directly on a high-value heatmap cluster), a context menu or toolbar must allow these actions:
Extract & Save: Export the selected region as an MP3 or WAV using pedalboard. Name it automatically: YYYY-MM-DD_HHMM_[MainHeatmapTrait].mp3.
Apply Fades: Add a quick 2-second fade-in/fade-out to the extracted audio so it doesn't clip.
Delete/Ignore: Mark a region as "Trash" (greys it out on the timeline).
Reorder/Playlist: Add the extracted region to a "Keepers" list/queue visible on the side of the UI.
6. Development Phases
Do not build this all at once. Let's build it in these exact phases. I will prompt you for the next phase when the current one works.
Phase 1: The Core UI & Audio Loader. Build the PyQt6 window, integrate PyQtGraph, and write the memory-efficient WAV loader that displays the visual waveform.
Phase 2: The Heatmap Base Class. Create the extensible Python class for heatmaps. Implement the simplest one first: The "Energy/Silence" heatmap. Map it visually to the PyQtGraph timeline.
Phase 3: The AI Integrations. Integrate YAMNet for "Talking" detection, and Essentia for "Danceability". Ensure these run on block-chunks so they don't crash the RAM.
Phase 4: Selection & Export Tools. Implement the click-and-drag region selection, and the pedalboard export functions with fade-in/out.
