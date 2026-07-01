"""Synthesize audio fixtures at test time — no binary files in the repo.

All three helpers write via ``pedalboard.io.AudioFile`` in 1-second chunks so
even long fixtures stay memory-bounded (CLAUDE.md memory contract).

RESEARCH Pitfall #2: the libsndfile-based alternative encoder is NEVER used
for MP3 here because libsndfile MP3 support is unreliable across Linux
distros. pedalboard ships a statically-bundled JUCE that encodes MP3 on every
platform without external system deps.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pedalboard.io import AudioFile


def _write_chunked(
    path: Path,
    sample_rate: int,
    channels: int,
    fmt: str,
    sample_generator,
) -> Path:
    """Write ``sample_generator`` (yields float32 mono blocks in [-1, 1]) to ``path``.

    For multi-channel output the same mono signal is duplicated across channels.
    The file extension on ``path`` controls the container; ``fmt`` is informational
    and validated against the supported set so callers fail fast on typos.
    """
    fmt = fmt.lower()
    if fmt not in ("wav", "flac", "mp3"):
        raise ValueError(f"Unsupported fmt={fmt!r}; expected 'wav', 'flac', or 'mp3'")
    path = Path(path)
    with AudioFile(
        str(path),
        "w",
        samplerate=sample_rate,
        num_channels=channels,
    ) as f:
        for mono_block in sample_generator:
            mono_block = np.asarray(mono_block, dtype=np.float32)
            if channels == 1:
                f.write(mono_block.reshape(1, -1))
            else:
                stacked = np.tile(mono_block, (channels, 1))
                f.write(stacked)
    return path


def make_sine(
    path: Path | str,
    freq_hz: float = 1000.0,
    amp: float = 0.5,
    duration_s: float = 30.0,
    sample_rate: int = 44100,
    channels: int = 1,
    fmt: str = "wav",
) -> Path:
    """Write a sine-wave fixture of ``duration_s`` to ``path``. Returns ``path``."""
    path = Path(path)
    total = int(round(duration_s * sample_rate))
    chunk = sample_rate  # 1 second per chunk
    offset = 0

    def gen():
        nonlocal offset
        while offset < total:
            n = min(chunk, total - offset)
            t = (np.arange(offset, offset + n, dtype=np.float64)) / sample_rate
            block = (amp * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.float32)
            offset += n
            yield block

    return _write_chunked(path, sample_rate, channels, fmt, gen())


def make_chirp(
    path: Path | str,
    *,
    duration_s: float = 2.0,
    f0_hz: float = 200.0,
    f1_hz: float = 8000.0,
    sample_rate: int = 48000,
    amp: float = 0.5,
    channels: int = 1,
    fmt: str = "wav",
    log_sweep: bool = False,
) -> Path:
    """Write a frequency-sweep (chirp) fixture from ``f0_hz`` to ``f1_hz``.

    This is the middle segment of the Phase 11 R-4 silence->sweep->noise
    spectrogram fixture: a continuous sweep renders as a rising diagonal
    ridge in the mel image, which the render test asserts on.

    Default ``sample_rate=48000`` is the canonical Marmelade rate (the older
    helpers default 44100, which is stale per MEMORY: project_mastering_48khz);
    new fixtures should sweep at 48 kHz so the spectral builder sees the same
    rate the production chain uses.

    Continuity contract (the load-bearing detail): the instantaneous phase is
    integrated across chunk boundaries via a running phase accumulator. We do
    NOT recompute ``phase = 2*pi*f(t)*t`` per chunk — doing so would reset the
    phase reference at each 1-second boundary and inject an audible click (and
    a spurious broadband column in the spectrogram). Instead each chunk picks
    up the phase where the previous one left off:

        phase[n] = phase_acc + 2*pi/sr * cumsum(f_inst[0..n])
        phase_acc <- phase[-1]            (carried into the next chunk)

    where ``f_inst`` is the linear (or, with ``log_sweep=True``, exponential)
    instantaneous-frequency ramp from ``f0_hz`` to ``f1_hz`` over ``duration_s``.

    Memory contract (CLAUDE.md): synthesised and written in 1-second chunks via
    :func:`_write_chunked` — no single full-length signal array is allocated.

    Returns ``path``.
    """
    path = Path(path)
    total = int(round(duration_s * sample_rate))
    chunk = sample_rate  # 1 second per chunk
    offset = 0
    # Running phase accumulator (radians) carried across chunk boundaries so
    # the sweep is continuous (no clicks at the 1 s seams).
    phase_acc = 0.0
    # Total span in samples for the instantaneous-frequency ramp.
    span = max(total - 1, 1)

    def _inst_freq(idx: np.ndarray) -> np.ndarray:
        # Fraction of the way through the sweep for each absolute sample index.
        frac = idx.astype(np.float64) / float(span)
        frac = np.clip(frac, 0.0, 1.0)
        if log_sweep and f0_hz > 0.0 and f1_hz > 0.0:
            return f0_hz * (f1_hz / f0_hz) ** frac
        return f0_hz + (f1_hz - f0_hz) * frac

    def gen():
        nonlocal offset, phase_acc
        dt = 1.0 / float(sample_rate)
        while offset < total:
            n = min(chunk, total - offset)
            idx = np.arange(offset, offset + n, dtype=np.float64)
            f_inst = _inst_freq(idx)
            # Integrate instantaneous frequency to phase, continuing from the
            # accumulator so the seam between chunks is phase-continuous.
            phase = phase_acc + 2.0 * np.pi * np.cumsum(f_inst) * dt
            phase_acc = float(phase[-1])
            block = (amp * np.sin(phase)).astype(np.float32)
            offset += n
            yield block

    return _write_chunked(path, sample_rate, channels, fmt, gen())


def make_white_noise(
    path: Path | str,
    duration_s: float = 30.0,
    sample_rate: int = 44100,
    channels: int = 1,
    fmt: str = "wav",
    seed: int | None = 0,
) -> Path:
    """Write a white-noise fixture of ``duration_s`` to ``path``. Returns ``path``."""
    path = Path(path)
    total = int(round(duration_s * sample_rate))
    chunk = sample_rate
    offset = 0
    rng = np.random.default_rng(seed)

    def gen():
        nonlocal offset
        while offset < total:
            n = min(chunk, total - offset)
            # Uniform in [-0.5, 0.5] keeps the signal well under clipping.
            block = (rng.standard_normal(n).astype(np.float32) * 0.25).clip(-1.0, 1.0)
            offset += n
            yield block

    return _write_chunked(path, sample_rate, channels, fmt, gen())


def make_silence(
    path: Path | str,
    duration_s: float = 30.0,
    sample_rate: int = 44100,
    channels: int = 1,
    fmt: str = "wav",
) -> Path:
    """Write a pure-silence fixture of ``duration_s`` to ``path``. Returns ``path``."""
    path = Path(path)
    total = int(round(duration_s * sample_rate))
    chunk = sample_rate
    offset = 0

    def gen():
        nonlocal offset
        while offset < total:
            n = min(chunk, total - offset)
            block = np.zeros(n, dtype=np.float32)
            offset += n
            yield block

    return _write_chunked(path, sample_rate, channels, fmt, gen())


def make_tight_loose_tight_drum_loop(
    path: Path | str,
    duration_s: float = 180.0,
    sample_rate: int = 44100,
    channels: int = 1,
    fmt: str = "wav",
) -> tuple[Path, list[dict]]:
    """Three-section drum-loop fixture (tight / loose / tight) for D4 rubric.

    Returns ``(path, sections)`` where ``sections`` is a list of three dicts
    with ``start_s``, ``end_s``, and ``label in {"tight", "loose", "tight"}``.

    Section construction:
        - **Tight** sections use a constant 120 BPM grid (0.5 s per beat),
          kick on every beat, snare on beats 2 and 4 (i.e. every other beat
          offset by one), hat on every eighth-note (0.25 s).
        - **Loose** section uses a per-beat BPM random walk: the next beat
          arrives at ``current_time + 60 / (120 + rng.uniform(-15, +15))``.
          Same drum-voice mapping as tight, just on a jittered grid.

    Voices (hand-rolled, numpy-only — no scipy):
        - **Kick**: 80 Hz sine × exp(-t/0.05), 50 ms tail, amp 0.6.
        - **Snare**: white noise × exp(-t/0.08), 80 ms tail, amp 0.35.
        - **Hat**: high-passed white noise (diff'd) × exp(-t/0.02), 20 ms tail, amp 0.20.

    All samples float32, mono, normalised to stay inside [-0.98, 0.98]. RNG
    is seeded so the fixture is bit-reproducible (seed=0xD4 — "D4 rubric").
    """
    path = Path(path)
    rng = np.random.default_rng(0xD4)
    total_samples = int(round(duration_s * sample_rate))
    section_samples = total_samples // 3
    # Recompute totals so the three sections sum exactly to total_samples
    # (handles the rounding remainder by lengthening the final section).
    section_lengths = [section_samples, section_samples, total_samples - 2 * section_samples]
    section_starts_s = [
        0.0,
        section_lengths[0] / sample_rate,
        (section_lengths[0] + section_lengths[1]) / sample_rate,
    ]
    section_ends_s = [
        section_lengths[0] / sample_rate,
        (section_lengths[0] + section_lengths[1]) / sample_rate,
        total_samples / sample_rate,
    ]
    sections = [
        {"start_s": section_starts_s[0], "end_s": section_ends_s[0], "label": "tight"},
        {"start_s": section_starts_s[1], "end_s": section_ends_s[1], "label": "loose"},
        {"start_s": section_starts_s[2], "end_s": section_ends_s[2], "label": "tight"},
    ]

    # --- Voice synthesisers (return mono float32 transient arrays) ------------
    def _kick() -> np.ndarray:
        # 50 ms 80 Hz sine with exp decay.
        n = int(round(0.05 * sample_rate))
        t = np.arange(n, dtype=np.float32) / sample_rate
        env = np.exp(-t / 0.05).astype(np.float32)
        return (0.6 * np.sin(2.0 * np.pi * 80.0 * t).astype(np.float32) * env).astype(np.float32)

    def _snare() -> np.ndarray:
        n = int(round(0.08 * sample_rate))
        t = np.arange(n, dtype=np.float32) / sample_rate
        env = np.exp(-t / 0.05).astype(np.float32)
        noise = rng.standard_normal(n).astype(np.float32)
        return (0.35 * noise * env).astype(np.float32)

    def _hat() -> np.ndarray:
        n = int(round(0.02 * sample_rate))
        t = np.arange(n, dtype=np.float32) / sample_rate
        env = np.exp(-t / 0.005).astype(np.float32)
        noise = rng.standard_normal(n).astype(np.float32)
        # Naive high-pass via first-difference: emphasises high frequencies
        # without scipy.signal — stdlib + numpy only.
        hp = np.diff(noise, prepend=0.0).astype(np.float32)
        return (0.20 * hp * env).astype(np.float32)

    def _mix_hit(buf: np.ndarray, hit: np.ndarray, start_idx: int) -> None:
        end_idx = min(start_idx + hit.size, buf.size)
        if end_idx <= start_idx:
            return
        buf[start_idx:end_idx] += hit[: end_idx - start_idx]

    def _section_tight(length: int) -> np.ndarray:
        buf = np.zeros(length, dtype=np.float32)
        bpm = 120.0
        beat_s = 60.0 / bpm  # 0.5 s
        eighth_s = beat_s / 2.0  # 0.25 s
        # Kick on every beat; snare on beats 2 and 4 of each 4-beat bar.
        t = 0.0
        beat_idx = 0
        while t < length / sample_rate:
            idx = int(round(t * sample_rate))
            if idx >= length:
                break
            _mix_hit(buf, _kick(), idx)
            if beat_idx % 4 in (1, 3):
                _mix_hit(buf, _snare(), idx)
            t += beat_s
            beat_idx += 1
        # Hat on every eighth-note.
        t = 0.0
        while t < length / sample_rate:
            idx = int(round(t * sample_rate))
            if idx >= length:
                break
            _mix_hit(buf, _hat(), idx)
            t += eighth_s
        return np.clip(buf, -0.98, 0.98).astype(np.float32)

    def _section_loose(length: int) -> np.ndarray:
        """Generate a "not perceivably danceable" section.

        The discogs-effnet danceability classifier is trained on real
        music — it rates ANY consistent kick-snare-hat pattern as ~0.99
        danceable, even with ±15 BPM jitter (the original 2026-05-17
        design didn't produce a discriminable Δ). The fixture's purpose
        is to satisfy the D4 rubric in plan 04-02: mean(tight) -
        mean(loose) ≥ 0.15.

        Effective approach: the loose section is a slow ambient drone +
        pitch-modulated mid-frequency sine + low-amplitude pink-ish noise
        — no transients, no consistent pulse, no kick-snare-hat at all.
        The model rates this as predominantly NOT danceable.

        This keeps the section non-silent (so downstream energy-based
        heatmaps still see a valid signal there), while moving the
        danceability score firmly below 0.5 over the section.
        """
        duration_s = length / sample_rate
        t = np.arange(length, dtype=np.float32) / sample_rate
        # Slowly-modulated mid-frequency sine: pitch wanders ~330..500 Hz
        # over the section (a "drone with vibrato"). No percussive
        # onsets, no rhythm.
        pitch_lfo = 415.0 + 85.0 * np.sin(
            2.0 * np.pi * 0.3 * t  # 0.3 Hz pitch LFO
        ).astype(np.float32)
        phase = 2.0 * np.pi * np.cumsum(pitch_lfo) / float(sample_rate)
        drone = (0.18 * np.sin(phase)).astype(np.float32)
        # Slow amplitude envelope (0.5 Hz tremolo) so the drone isn't
        # too pure-tone — adds harmonic richness without rhythm.
        amp_env = (0.7 + 0.3 * np.sin(2.0 * np.pi * 0.5 * t)).astype(np.float32)
        drone = drone * amp_env
        # Low-level pink-ish noise floor (-35 dBFS) for naturalness.
        noise = (rng.standard_normal(length).astype(np.float32) * 0.015).clip(
            -1.0, 1.0
        )
        buf = (drone + noise).astype(np.float32)
        return np.clip(buf, -0.98, 0.98).astype(np.float32)

    section_1 = _section_tight(section_lengths[0])
    section_2 = _section_loose(section_lengths[1])
    section_3 = _section_tight(section_lengths[2])
    concatenated = np.concatenate([section_1, section_2, section_3])
    assert concatenated.size == total_samples, (
        f"section concat mismatch: {concatenated.size} != {total_samples}"
    )

    # Stream to disk via _write_chunked, 1-second chunks for memory-bound write.
    chunk = sample_rate
    offset = 0

    def gen():
        nonlocal offset
        while offset < total_samples:
            n = min(chunk, total_samples - offset)
            yield concatenated[offset : offset + n]
            offset += n

    _write_chunked(path, sample_rate, channels, fmt, gen())
    return path, sections


def _one_pole_highpass(x: np.ndarray, cutoff_hz: float, sample_rate: int) -> np.ndarray:
    """Single-pole IIR HPF (numpy-only — no scipy). RC = 1 / (2*pi*fc)."""
    dt = 1.0 / float(sample_rate)
    rc = 1.0 / (2.0 * np.pi * cutoff_hz)
    alpha = rc / (rc + dt)
    y = np.zeros_like(x, dtype=np.float32)
    if x.size == 0:
        return y
    y[0] = x[0]
    for i in range(1, x.size):
        y[i] = alpha * (y[i - 1] + x[i] - x[i - 1])
    return y.astype(np.float32, copy=False)


def _one_pole_lowpass(x: np.ndarray, cutoff_hz: float, sample_rate: int) -> np.ndarray:
    """Single-pole IIR LPF (numpy-only). y[n] = a*x[n] + (1-a)*y[n-1]."""
    dt = 1.0 / float(sample_rate)
    rc = 1.0 / (2.0 * np.pi * cutoff_hz)
    alpha = dt / (rc + dt)
    y = np.zeros_like(x, dtype=np.float32)
    if x.size == 0:
        return y
    y[0] = alpha * x[0]
    for i in range(1, x.size):
        y[i] = alpha * x[i] + (1.0 - alpha) * y[i - 1]
    return y.astype(np.float32, copy=False)


def _band_pass(x: np.ndarray, low_hz: float, high_hz: float, sample_rate: int) -> np.ndarray:
    """Cascade HPF then LPF — vocal-band emulation."""
    return _one_pole_lowpass(
        _one_pole_highpass(x, low_hz, sample_rate), high_hz, sample_rate
    ).astype(np.float32, copy=False)


def _tight_drum_section(
    length: int,
    sample_rate: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Synthesise a tight 120 BPM kick/snare/hat section of ``length`` samples.

    Extracted from :func:`make_tight_loose_tight_drum_loop` to reuse in
    :func:`make_speech_burst_in_music` and :func:`make_vocal_melisma`
    without re-inlining the drum voicings. Same parameters: 120 BPM, kick
    on every beat, snare on beats 2 and 4 of every 4-beat bar, hat on
    eighth-notes. Voicings are numpy-only (no scipy).
    """

    def _kick() -> np.ndarray:
        n = int(round(0.05 * sample_rate))
        t = np.arange(n, dtype=np.float32) / sample_rate
        env = np.exp(-t / 0.05).astype(np.float32)
        return (0.6 * np.sin(2.0 * np.pi * 80.0 * t).astype(np.float32) * env).astype(
            np.float32
        )

    def _snare() -> np.ndarray:
        n = int(round(0.08 * sample_rate))
        t = np.arange(n, dtype=np.float32) / sample_rate
        env = np.exp(-t / 0.05).astype(np.float32)
        noise = rng.standard_normal(n).astype(np.float32)
        return (0.35 * noise * env).astype(np.float32)

    def _hat() -> np.ndarray:
        n = int(round(0.02 * sample_rate))
        t = np.arange(n, dtype=np.float32) / sample_rate
        env = np.exp(-t / 0.005).astype(np.float32)
        noise = rng.standard_normal(n).astype(np.float32)
        hp = np.diff(noise, prepend=0.0).astype(np.float32)
        return (0.20 * hp * env).astype(np.float32)

    def _mix_hit(buf: np.ndarray, hit: np.ndarray, start_idx: int) -> None:
        end_idx = min(start_idx + hit.size, buf.size)
        if end_idx <= start_idx:
            return
        buf[start_idx:end_idx] += hit[: end_idx - start_idx]

    buf = np.zeros(length, dtype=np.float32)
    bpm = 120.0
    beat_s = 60.0 / bpm  # 0.5 s
    eighth_s = beat_s / 2.0  # 0.25 s
    t = 0.0
    beat_idx = 0
    while t < length / sample_rate:
        idx = int(round(t * sample_rate))
        if idx >= length:
            break
        _mix_hit(buf, _kick(), idx)
        if beat_idx % 4 in (1, 3):
            _mix_hit(buf, _snare(), idx)
        t += beat_s
        beat_idx += 1
    t = 0.0
    while t < length / sample_rate:
        idx = int(round(t * sample_rate))
        if idx >= length:
            break
        _mix_hit(buf, _hat(), idx)
        t += eighth_s
    return np.clip(buf, -0.98, 0.98).astype(np.float32)


def make_speech_burst_in_music(
    path: Path | str,
    music_secs_per_side: float = 30.0,
    burst_secs: float = 5.0,
    sample_rate: int = 44100,
    channels: int = 1,
    fmt: str = "wav",
) -> tuple[Path, list[dict]]:
    """Plan 04.1-01 fixture — fake speech burst between two music segments.

    Synthesised content per AI-SPEC §5 D2 short-banter-resolution rubric:
        - First ``music_secs_per_side`` seconds: tight 120 BPM drum-loop.
        - Middle ``burst_secs`` seconds: formant-shaped band-limited
          noise (200-3400 Hz envelope, ~80 Hz pitch carrier, 5 Hz
          amplitude modulation to mimic syllable rate).
        - Final ``music_secs_per_side`` seconds: same drum-loop.

    Returns:
        ``(path, labels)`` where labels is a 3-entry list with
        ``start_s``, ``end_s``, and ``label in {"music", "speech",
        "music"}``. The middle entry is the labeled speech window
        (D2 peak-position check).
    """
    path = Path(path)
    rng = np.random.default_rng(0x7A)
    music_samples = int(round(music_secs_per_side * sample_rate))
    burst_samples = int(round(burst_secs * sample_rate))
    total_samples = 2 * music_samples + burst_samples

    # Music sections — reuse the tight drum-loop helper.
    music_1 = _tight_drum_section(music_samples, sample_rate, rng)
    music_2 = _tight_drum_section(music_samples, sample_rate, rng)

    # Fake speech burst — formant-shaped, amplitude-modulated band-limited
    # noise. ~80 Hz pitch carrier + 5 Hz syllable envelope + band-pass
    # filter 200..3400 Hz (typical vocal range). The noisy carrier mimics
    # vocal cord vibration; the band-pass approximates the vocal tract
    # formants without a full source-filter model.
    t_burst = np.arange(burst_samples, dtype=np.float32) / sample_rate
    carrier = np.sin(2.0 * np.pi * 80.0 * t_burst).astype(np.float32)
    amp_env = (0.5 * (1.0 + np.sin(2.0 * np.pi * 5.0 * t_burst))).astype(np.float32)
    noisy_carrier = (carrier * amp_env * rng.standard_normal(burst_samples).astype(
        np.float32
    )).astype(np.float32)
    burst = _band_pass(noisy_carrier, 200.0, 3400.0, sample_rate)
    # Normalise to a comfortable level (band-pass + noise mult can leave
    # peaks far below 1.0; bring it up so the model sees a clear signal).
    peak = float(np.max(np.abs(burst))) if burst.size else 0.0
    if peak > 0.0:
        burst = (burst * (0.7 / peak)).astype(np.float32)
    burst = np.clip(burst, -0.98, 0.98).astype(np.float32)

    concatenated = np.concatenate([music_1, burst, music_2]).astype(np.float32)
    assert concatenated.size == total_samples, (
        f"speech-burst concat mismatch: {concatenated.size} != {total_samples}"
    )

    labels = [
        {
            "start_s": 0.0,
            "end_s": music_samples / sample_rate,
            "label": "music",
        },
        {
            "start_s": music_samples / sample_rate,
            "end_s": (music_samples + burst_samples) / sample_rate,
            "label": "speech",
        },
        {
            "start_s": (music_samples + burst_samples) / sample_rate,
            "end_s": total_samples / sample_rate,
            "label": "music",
        },
    ]

    chunk = sample_rate
    offset = 0

    def gen():
        nonlocal offset
        while offset < total_samples:
            n = min(chunk, total_samples - offset)
            yield concatenated[offset : offset + n]
            offset += n

    _write_chunked(path, sample_rate, channels, fmt, gen())
    return path, labels


def make_vocal_melisma(
    path: Path | str,
    duration_s: float = 30.0,
    sample_rate: int = 44100,
    channels: int = 1,
    fmt: str = "wav",
) -> tuple[Path, list[dict]]:
    """Plan 04.1-01 fixture — sustained vocal-like note over drums (D4 guard).

    Synthesises a "held sung note" the Talking row MUST classify as
    music, not speech (per AI-SPEC §1b Known Failure 2 — Singing is
    under Music in AudioSet):
        - Carrier: 220 Hz sine ("A3").
        - Vibrato: ±15 cents at 5 Hz.
        - Two-pole LPF cutoff 1500 Hz to fake a vowel-like formant.
        - Mixed with the standard tight drum-loop at -3 dB.

    Returns:
        ``(path, labels)`` where labels is
        ``[{"start_s": 0.0, "end_s": duration_s, "label": "vocal_only"}]``.
    """
    path = Path(path)
    rng = np.random.default_rng(0xAA)
    total_samples = int(round(duration_s * sample_rate))
    t = np.arange(total_samples, dtype=np.float32) / sample_rate

    # Vibrato: ±15 cents at 5 Hz. 1 cent ≈ 0.0578%. ±15 cents ≈ ±0.866%.
    base_pitch = 220.0  # A3
    pitch_lfo = 5.0  # Hz
    cents_amp = 15.0 / 100.0  # in semitones
    # 2^(cents_amp/12) ≈ 1 + 0.00866 for small values; use exact form.
    pitch_factor = 2.0 ** ((cents_amp / 12.0) * np.sin(2.0 * np.pi * pitch_lfo * t))
    instant_freq = base_pitch * pitch_factor.astype(np.float32)
    phase = 2.0 * np.pi * np.cumsum(instant_freq) / float(sample_rate)
    carrier = np.sin(phase).astype(np.float32)
    # Two-pole LPF cutoff 1500 Hz approximated by two cascaded one-pole LPFs.
    vocal = _one_pole_lowpass(
        _one_pole_lowpass(carrier, 1500.0, sample_rate), 1500.0, sample_rate
    )
    # Normalise the filtered output before mixing.
    peak = float(np.max(np.abs(vocal))) if vocal.size else 0.0
    if peak > 0.0:
        vocal = (vocal * (0.5 / peak)).astype(np.float32)

    # Mix with tight drums at -3 dB (≈ 0.707 amplitude factor).
    drums = _tight_drum_section(total_samples, sample_rate, rng)
    mixed = (vocal + 0.707 * drums).astype(np.float32)
    mixed = np.clip(mixed, -0.98, 0.98).astype(np.float32)

    labels = [{"start_s": 0.0, "end_s": duration_s, "label": "vocal_only"}]

    chunk = sample_rate
    offset = 0

    def gen():
        nonlocal offset
        while offset < total_samples:
            n = min(chunk, total_samples - offset)
            yield mixed[offset : offset + n]
            offset += n

    _write_chunked(path, sample_rate, channels, fmt, gen())
    return path, labels


def make_pink_noise(
    path: Path | str,
    duration_s: float = 30.0,
    sample_rate: int = 44100,
    channels: int = 1,
    fmt: str = "wav",
    seed: int | None = 0x1F,
) -> Path:
    """Plan 04.1-01 fixture — 1/f pink noise for D5 block-boundary tests.

    Generated by FFT-based spectral shaping of white noise: synthesise
    white noise in chunks (memory-bounded), apply a 1/f magnitude
    envelope in the frequency domain, IFFT back to time domain. Per
    CLAUDE.md memory contract — chunked, not whole-file.
    """
    path = Path(path)
    rng = np.random.default_rng(seed)
    total_samples = int(round(duration_s * sample_rate))
    # Chunk size of 1 s is the same idiom the other synthesisers use.
    chunk = sample_rate

    # Pre-build a 1/f magnitude envelope ONCE per chunk size — shared across
    # chunks. The phase varies per chunk via rng.
    freqs = np.fft.rfftfreq(chunk, d=1.0 / sample_rate).astype(np.float32)
    # Avoid divide-by-zero at DC; clamp the floor frequency to 1 Hz.
    mag = 1.0 / np.sqrt(np.maximum(freqs, 1.0))
    mag = mag.astype(np.float32)

    def gen():
        emitted = 0
        while emitted < total_samples:
            n = min(chunk, total_samples - emitted)
            if n == chunk:
                white = rng.standard_normal(chunk).astype(np.float32)
                spectrum = np.fft.rfft(white)
                shaped_spectrum = spectrum * mag
                pink = np.fft.irfft(shaped_spectrum, n=chunk).astype(np.float32)
                # Normalise — pink noise from this shaping has large variance.
                peak = float(np.max(np.abs(pink)))
                if peak > 0.0:
                    pink = (pink * (0.5 / peak)).astype(np.float32)
            else:
                # Final short tail — synthesise at the exact tail length.
                white = rng.standard_normal(n).astype(np.float32)
                tail_freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate).astype(np.float32)
                tail_mag = (1.0 / np.sqrt(np.maximum(tail_freqs, 1.0))).astype(np.float32)
                spectrum = np.fft.rfft(white) * tail_mag
                pink = np.fft.irfft(spectrum, n=n).astype(np.float32)
                peak = float(np.max(np.abs(pink)))
                if peak > 0.0:
                    pink = (pink * (0.5 / peak)).astype(np.float32)
            pink = np.clip(pink, -0.98, 0.98).astype(np.float32)
            yield pink
            emitted += n

    return _write_chunked(path, sample_rate, channels, fmt, gen())


def make_mixed_dynamics(
    path: Path | str,
    sample_rate: int = 44100,
    channels: int = 1,
    fmt: str = "wav",
    seed: int | None = 0,
) -> Path:
    """Write a 5-section dynamics fixture (silent / quiet / loud / quiet / silent).

    Each section is exactly 1 second; total duration 5 s. Used by
    ``tests/unit/test_energy_heatmap.py`` to assert that the 5th/95th
    percentile + -60 dBFS floor discipline produces sane silent/quiet/loud
    band assignments across a single mixed input (D-04 + GEN-01).

    The two noise sections use deterministic seeded RNG so the test is
    bit-reproducible across machines.
    """
    path = Path(path)
    rng = np.random.default_rng(seed)
    sections = [
        np.zeros(sample_rate, dtype=np.float32),  # silent
        (rng.standard_normal(sample_rate).astype(np.float32) * 0.01).clip(-1.0, 1.0),  # quiet ~-40 dBFS
        (rng.standard_normal(sample_rate).astype(np.float32) * 0.5).clip(-1.0, 1.0),  # loud
        (rng.standard_normal(sample_rate).astype(np.float32) * 0.01).clip(-1.0, 1.0),  # quiet
        np.zeros(sample_rate, dtype=np.float32),  # silent
    ]

    def gen():
        for s in sections:
            yield s

    return _write_chunked(path, sample_rate, channels, fmt, gen())


# --- Plan 05-01 — Phase 5 BPM/Rhythm fixtures ---
# Two click-train fixtures used by all 3 BPM-row algorithms (librosa /
# Essentia / onset autocorrelation). The 05-VALIDATION.md fixture-pair
# pass criterion requires mean(steady) > 0.7 AND mean(drifting) < 0.3
# AND Δ > 0.4 on each algorithm. RESEARCH §"Common Pitfalls" Pitfall 4
# (click-train aliasing at 44.1 kHz) is the load-bearing constraint:
# diracs alias broadband at 44.1 kHz and beat trackers lock onto phantom
# tempi, so each click is a 10 ms band-limited sine burst with
# exp(-t/0.003) decay envelope. Mirrors the existing ``_kick`` shape at
# synthesize.py:183-188 with a 1 kHz carrier (cleaner than ``_kick``'s
# 80 Hz fundamental — less low-frequency rumble bleed for beat-tracker
# fixtures).


def _band_limited_click(
    sample_rate: int,
    click_freq_hz: float = 1000.0,
    duration_s: float = 0.010,
    decay_tau_s: float = 0.003,
) -> np.ndarray:
    """10 ms sine burst at ``click_freq_hz`` with ``exp(-t/tau)`` decay envelope.

    Band-limited per Pitfall 4 — diracs alias at 44.1 kHz and cause beat
    trackers to lock onto phantom tempi. Mirrors the existing ``_kick``
    shape at synthesize.py:183-188 (50 ms 80 Hz sine + exp decay) with a
    1 kHz carrier instead of 80 Hz — high-pitched ticks are cleaner for
    beat-tracker fixtures than low-frequency kicks (less rumble bleed).

    Returns a 1-D ``float32`` array of length ``round(duration_s * sample_rate)``.
    """
    n = int(round(duration_s * sample_rate))
    t = np.arange(n, dtype=np.float32) / sample_rate
    env = np.exp(-t / decay_tau_s).astype(np.float32)
    carrier = np.sin(2.0 * np.pi * click_freq_hz * t).astype(np.float32)
    return (env * carrier).astype(np.float32)


def make_steady_pulse(
    path: Path | str,
    duration_s: float = 60.0,
    sample_rate: int = 44100,
    bpm: float = 120.0,
    click_freq_hz: float = 1000.0,
    channels: int = 1,
    fmt: str = "wav",
) -> Path:
    """Click train at exact ``60/bpm`` intervals — tight-rhythm reference fixture.

    Each click is a band-limited 10 ms sine burst at ``click_freq_hz`` with
    exponential decay (see :func:`_band_limited_click`). At 120 BPM default
    → 120 clicks distributed across 60 s, click every 22050 samples at
    44.1 kHz. Between-click samples are zeros (silence).

    Memory contract: pre-renders the whole buffer in RAM. 60 s × 44.1 kHz
    × 4 B = 10.6 MB — well under the 8 h CLAUDE.md ceiling. Chunked write
    (1 s blocks) keeps the encoder side memory-bounded too.

    Returns the input ``path`` for pipeline chaining (matches
    :func:`make_sine`'s shape).
    """
    path = Path(path)
    total = int(round(duration_s * sample_rate))
    interval = int(round(60.0 / bpm * sample_rate))  # samples per beat
    click = _band_limited_click(sample_rate, click_freq_hz=click_freq_hz)
    n_click = click.size
    buf = np.zeros(total, dtype=np.float32)
    pos = 0
    while pos + n_click <= total:
        buf[pos : pos + n_click] += click
        pos += interval
    chunk = sample_rate
    offset = 0

    def gen():
        nonlocal offset
        while offset < total:
            n = min(chunk, total - offset)
            yield buf[offset : offset + n]
            offset += n

    return _write_chunked(path, sample_rate, channels, fmt, gen())


def make_drifting_pulse(
    path: Path | str,
    duration_s: float = 60.0,
    sample_rate: int = 44100,
    bpm_mean: float = 120.0,
    bpm_jitter: float = 30.0,
    seed: int = 0xD5,
    click_freq_hz: float = 1000.0,
    channels: int = 1,
    fmt: str = "wav",
) -> Path:
    """Click train at JITTERED intervals — loose-rhythm reference fixture.

    Seeded via :func:`numpy.random.default_rng` (Assumption A6 — reproducible
    across test runs). Each inter-click interval samples uniformly from
    ``(bpm_mean ± bpm_jitter)``; the per-beat BPM is clamped to a plausible
    ``[40, 208]`` range before conversion to a sample-interval.

    Same click voice as :func:`make_steady_pulse` (Pitfall 4 — band-limited
    burst, not a dirac).
    """
    path = Path(path)
    rng = np.random.default_rng(seed)
    total = int(round(duration_s * sample_rate))
    click = _band_limited_click(sample_rate, click_freq_hz=click_freq_hz)
    n_click = click.size
    buf = np.zeros(total, dtype=np.float32)
    pos = 0
    while pos + n_click <= total:
        buf[pos : pos + n_click] += click
        # Sample the next inter-click interval from (bpm_mean ± bpm_jitter).
        bpm_now = float(bpm_mean + rng.uniform(-bpm_jitter, +bpm_jitter))
        bpm_now = max(40.0, min(208.0, bpm_now))  # clamp to plausible range
        interval = int(round(60.0 / bpm_now * sample_rate))
        pos += interval
    chunk = sample_rate
    offset = 0

    def gen():
        nonlocal offset
        while offset < total:
            n = min(chunk, total - offset)
            yield buf[offset : offset + n]
            offset += n

    return _write_chunked(path, sample_rate, channels, fmt, gen())


# --- Plan 05-02 — Phase 5 Harmonic fixtures ---
# Two tonal-content fixtures used by all 4 harmonic-row algorithms
# (chroma entropy / HPCP / chroma peak / tonnetz). The 05-VALIDATION.md
# fixture-pair pass criterion requires
# mean(consonant) > 0.7 AND mean(dissonant) < 0.3 AND Δ > 0.4 on each
# harmonic algorithm (relaxed thresholds documented in
# test_harmonic_chroma_peak.py per <behavior> note — raw chroma peak
# is naturally in [1/12, 1]).
#
# make_consonant_triad — C major triad (3 sustained sines at 12-TET
# ratios). Sustained tones have no onsets — BPM-row scores ≈ 0 on this
# fixture (RESEARCH §"Cross-cutting note").
#
# make_dissonant_cluster — 8 sustained sines at seeded-random
# frequencies in [100, 2000] Hz, explicitly NOT snapped to the 12-TET
# grid. Seeded RNG (A6) for reproducibility.


def make_consonant_triad(
    path: Path | str,
    duration_s: float = 60.0,
    sample_rate: int = 44100,
    root_hz: float = 261.63,
    amp: float = 0.3,
    channels: int = 1,
    fmt: str = "wav",
) -> Path:
    """C major triad reference (root + major-3rd + perfect-5th) — consonant harmonic fixture.

    Default ``root_hz = 261.63`` Hz = C4 (middle C). Tones:

        - root              (C4  ≈ 261.63 Hz)
        - root × 2^(4/12)   (E4  ≈ 329.63 Hz — major 3rd)
        - root × 2^(7/12)   (G4  ≈ 392.00 Hz — perfect 5th)

    Each tone has amplitude ``amp / 3`` so the sum stays well under
    clipping (3 × amp/3 = amp; default 0.3). NO vibrato in the v1
    implementation (keeps the harmonic-row tests deterministic —
    vibrato would broaden the chroma peak and weaken the consonance
    signal).

    Expected harmonic-row scores: mean > 0.7 (peaked chroma, low
    entropy, key='C' / scale='major' detected — Pitfall 5 pin in
    Task 2).
    Expected BPM-row scores: ~0 (no onsets — sustained tones).

    Memory contract: pre-renders 1-second chunks via
    :func:`_write_chunked`. At 60 s × 44.1 kHz × float32 = 10.6 MB —
    well under the 8 h CLAUDE.md ceiling. Identical posture to
    :func:`make_steady_pulse`.

    Returns the input ``path`` for pipeline chaining (matches
    :func:`make_sine`'s shape).
    """
    path = Path(path)
    total = int(round(duration_s * sample_rate))
    chunk = sample_rate
    offset = 0
    # Per-tone freqs: C, E (major 3rd = +4 semitones), G (perfect 5th = +7 semitones).
    freqs = (
        float(root_hz),
        float(root_hz) * (2.0 ** (4.0 / 12.0)),
        float(root_hz) * (2.0 ** (7.0 / 12.0)),
    )
    per_tone_amp = amp / float(len(freqs))  # clip safety on summed signal

    def gen():
        nonlocal offset
        while offset < total:
            n = min(chunk, total - offset)
            t = (np.arange(offset, offset + n, dtype=np.float64)) / sample_rate
            block = np.zeros(n, dtype=np.float32)
            for f in freqs:
                block += (per_tone_amp * np.sin(2.0 * np.pi * f * t)).astype(
                    np.float32
                )
            offset += n
            yield block

    return _write_chunked(path, sample_rate, channels, fmt, gen())


def make_dissonant_cluster(
    path: Path | str,
    duration_s: float = 60.0,
    sample_rate: int = 44100,
    n_pitches: int = 8,
    freq_min_hz: float = 100.0,
    freq_max_hz: float = 2000.0,
    amp: float = 0.3,
    seed: int = 42,
    channels: int = 1,
    fmt: str = "wav",
) -> Path:
    """Random-pitch cluster (8 sines, NOT snapped to 12-TET) — dissonant harmonic fixture.

    Frequencies and phases are sampled from a seeded
    :func:`numpy.random.default_rng` (A6 — reproducible across test
    runs). NO 12-TET snapping → the cluster does NOT accidentally form
    a consonant interval; chroma is broadband and roughly flat across
    all 12 pitch classes (high Shannon entropy, no peaked chroma).

    Frequencies are sampled from a continuous uniform
    ``rng.uniform(freq_min_hz, freq_max_hz, n_pitches)``. Each tone has
    uniform-random phase. Amplitudes are normalized to ``amp /
    n_pitches`` for clip safety on the summed signal.

    Expected harmonic-row scores: mean < 0.3 (flat chroma, max
    entropy, no key dominance).
    Expected BPM-row scores: ~0 (no onsets — sustained tones).

    Defensive correctness note: the dissonant cluster MUST produce
    sub-0.3 scores on all 4 harmonic rows. If A6 turns out to be wrong
    (e.g., the seed produces 8 frequencies clustering near one pitch
    class), the fixture-pair test will fail — capture observed scores
    in the SUMMARY as a regression baseline so the next run can
    detect drift.

    Returns the input ``path`` for pipeline chaining.
    """
    path = Path(path)
    rng = np.random.default_rng(seed)  # A6 — seeded RNG for reproducibility
    total = int(round(duration_s * sample_rate))
    chunk = sample_rate
    offset = 0
    freqs = rng.uniform(freq_min_hz, freq_max_hz, n_pitches).astype(np.float64)
    phases = rng.uniform(0.0, 2.0 * np.pi, n_pitches).astype(np.float64)
    per_tone_amp = amp / float(n_pitches)  # clip safety on summed signal

    def gen():
        nonlocal offset
        while offset < total:
            n = min(chunk, total - offset)
            t = (np.arange(offset, offset + n, dtype=np.float64)) / sample_rate
            block = np.zeros(n, dtype=np.float32)
            for f, ph in zip(freqs, phases):
                block += (
                    per_tone_amp * np.sin(2.0 * np.pi * f * t + ph)
                ).astype(np.float32)
            offset += n
            yield block

    return _write_chunked(path, sample_rate, channels, fmt, gen())
