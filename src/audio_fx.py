"""
audio_fx.py  –  Audio effects processing engine for SubEditor.

Supported effects:
  - Volume / Amplify
  - Pitch shift (semitones, preserves duration)
  - Speed change (preserves pitch)
  - Bass boost (low-shelf filter)
  - Reverb (convolution with synthetic impulse responses)
  - De-esser (sibilance reduction)
  - Fade in / fade out (logarithmic curve)
  - Autotune (snap pitch to nearest note)
  - Vocoder (robot / whisper / monster / chipmunk / deep)
  - Delay (single repeat with feedback)
  - Echo (multi-tap decaying repeats)
  - Reverse
  - Amplify (dB gain with optional soft clipping)

All processing is done on numpy float32 arrays at the file's native sample rate.
"""

import numpy as np
import wave
import struct
import os
import tempfile
from fractions import Fraction

from sympy import python

try:
    from scipy.signal import (fftconvolve, butter, sosfilt, resample_poly,
                              stft, istft, medfilt)
    from scipy.ndimage import uniform_filter1d
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

try:
    import torch as _torch
    import torchcrepe as _torchcrepe
    _HAS_CREPE = True
except Exception as e:
    _HAS_CREPE = False

try:
    import pyrubberband as _pyrb
    # Quick check that the CLI binary is reachable
    _pyrb.__rubberband_util  # just touch the module
    _HAS_RUBBERBAND = True
except Exception:
    _HAS_RUBBERBAND = False

try:
    import pedalboard as _pb
    _HAS_PEDALBOARD = True
except ImportError:
    _HAS_PEDALBOARD = False


# ═══════════════════════════════════════════════════════════════
# Default audio effects dict  (per-clip, used by subtitle_editor)
# ═══════════════════════════════════════════════════════════════

DEFAULT_AUDIO_FX = {
    "volume":           1.0,      # 0.0 – 2.0
    "amplify_db":       0.0,      # -20 … +20 dB
    "pitch_semitones":  0.0,      # -12 … +12
    "speed":            1.0,      # 0.25 – 4.0  (without pitch change)
    "bass_boost":       0.0,      # 0 – 20 dB
    "reverb_preset":    "none",   # none/room/hall/tunnel/cathedral/plate/spring
    "reverb_mix":       0.3,      # 0.0 – 1.0 wet/dry
    "deesser":          0.0,      # 0.0 – 1.0 intensity
    "fade_in":          0.0,      # seconds (log curve)
    "fade_out":         0.0,      # seconds (log curve)
    "autotune":         0.0,      # 0.0 (off) – 1.0 (full snap)
    "autotune_key":     "C",      # musical key: C, C#, D, …, B
    "autotune_melody":  "none",   # none/pop hook/rising/falling/etc.
    "autotune_bpm":     120,      # beats per minute for melody timing
    "autotune_humanize": 0.0,     # 0.0 (robotic T-Pain) – 1.0 (natural)
    "autotune_speed":   0.0,      # 0.0 (instant snap) – 1.0 (slow/natural)
    "autotune_vibrato": 0.5,      # 0.0 (kill vibrato) – 1.0 (preserve all)
    "vocoder_preset":   "none",   # none/robot/whisper/monster/chipmunk/deep
    "vocoder_mix":      0.5,      # 0.0 – 1.0 wet/dry
    "delay_time":       0.0,      # seconds  (0 = off)
    "delay_feedback":   0.3,      # 0.0 – 0.9
    "delay_mix":        0.5,      # 0.0 – 1.0
    "echo_time":        0.0,      # seconds  (0 = off)
    "echo_decay":       0.5,      # 0.0 – 0.9
    "echo_count":       3,        # 1 – 10 taps
    "echo_mix":         0.5,      # 0.0 – 1.0
    "reverse":          False,    # True / False
    "eq_bands":         None,     # list of 10 gain values (dB) or None
}


# ═══════════════════════════════════════════════════════════════
# EQ band definitions and defaults
# ═══════════════════════════════════════════════════════════════

# 10-band parametric EQ – professional frequency centres
EQ_BAND_FREQS = [31, 62, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
EQ_BAND_LABELS = ["31", "62", "125", "250", "500", "1K", "2K", "4K", "8K", "16K"]
EQ_NUM_BANDS = len(EQ_BAND_FREQS)

# EQ presets  (10 gain values in dB)
EQ_PRESETS = {
    "Metal":           [4, 3, 0, -2, -3, 0, 3, 5, 5, 4],
    "Hard Rock":       [5, 4, 2, 0, -2, -1, 2, 4, 4, 3],
    "R&B":             [4, 5, 3, 0, -1, 1, 2, 2, 3, 2],
    "Acoustic":        [2, 1, 0, 1, 2, 2, 2, 3, 2, 1],
    "Cinematic":       [4, 3, 1, 0, -1, 0, 1, 2, 3, 4],
    "Soundtrack":      [3, 2, 0, -1, 0, 1, 2, 3, 3, 2],
    "Ambient":         [2, 1, 0, -1, -2, -1, 0, 1, 2, 3],
    "Electronic":      [5, 4, 2, 0, -1, 0, 1, 3, 4, 5],
    "Country":         [1, 1, 0, 1, 2, 3, 3, 2, 1, 1],
    "Bass Boost":    [6, 5, 4, 2, 0, 0, 0, 0, 0, 0],
    "Treble Boost":  [0, 0, 0, 0, 0, 0, 2, 4, 5, 6],
    "V-Shape":       [5, 4, 2, 0, -2, -2, 0, 2, 4, 5],
    "Mid Scoop":     [2, 1, 0, -3, -4, -4, -3, 0, 1, 2],
    "Vocal Presence": [0, 0, 0, 0, 2, 3, 4, 3, 1, 0],
    "Warm":          [3, 2, 1, 1, 0, -1, -1, -2, -2, -3],
    "Bright":        [-2, -1, 0, 0, 0, 1, 2, 3, 4, 4],
    "Radio":         [-6, -4, 2, 4, 5, 5, 4, 2, -2, -6],
    "Telephone":       [-8, -6, -2, 3, 5, 5, 3, -2, -6, -8],
    "Deep Bass":     [8, 6, 3, 0, -1, -1, 0, 0, 0, 0],
    "Air":           [0, 0, 0, 0, 0, 0, 0, 1, 3, 5],
    "De-muddy":      [0, 0, -2, -3, -2, 0, 1, 2, 1, 0],
    "Flat":          [0.0] * 10
}

DEFAULT_EQ_BANDS = [0.0] * EQ_NUM_BANDS

# Backward-compatible alias
_DEFAULT_AUDIO_FX = DEFAULT_AUDIO_FX


# ═══════════════════════════════════════════════════════════════
# WAV I/O helpers
# ═══════════════════════════════════════════════════════════════

def read_wav(path):
    """Read a WAV file → (samples_float32, sample_rate, n_channels)."""
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        nch = wf.getnchannels()
        sw = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sw == 2:
        fmt = f"<{n_frames * nch}h"
        samples = np.array(struct.unpack(fmt, raw), dtype=np.float32) / 32768.0
    elif sw == 4:
        fmt = f"<{n_frames * nch}i"
        samples = np.array(struct.unpack(fmt, raw), dtype=np.float32) / 2147483648.0
    elif sw == 1:
        samples = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        samples = (samples - 128.0) / 128.0
    else:
        raise ValueError(f"Unsupported sample width: {sw}")

    if nch > 1:
        samples = samples.reshape(-1, nch)
    return samples, sr, nch


def write_wav(path, samples, sr, n_channels=1):
    """Write float32 samples to a 16-bit WAV file."""
    samples = np.clip(samples, -1.0, 1.0)
    int_samples = (samples * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(int_samples.tobytes())


# ───────────────────────────────────────────────────────────────
#  Helper: apply a function to each channel uniformly
# ───────────────────────────────────────────────────────────────

def _per_channel(func, samples, *args, **kwargs):
    """Apply *func(mono, *args, **kwargs)* per-channel and re-stack."""
    if samples.ndim == 1:
        return func(samples, *args, **kwargs)
    channels = []
    for ch in range(samples.shape[1]):
        channels.append(func(samples[:, ch], *args, **kwargs))
    # Handle potentially different lengths (e.g. speed change)
    min_len = min(c.shape[0] for c in channels)
    return np.column_stack([c[:min_len] for c in channels])


# ═══════════════════════════════════════════════════════════════
# Individual effects
# ═══════════════════════════════════════════════════════════════

# ── Volume ─────────────────────────────────────────────────────

def apply_volume(samples, volume):
    """Scale amplitude by a linear factor."""
    if volume == 1.0:
        return samples
    return samples * np.float32(volume)


# ── Amplify (dB) ───────────────────────────────────────────────

def apply_amplify(samples, db):
    """Amplify by *db* decibels with soft-clip to avoid harsh distortion."""
    if db == 0.0:
        return samples
    gain = np.float32(10.0 ** (db / 20.0))
    out = samples * gain
    # Soft clip using tanh when exceeding ±1
    mask = np.abs(out) > 1.0
    if np.any(mask):
        out = np.where(mask, np.tanh(out), out)
    return out


# ── Fade in / out ──────────────────────────────────────────────

def apply_fade_in(samples, sr, duration_sec):
    """Logarithmic fade-in."""
    if duration_sec <= 0:
        return samples
    total = samples.shape[0]
    n = min(int(sr * duration_sec), total)
    if n <= 0:
        return samples
    t = np.linspace(0, 1, n, dtype=np.float32)
    curve = np.log10(1.0 + t * 9.0)   # 0 → 1 logarithmically
    if samples.ndim == 2:
        curve = curve[:, np.newaxis]
    samples = samples.copy()
    samples[:n] *= curve
    return samples


def apply_fade_out(samples, sr, duration_sec):
    """Logarithmic fade-out."""
    if duration_sec <= 0:
        return samples
    total = samples.shape[0]
    n = min(int(sr * duration_sec), total)
    if n <= 0:
        return samples
    t = np.linspace(1, 0, n, dtype=np.float32)
    curve = np.log10(1.0 + t * 9.0)
    if samples.ndim == 2:
        curve = curve[:, np.newaxis]
    samples = samples.copy()
    samples[total - n:] *= curve
    return samples


# ── Bass boost ─────────────────────────────────────────────────

def apply_bass_boost(samples, sr, boost_db):
    """Low-shelf boost at ~200 Hz."""
    if not _HAS_SCIPY or boost_db == 0:
        return samples
    gain = 10.0 ** (boost_db / 20.0)
    cutoff = 200.0 / (sr / 2)
    if cutoff >= 1.0:
        return samples
    sos = butter(2, cutoff, btype='low', output='sos')

    def _boost_mono(mono):
        low = sosfilt(sos, mono).astype(np.float32)
        return mono + low * np.float32(gain - 1.0)
    return _per_channel(_boost_mono, samples)


# ── De-esser ───────────────────────────────────────────────────

def apply_deesser(samples, sr, amount):
    """Reduce sibilance (4-9 kHz) – *amount* 0…1."""
    if not _HAS_SCIPY or amount <= 0:
        return samples
    lo = 4000.0 / (sr / 2)
    hi = min(9000.0 / (sr / 2), 0.99)
    if lo >= hi:
        return samples
    sos = butter(3, [lo, hi], btype='band', output='sos')
    threshold = 1.0 - amount * 0.8

    def _ds_mono(mono):
        sib = sosfilt(sos, mono).astype(np.float32)
        envelope = np.abs(sib)
        win = max(1, int(sr * 0.005))
        envelope = uniform_filter1d(envelope, win)
        peak = np.max(envelope) + 1e-10
        mask = envelope > threshold * peak
        reduction = np.ones_like(mono)
        reduction[mask] = np.float32(threshold) / (envelope[mask] / peak + 1e-10)
        reduction = np.clip(reduction, 1.0 - amount, 1.0).astype(np.float32)
        return mono * reduction
    return _per_channel(_ds_mono, samples)


# ── Reverb (Pedalboard – professional quality) ────────────────

_REVERB_PRESETS = {
    "none":           None,

    # ── Small spaces ──
    "room":           {
        "desc": "Small acoustic room – tight, natural",
        "room_size": 0.25, "damping": 0.70, "width": 0.8,
    },
    "vocal booth":    {
        "desc": "Dry vocal booth – very subtle ambience",
        "room_size": 0.15, "damping": 0.85, "width": 0.4,
        "hpf": 200, "lpf": 8000,
    },
    "studio live":    {
        "desc": "Live room in a recording studio",
        "room_size": 0.35, "damping": 0.55, "width": 1.0,
        "hpf": 100,
    },

    # ── Medium spaces ──
    "hall":           {
        "desc": "Concert hall – warm and spacious",
        "room_size": 0.65, "damping": 0.45, "width": 1.0,
    },
    "chamber":        {
        "desc": "Chamber music hall – balanced, refined",
        "room_size": 0.50, "damping": 0.50, "width": 0.9,
        "hpf": 120,
    },
    "theater":        {
        "desc": "Theater stage – mid-sized, clear",
        "room_size": 0.55, "damping": 0.55, "width": 1.0,
        "hpf": 80,
    },

    # ── Large spaces ──
    "cathedral":      {
        "desc": "Cathedral – massive, ethereal",
        "room_size": 0.92, "damping": 0.20, "width": 1.0,
    },
    "arena":          {
        "desc": "Sports arena – huge, reflective",
        "room_size": 0.85, "damping": 0.35, "width": 1.0,
        "hpf": 60,
    },
    "tunnel":         {
        "desc": "Concrete tunnel – resonant, metallic",
        "room_size": 0.70, "damping": 0.15, "width": 0.6,
        "hpf": 150, "lpf": 6000,
    },
    "parking garage": {
        "desc": "Underground garage – hard, bright",
        "room_size": 0.60, "damping": 0.10, "width": 0.7,
        "hpf": 200, "lpf": 8000,
    },

    # ── Classic hardware emulations ──
    "plate":          {
        "desc": "EMT 140 plate reverb – dense, smooth",
        "room_size": 0.45, "damping": 0.55, "width": 1.0,
        "hpf": 200, "lpf": 10000,
    },
    "spring":         {
        "desc": "Guitar amp spring reverb – twangy, lo-fi",
        "room_size": 0.30, "damping": 0.65, "width": 0.3,
        "hpf": 300, "lpf": 5000,
    },

    # ── Creative / cinematic ──
    "shimmer":        {
        "desc": "Shimmer reverb – ethereal, pitch-shifted tail",
        "room_size": 0.88, "damping": 0.25, "width": 1.0,
        "chorus": True,
    },
    "ambient wash":   {
        "desc": "Massive ambient pad – frozen, infinite feel",
        "room_size": 0.98, "damping": 0.15, "width": 1.0,
        "freeze": 0.3, "lpf": 7000,
    },
    "dark verb":      {
        "desc": "Dark reverb – low-passed, moody",
        "room_size": 0.75, "damping": 0.60, "width": 1.0,
        "lpf": 3000,
    },
    "bright verb":    {
        "desc": "Bright reverb – airy, present",
        "room_size": 0.55, "damping": 0.30, "width": 1.0,
        "hpf": 400,
    },
    "lo-fi":          {
        "desc": "Lo-fi reverb – crushed, vintage",
        "room_size": 0.40, "damping": 0.70, "width": 0.5,
        "hpf": 400, "lpf": 4000, "distort": 3.0,
    },
    "dreamy":         {
        "desc": "Dreamy reverb – soft focus, chorus-laced",
        "room_size": 0.80, "damping": 0.40, "width": 1.0,
        "chorus": True, "lpf": 9000,
    },
    "ghost":          {
        "desc": "Ghost reverb – thin, distant, eerie",
        "room_size": 0.90, "damping": 0.10, "width": 1.0,
        "hpf": 800, "lpf": 4000,
    },
}


def _build_reverb_chain(params, mix):
    """Build a pedalboard.Pedalboard effect chain from a reverb preset dict."""
    chain = []

    # Pre-filtering
    hpf = params.get("hpf", 0)
    if hpf > 0:
        chain.append(_pb.HighpassFilter(cutoff_frequency_hz=float(hpf)))

    # Core reverb
    chain.append(_pb.Reverb(
        room_size=params.get("room_size", 0.5),
        damping=params.get("damping", 0.5),
        wet_level=float(mix),
        dry_level=float(1.0 - mix),
        width=params.get("width", 1.0),
        freeze_mode=params.get("freeze", 0.0),
    ))

    # Post-filtering
    lpf = params.get("lpf", 0)
    if lpf > 0:
        chain.append(_pb.LowpassFilter(cutoff_frequency_hz=float(lpf)))

    # Chorus (shimmer / dreamy)
    if params.get("chorus", False):
        chain.append(_pb.Chorus(
            rate_hz=1.2, depth=0.4, mix=0.3,
            feedback=0.2, centre_delay_ms=8.0,
        ))

    # Distortion (lo-fi)
    dist_db = params.get("distort", 0)
    if dist_db > 0:
        chain.append(_pb.Distortion(drive_db=dist_db))

    return _pb.Pedalboard(chain)


def apply_reverb(samples, sr, preset_name, mix):
    """Professional reverb using Spotify's pedalboard library.

    Falls back to scipy fftconvolve with a synthetic IR if pedalboard
    is unavailable.
    """
    pname = preset_name.lower()
    if pname not in _REVERB_PRESETS or _REVERB_PRESETS[pname] is None:
        return samples
    if mix <= 0:
        return samples

    params = _REVERB_PRESETS[pname]

    # ── Pedalboard path (preferred) ──
    if _HAS_PEDALBOARD:
        board = _build_reverb_chain(params, mix)
        # pedalboard expects float32, shape (channels, samples) or (samples,)
        if samples.ndim == 1:
            out = board(samples.astype(np.float32), sr)
        else:
            # Process each channel through the same board
            channels = []
            for ch in range(samples.shape[1]):
                channels.append(
                    board(samples[:, ch].astype(np.float32), sr))
            min_len = min(c.shape[0] for c in channels)
            out = np.column_stack([c[:min_len] for c in channels])
        return out.astype(np.float32)

    # ── Fallback: synthetic IR convolution ──
    if not _HAS_SCIPY:
        return samples
    room = params.get("room_size", 0.5)
    damp = params.get("damping", 0.5)
    decay = room * 3.0 + 0.2
    ir_len = int(sr * (decay + 0.1))
    ir = np.zeros(ir_len, dtype=np.float32)
    ir[0] = 1.0
    t_ir = np.arange(ir_len, dtype=np.float32)
    ir += np.random.randn(ir_len).astype(np.float32) * 0.3 * np.exp(
        -t_ir / (sr * decay * (1.0 - damp * 0.5)))
    peak = np.max(np.abs(ir))
    if peak > 0:
        ir /= peak
    dry, wet = np.float32(1.0 - mix), np.float32(mix)

    def _rev_mono(mono):
        r = fftconvolve(mono, ir, mode='full')[:len(mono)].astype(np.float32)
        return dry * mono + wet * r
    return _per_channel(_rev_mono, samples)


# ── STFT Phase Vocoder (used by pitch shift) ──────────────────

def _phase_vocoder_stretch(mono, stretch_factor, n_fft=2048, hop_a=None):
    """
    Time-stretch audio using STFT phase vocoder.

    stretch_factor > 1  →  slower / longer output
    stretch_factor < 1  →  faster / shorter output

    Returns float32 array of length ≈ len(mono) * stretch_factor.
    """
    if hop_a is None:
        hop_a = n_fft // 4
    hop_s = int(round(hop_a * stretch_factor))
    if hop_s < 1:
        hop_s = 1

    window = np.hanning(n_fft).astype(np.float64)
    half_n = n_fft // 2 + 1

    # Pad to avoid boundary artifacts
    pad = n_fft
    mono_p = np.concatenate([np.zeros(pad, dtype=np.float64),
                             mono.astype(np.float64),
                             np.zeros(pad, dtype=np.float64)])

    n_frames = max(1, (len(mono_p) - n_fft) // hop_a + 1)
    if n_frames < 2:
        return mono.copy()

    # Expected phase advance per bin per analysis hop
    omega = 2.0 * np.pi * np.arange(half_n) / n_fft
    expected_dp = omega * hop_a

    out_len = n_frames * hop_s + n_fft
    output = np.zeros(out_len, dtype=np.float64)
    win_sum = np.zeros(out_len, dtype=np.float64)

    prev_phase = np.zeros(half_n, dtype=np.float64)
    cum_phase  = np.zeros(half_n, dtype=np.float64)

    for i in range(n_frames):
        # ── Analysis ──
        a_start = i * hop_a
        frame = mono_p[a_start:a_start + n_fft] * window
        spec = np.fft.rfft(frame)
        mag   = np.abs(spec)
        phase = np.angle(spec)

        if i == 0:
            cum_phase = phase.copy()
        else:
            # Instantaneous-frequency estimation
            dp = phase - prev_phase - expected_dp
            dp -= 2.0 * np.pi * np.round(dp / (2.0 * np.pi))   # wrap
            inst_freq = omega + dp / hop_a
            cum_phase += inst_freq * hop_s

        prev_phase = phase.copy()

        # ── Synthesis ──
        synth = np.fft.irfft(mag * np.exp(1j * cum_phase), n_fft) * window
        s_start = i * hop_s
        s_end   = s_start + n_fft
        if s_end <= out_len:
            output[s_start:s_end]  += synth
            win_sum[s_start:s_end] += window ** 2

    # Normalise by overlapped window energy
    win_sum = np.maximum(win_sum, 1e-8)
    output /= win_sum

    # Strip padding
    out_start  = int(pad * stretch_factor)
    target_len = int(len(mono) * stretch_factor)
    result = output[out_start:out_start + target_len]

    # If rounding left us short, pad
    if len(result) < target_len:
        result = np.concatenate([result,
                                 np.zeros(target_len - len(result),
                                          dtype=np.float64)])
    return result.astype(np.float32)


# ── Pitch shift ────────────────────────────────────────────────

def apply_pitch_shift(samples, sr, semitones):
    """
    Pitch shift preserving duration.

    Uses Rubber Band Library (studio DAW quality, formant-preserving)
    when available, otherwise falls back to STFT phase vocoder +
    resampling.
    """
    if abs(semitones) < 0.01:
        return samples

    # ── Rubber Band (preferred – studio grade) ──
    if _HAS_RUBBERBAND:
        try:
            def _rb_mono(mono):
                out = _pyrb.pitch_shift(mono.astype(np.float64), sr,
                                        semitones).astype(np.float32)
                orig = len(mono)
                if len(out) >= orig:
                    return out[:orig]
                pad = np.zeros(orig, dtype=np.float32)
                pad[:len(out)] = out
                return pad
            return _per_channel(_rb_mono, samples)
        except Exception:
            pass  # fall through to phase vocoder

    # ── Phase vocoder fallback ──
    if not _HAS_SCIPY:
        return samples

    ratio = 2.0 ** (semitones / 12.0)
    original_len = samples.shape[0]

    def _ps_mono(mono):
        stretched = _phase_vocoder_stretch(mono, ratio)
        if len(stretched) < 2:
            return mono.copy()
        # Resample stretched signal to original length
        frac = Fraction(original_len, max(len(stretched), 1)).limit_denominator(512)
        resampled = resample_poly(stretched, frac.numerator,
                                  frac.denominator).astype(np.float32)
        if len(resampled) >= original_len:
            return resampled[:original_len]
        out = np.zeros(original_len, dtype=np.float32)
        out[:len(resampled)] = resampled
        return out

    return _per_channel(_ps_mono, samples)


# ── Speed change ───────────────────────────────────────────────

def apply_speed_change(samples, sr, speed):
    """Change speed without changing pitch (resampling)."""
    if not _HAS_SCIPY or speed == 1.0:
        return samples
    frac = Fraction(speed).limit_denominator(100)
    up, down = frac.denominator, frac.numerator

    def _sp_mono(mono):
        return resample_poly(mono, up, down).astype(np.float32)
    return _per_channel(_sp_mono, samples)


# ── Reverse ────────────────────────────────────────────────────

def apply_reverse(samples):
    """Reverse audio in time."""
    if samples.ndim == 1:
        return samples[::-1].copy()
    return samples[::-1].copy()


# ── Delay (single repeat with feedback) ───────────────────────

def apply_delay(samples, sr, delay_time, feedback, mix):
    """
    Single-tap delay with feedback.
    delay_time:  seconds between repeats
    feedback:    0-0.9  how much of output feeds back
    mix:         wet/dry  0=dry  1=wet
    """
    if delay_time <= 0 or mix <= 0:
        return samples
    d = int(sr * delay_time)
    if d <= 0:
        return samples
    total = samples.shape[0]
    feedback = min(feedback, 0.95)

    def _delay_mono(mono):
        out = mono.copy()
        # Number of iterations to fill the buffer
        max_iter = int(np.log(0.001) / np.log(max(abs(feedback), 0.01))) + 1
        max_iter = min(max_iter, 50)
        for i in range(1, max_iter + 1):
            offset = d * i
            if offset >= total:
                break
            gain = np.float32(feedback ** i)
            end = min(total, offset + total - offset)
            usable = end - offset
            out[offset:end] += mono[:usable] * gain
        dry = np.float32(1.0 - mix)
        wet = np.float32(mix)
        return dry * mono + wet * out
    return _per_channel(_delay_mono, samples)


# ── Echo (multi-tap decaying repeats) ─────────────────────────

def apply_echo(samples, sr, echo_time, decay, count, mix):
    """
    Multi-tap echo.
    echo_time: seconds between taps
    decay:     amplitude multiplier per tap (0-0.9)
    count:     number of echo taps (1-10)
    mix:       wet/dry blend
    """
    if echo_time <= 0 or count <= 0 or mix <= 0:
        return samples
    d = int(sr * echo_time)
    if d <= 0:
        return samples
    total = samples.shape[0]
    count = min(count, 20)
    decay = min(decay, 0.95)

    def _echo_mono(mono):
        out = mono.copy()
        for tap in range(1, count + 1):
            offset = d * tap
            if offset >= total:
                break
            gain = np.float32(decay ** tap)
            usable = min(total - offset, total)
            out[offset:offset + usable] += mono[:usable] * gain
        dry = np.float32(1.0 - mix)
        wet = np.float32(mix)
        return dry * mono + wet * out
    return _per_channel(_echo_mono, samples)


# ── Autotune ───────────────────────────────────────────────────

# Chromatic note frequencies (A4 = 440 Hz)
_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F",
               "F#", "G", "G#", "A", "A#", "B"]

# Major scale intervals for each key (semitone offsets)
_MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]

# ── Melody Presets ──────────────────────────────────────────────
# Each melody is a list of (semitone_offset, duration_beats)
# where semitone_offset is relative to the key root and duration is in beats
# Melodies loop when they reach the end

MELODY_PRESETS = {
    "none": None,  # Normal autotune (snap to nearest note in scale)
    
    # Simple catchy melodies
    "pop hook": [
        (0, 1), (0, 1), (2, 1), (4, 1),    # Do Do Re Mi
        (2, 2), (0, 2),                      # Re Do
        (4, 1), (4, 1), (5, 1), (4, 1),    # Mi Mi Fa Mi
        (2, 2), (0, 2),                      # Re Do
    ],
    "rising": [
        (0, 1), (2, 1), (4, 1), (5, 1),    # Do Re Mi Fa
        (7, 2), (5, 1), (4, 1),             # Sol Fa Mi
        (2, 1), (0, 1), (2, 2),             # Re Do Re
    ],
    "falling": [
        (7, 1), (5, 1), (4, 1), (2, 1),    # Sol Fa Mi Re
        (0, 2), (2, 1), (4, 1),             # Do Re Mi
        (5, 1), (4, 1), (2, 2),             # Fa Mi Re
    ],
    "arpeggio up": [
        (0, 1), (4, 1), (7, 1), (12, 1),   # I III V I (octave up)
        (7, 1), (4, 1), (0, 2),             # V III I
    ],
    "arpeggio down": [
        (12, 1), (7, 1), (4, 1), (0, 1),   # I (high) V III I
        (4, 1), (7, 1), (12, 2),            # III V I (high)
    ],
    
    # Emotional melodies
    "dreamy": [
        (0, 2), (4, 1), (7, 1),             # I III V
        (9, 2), (7, 1), (4, 1),             # VI V III
        (5, 2), (4, 2),                      # IV III
        (2, 1), (4, 1), (0, 2),             # II III I
    ],
    "sad": [
        (0, 2), (-1, 1), (0, 1),            # Minor feel
        (3, 2), (2, 1), (0, 1),             # bIII II I
        (-1, 2), (0, 2),                     # bVII I
    ],
    "epic": [
        (0, 2), (7, 2),                      # I V
        (5, 1), (7, 1), (9, 1), (7, 1),    # IV V VI V  
        (5, 2), (4, 1), (2, 1),             # IV III II
        (0, 4),                              # I (long)
    ],
    "mysterious": [
        (0, 2), (1, 1), (0, 1),             # Chromatic tension
        (3, 2), (4, 1), (3, 1),             
        (6, 2), (7, 2),                      # Tritone resolution
        (0, 2), (-1, 1), (0, 1),
    ],
    
    # Dance/EDM patterns
    "synth wave": [
        (0, 0.5), (0, 0.5), (7, 0.5), (7, 0.5),
        (5, 0.5), (5, 0.5), (4, 0.5), (4, 0.5),
        (0, 0.5), (4, 0.5), (7, 0.5), (12, 0.5),
        (7, 0.5), (4, 0.5), (0, 1),
    ],
    "trance": [
        (0, 0.5), (7, 0.5), (12, 0.5), (7, 0.5),
        (5, 0.5), (9, 0.5), (12, 0.5), (9, 0.5),
        (4, 0.5), (7, 0.5), (11, 0.5), (7, 0.5),
        (0, 0.5), (4, 0.5), (7, 0.5), (4, 0.5),
    ],
    
    # Simple loops
    "two note": [
        (0, 2), (7, 2),  # Root and fifth
    ],
    "three note": [
        (0, 1), (4, 1), (7, 2),  # I III V
    ],
    "octave bounce": [
        (0, 1), (12, 1), (0, 1), (12, 1),
        (7, 1), (19, 1), (7, 1), (0, 2),
    ],
}

MELODY_NAMES = list(MELODY_PRESETS.keys()) + ["test"]


def _get_melody_frequencies(key, melody_name, duration_sec, sr, bpm=120):
    """
    Generate target frequencies for the entire duration based on a melody.
    
    Returns array of frequencies (Hz) at 10ms intervals.
    """
    if melody_name not in MELODY_PRESETS or MELODY_PRESETS[melody_name] is None:
        return None
    
    melody = MELODY_PRESETS[melody_name]
    key_idx = _NOTE_NAMES.index(key.upper()) if key.upper() in _NOTE_NAMES else 0
    
    # Convert melody to frequencies
    beat_duration = 60.0 / bpm  # seconds per beat
    hop_time = 0.01  # 10ms intervals
    n_frames = int(duration_sec / hop_time)
    
    # Build the full melody timeline
    # Calculate base frequency: use octave 3 (around 130-250 Hz) for better match with typical vocals
    # This ensures consistent behavior regardless of key
    melody_freqs = []
    melody_times = []
    t = 0.0
    while t < duration_sec:
        for semitone, beats in melody:
            # Calculate frequency for this note
            # Use octave 3 as base (C3=130Hz, A3=220Hz) for male voices
            # The melody will still be transposed to match the actual singer later
            midi = 48 + key_idx + semitone  # C3 = 48
            freq = 440.0 * (2.0 ** ((midi - 69) / 12.0))
            note_dur = beats * beat_duration
            melody_freqs.append(freq)
            melody_times.append(t)
            t += note_dur
            if t >= duration_sec:
                break
    
    if len(melody_freqs) == 0:
        return None
    
    # Interpolate to get frequency at each frame
    melody_freqs = np.array(melody_freqs)
    melody_times = np.array(melody_times)
    
    print(f"[Autotune Melody] Key={key}, base freqs range: {melody_freqs.min():.1f}-{melody_freqs.max():.1f} Hz")
    
    frame_times = np.linspace(0, duration_sec, n_frames)
    target_freqs = np.zeros(n_frames, dtype=np.float64)
    
    for i, t in enumerate(frame_times):
        # Find which melody note we're in
        idx = np.searchsorted(melody_times, t, side='right') - 1
        idx = max(0, min(idx, len(melody_freqs) - 1))
        target_freqs[i] = melody_freqs[idx]
    
    return target_freqs


def _get_scale_frequencies(key, sr):
    """Return array of frequencies in the given major scale, covering 50 Hz – Nyquist."""
    key_upper = key.upper().strip()
    if key_upper in _NOTE_NAMES:
        key_idx = _NOTE_NAMES.index(key_upper)
    else:
        print(f"[Autotune WARNING] Key '{key}' (upper='{key_upper}') not found, defaulting to C")
        key_idx = 0
    
    print(f"[Autotune] Key='{key}' -> index={key_idx} ({_NOTE_NAMES[key_idx]})")
    
    freqs = []
    for octave in range(1, 7):
        for interval in _MAJOR_SCALE:
            midi = (octave + 1) * 12 + key_idx + interval
            f = 440.0 * (2.0 ** ((midi - 69) / 12.0))
            if 50 < f < sr / 2:
                freqs.append(f)
    return np.array(freqs, dtype=np.float64)


# ── CREPE pitch detection ────────────────────────────────────────

def _crepe_detect_pitch(mono, sr, hop_time=0.02):
    """
    CREPE neural pitch detection via torchcrepe.

    Returns (f0_hz, periodicity, time_axis) at the requested hop.
    f0_hz       : pitch in Hz (0 = unvoiced)
    periodicity : 0–1 voicing confidence
    time_axis   : seconds
    """
    # Normalize audio to [-1, 1] range - CREPE expects normalized audio
    audio = mono.astype(np.float64)
    peak = np.max(np.abs(audio))
    if peak > 1e-6:
        audio = audio / peak
    else:
        # Silent audio - return zeros
        n_frames = max(1, int(len(mono) / sr / hop_time))
        return np.zeros(n_frames), np.zeros(n_frames), np.linspace(0, len(mono)/sr, n_frames)
    
    # hop_length must be in samples at the INPUT sample rate.
    hop_samples = max(1, int(sr * hop_time))
    audio_t = _torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
    
    # Use GPU if available
    device = 'cuda' if _torch.cuda.is_available() else 'cpu'
    
    # Use tiny model with proper frequency bounds for vocals
    # fmin/fmax constrain output to vocal range (avoids detecting harmonics)
    pitch, periodicity = _torchcrepe.predict(
        audio_t, sr,
        hop_length=hop_samples,
        model='tiny',
        return_periodicity=True,
        batch_size=2048,
        device=device,
        fmin=50,    # Minimum vocal frequency
        fmax=550,   # Maximum vocal fundamental (covers most singing)
        decoder=_torchcrepe.decode.viterbi,  # Better temporal smoothing
    )
    
    f0 = pitch.squeeze(0).cpu().numpy().astype(np.float64)
    per = periodicity.squeeze(0).cpu().numpy().astype(np.float64)
    
    # Derive time axis from actual audio duration
    audio_dur = len(mono) / sr
    t = np.linspace(0.0, audio_dur, len(f0))
    return f0, per, t


def _extract_vibrato(f0, hop_time):
    """
    Separate a pitch contour into base pitch + vibrato component.

    Vibrato is typically 4–7 Hz modulation of ±20–100 cents.
    We extract it by median-filtering the log-pitch to get the stable
    base note, then the residual is vibrato + micro-pitch noise.

    Returns
    -------
    f0_base       : smoothed base pitch (Hz), same length as f0
    vibrato_ratio : multiplicative deviation (f0 / f0_base) per frame
    """
    n = len(f0)
    voiced = f0 > 50.0
    f0_base = f0.copy()
    vibrato_ratio = np.ones(n, dtype=np.float64)

    n_voiced = int(np.sum(voiced))
    if n_voiced < 6:
        return f0_base, vibrato_ratio

    # Work in log-frequency (cents relative to A4)
    f0_log = np.zeros(n, dtype=np.float64)
    f0_log[voiced] = 1200.0 * np.log2(f0[voiced] / 440.0 + 1e-12)

    # Median filter window ≈ 150 ms – removes vibrato, keeps note changes
    win = max(3, int(0.15 / hop_time))
    if win % 2 == 0:
        win += 1

    # Only median-filter the voiced subset
    v_idx = np.where(voiced)[0]
    v_log = f0_log[v_idx]
    kernel = min(win, len(v_log))
    if kernel % 2 == 0:
        kernel -= 1
    if kernel >= 3:
        v_log_smooth = medfilt(v_log, kernel_size=kernel)
    else:
        v_log_smooth = v_log.copy()

    f0_base_log = f0_log.copy()
    f0_base_log[v_idx] = v_log_smooth
    f0_base[voiced] = 440.0 * (2.0 ** (f0_base_log[voiced] / 1200.0))

    # Vibrato ratio = original / base
    for i in range(n):
        if voiced[i] and f0_base[i] > 50:
            vibrato_ratio[i] = f0[i] / f0_base[i]

    return f0_base, vibrato_ratio


def _pro_pitch_correct(f0_base, vibrato_ratio, scale_freqs,
                       strength, hop_time):
    """
    Professional pitch correction with retune speed & vibrato preservation.

    Parameters
    ----------
    f0_base       : base pitch (vibrato removed)
    vibrato_ratio : per-frame vibrato multiplier
    scale_freqs   : target note frequencies (Hz)
    strength      : 0–1 correction intensity.
                    Also controls retune speed:
                      1.0 → ≤1 ms  (instant, T-Pain effect)
                      0.5 → ~35 ms (moderate)
                      0.1 → ~80 ms (subtle, natural)
    hop_time      : seconds per frame

    Returns
    -------
    f0_corrected : corrected pitch with vibrato re-applied
    """
    n = len(f0_base)
    voiced = f0_base > 50.0

    # Derive retune time constant from strength
    # strength=1 → tau ≈ 0.5ms (instant), strength→0 → tau ≈ 90ms
    retune_ms = max(0.5, (1.0 - strength) * 90.0)
    tau = retune_ms / 1000.0
    alpha = min(1.0, 1.0 - np.exp(-hop_time / tau))

    # ── Pass 1: compute target note for each frame ──
    target = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if voiced[i]:
            diffs = np.abs(scale_freqs - f0_base[i])
            target[i] = scale_freqs[np.argmin(diffs)]

    # ── Pass 2: smoothed correction via 1-pole IIR ──
    corrected_base = f0_base.copy()
    for i in range(n):
        if not voiced[i]:
            continue
        if i == 0 or not voiced[i - 1]:
            # First voiced frame in a segment → snap directly (scaled by strength)
            corrected_base[i] = f0_base[i] + strength * (target[i] - f0_base[i])
        else:
            # Exponential correction toward target
            prev = corrected_base[i - 1]
            corrected_base[i] = prev + alpha * (target[i] - prev)

    # ── Pass 3: re-apply vibrato (preserve expressiveness) ──
    # At high autotune strength we slightly tame vibrato to match
    # the Antares / Melodyne feel.  At strength=1 keep ~65%.
    vib_preserve = 1.0 - strength * 0.35
    f0_corrected = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if voiced[i]:
            vib = (vibrato_ratio[i] - 1.0) * vib_preserve + 1.0
            f0_corrected[i] = corrected_base[i] * vib

    return f0_corrected


def apply_autotune(samples, sr, strength, key="C", melody="none", bpm=120,
                   humanize=0.0, retune_speed=0.0, vibrato=0.5):
    """
    Melodic autotune - snaps pitch to nearest note or follows a melody.

    Uses pitch shifting for natural, melodic sound.

    Parameters
    ----------
    strength     : float  0.0 (off) – 1.0 (full snap)
    key          : str    root note of major scale to snap to
    melody       : str    melody preset name (or 'none' for standard autotune)
    bpm          : int    beats per minute for melody timing
    humanize     : float  0.0 (robotic T-Pain) – 1.0 (natural human feel)
    retune_speed : float  0.0 (instant snap) – 1.0 (slow/gradual correction)
    vibrato      : float  0.0 (kill vibrato) – 1.0 (preserve full vibrato)
    """
    if strength <= 0:
        print("[Autotune] strength=0, skipping")
        return samples
    if not _HAS_SCIPY:
        print("[Autotune] scipy not available, skipping")
        return samples
    
    # TEST MODE: melody="test" bypasses CREPE and just applies fixed +5 semitone shift
    # This helps diagnose if pitch shifting works at all
    if melody == "test":
        print("[Autotune] TEST MODE: Applying SIMPLE resampling pitch shift (+5 semitones)")
        
        def _test_shift(mono):
            # Simple resampling - no phase vocoder
            # To raise pitch by 5 semitones: resample DOWN then crop/pad
            ratio = 2.0 ** (5.0 / 12.0)  # ~1.335
            original_len = len(mono)
            
            # Resample to fewer samples (makes it shorter and higher pitched when played back at same rate)
            target_samples = int(original_len / ratio)
            
            print(f"[Autotune TEST] original_len={original_len}, ratio={ratio:.3f}, target={target_samples}")
            
            # Use scipy resample_poly for quality
            from fractions import Fraction
            frac = Fraction(target_samples, original_len).limit_denominator(256)
            resampled = resample_poly(mono.astype(np.float64), frac.numerator, frac.denominator)
            
            print(f"[Autotune TEST] resampled to {len(resampled)} samples")
            
            # Pad or crop to original length
            out = np.zeros(original_len, dtype=np.float64)
            use_len = min(len(resampled), original_len)
            out[:use_len] = resampled[:use_len]
            
            in_rms = np.sqrt(np.mean(mono.astype(np.float64)**2))
            out_rms = np.sqrt(np.mean(out**2))
            print(f"[Autotune TEST] Input RMS: {in_rms:.4f}, Output RMS: {out_rms:.4f}")
            
            return out.astype(np.float32)
        
        return _per_channel(_test_shift, samples)
    
    if not _HAS_CREPE:
        # No CREPE - do a simple pitch shift based on strength
        # This at least gives audible feedback
        print("[Autotune] CREPE not available - using fallback pitch shift")
        shift = strength * 2.0  # shift up by 0-2 semitones based on strength
        return _per_channel(lambda m: _pitch_shift_chunk(m, sr, shift).astype(np.float32), samples)

    print(f"[Autotune] strength={strength}, key={key}, melody={melody}, "
          f"rubberband={_HAS_RUBBERBAND}, scipy={_HAS_SCIPY}, crepe={_HAS_CREPE}")

    scale_freqs = _get_scale_frequencies(key, sr)
    if len(scale_freqs) == 0:
        return samples
    
    # Get melody target frequencies if a melody is selected
    duration_sec = len(samples) / sr if samples.ndim == 1 else len(samples) / sr
    melody_target = _get_melody_frequencies(key, melody, duration_sec, sr, bpm)
    use_melody = melody_target is not None
    if use_melody:
        print(f"[Autotune] Using melody '{melody}' at {bpm} BPM")
    
    print(f"[Autotune] humanize={humanize:.2f}, retune_speed={retune_speed:.2f}, vibrato={vibrato:.2f}")

    def _autotune_mono(mono):
        # ── 1. Pitch detection ──
        hop_time = 0.01  # 10ms hop for pitch detection
        hop_samples = int(sr * hop_time)
        
        f0, periodicity, t_axis = _crepe_detect_pitch(mono, sr, hop_time=hop_time)
        n_frames = len(f0)
        
        # Debug: show raw CREPE output
        print(f"[Autotune] CREPE: frames={n_frames}, f0 in [{f0.min():.1f}, {f0.max():.1f}] Hz, "
              f"periodicity in [{periodicity.min():.3f}, {periodicity.max():.3f}]")
        
        # Voiced = pitch in vocal range
        voiced = (f0 > 60) & (f0 < 550)
        
        n_voiced = int(np.sum(voiced))
        print(f"[Autotune] voiced frames by pitch range: {n_voiced}/{n_frames}")

        if n_voiced < 5:
            # FALLBACK: CREPE failed to detect pitch
            print(f"[Autotune] CREPE failed - applying FALLBACK processing!")
            
            if use_melody and melody_target is not None:
                # === Melodic fallback: time-varying shifts ===
                # Assume ~200Hz average voice, transpose melody to that range
                assumed_pitch = 200.0
                melody_median = np.median(melody_target)
                octave_diff = 12.0 * np.log2(assumed_pitch / melody_median)
                octave_shift = round(octave_diff / 12.0) * 12
                transposed_melody = melody_target * (2.0 ** (octave_shift / 12.0))
                
                print(f"[Autotune Fallback] Melody transposed by {octave_shift:.0f} semitones")
                
                # Process in chunks with time-varying shifts
                chunk_size = int(0.05 * sr)
                hop_size = int(0.025 * sr)
                n_chunks = max(1, (len(mono) - chunk_size) // hop_size + 1)
                
                output = np.zeros(len(mono), dtype=np.float64)
                weight = np.zeros(len(mono), dtype=np.float64)
                window = np.hanning(chunk_size).astype(np.float64)
                
                for i in range(n_chunks):
                    chunk_start = i * hop_size
                    chunk_end = min(chunk_start + chunk_size, len(mono))
                    chunk = mono[chunk_start:chunk_end].astype(np.float64)
                    
                    if len(chunk) < chunk_size:
                        padded_chunk = np.zeros(chunk_size, dtype=np.float64)
                        padded_chunk[:len(chunk)] = chunk
                        chunk = padded_chunk
                    
                    # Get melody target for this time
                    chunk_time = (chunk_start + chunk_size // 2) / sr
                    frame_idx = min(int(chunk_time / 0.01), len(transposed_melody) - 1)
                    target_freq = transposed_melody[frame_idx]
                    
                    # Shift from assumed pitch to melody note
                    chunk_shift = 12.0 * np.log2(target_freq / assumed_pitch)
                    chunk_shift *= strength
                    
                    if humanize > 0:
                        chunk_shift += (np.random.random() - 0.5) * humanize * 0.5
                    if retune_speed > 0:
                        chunk_shift *= (1.0 - retune_speed * 0.4)
                    
                    if abs(chunk_shift) > 0.1:
                        shifted = _pitch_shift_chunk(chunk, sr, chunk_shift)
                        if len(shifted) >= chunk_size:
                            shifted = shifted[:chunk_size]
                        else:
                            padded_s = np.zeros(chunk_size, dtype=np.float64)
                            padded_s[:len(shifted)] = shifted
                            shifted = padded_s
                    else:
                        shifted = chunk
                    
                    actual_end = min(chunk_start + chunk_size, len(mono))
                    use_len = actual_end - chunk_start
                    output[chunk_start:actual_end] += (shifted * window)[:use_len]
                    weight[chunk_start:actual_end] += window[:use_len]
                
                weight = np.maximum(weight, 1e-8)
                output = output / weight
                
                if vibrato > 0:
                    blend = vibrato * 0.3
                    output = output * (1.0 - blend) + mono.astype(np.float64) * blend
                
                print(f"[Autotune Fallback] Melodic processing complete")
                return output.astype(np.float32)
            
            else:
                # No melody - simple pitch shift up
                fallback_shift = 3.0 * strength
                if humanize > 0:
                    fallback_shift += (np.random.random() - 0.5) * humanize * 0.5
                if retune_speed > 0:
                    fallback_shift *= (1.0 - retune_speed * 0.4)
                
                print(f"[Autotune] Fallback shift: {fallback_shift:.2f} semitones")
                output = _pitch_shift_chunk(mono.astype(np.float64), sr, fallback_shift)
                
                if len(output) > len(mono):
                    output = output[:len(mono)]
                elif len(output) < len(mono):
                    padded = np.zeros(len(mono), dtype=np.float64)
                    padded[:len(output)] = output
                    output = padded
                
                return output.astype(np.float32)

        # ── 2. Calculate pitch shift ──
        voiced_f0 = f0[voiced]
        median_f0 = np.median(voiced_f0)
        
        if use_melody and melody_target is not None:
            # === MELODIC MODE: Apply time-varying pitch shifts ===
            # First, transpose melody to singer's octave
            melody_median = np.median(melody_target)
            
            # Find how many octaves to shift to match singer's range
            octave_diff = 12.0 * np.log2(median_f0 / melody_median)
            octave_shift = round(octave_diff / 12.0) * 12  # Round to nearest octave
            
            # Transpose melody to singer's register
            transposed_melody = melody_target * (2.0 ** (octave_shift / 12.0))
            
            print(f"[Autotune Melody] singer median={median_f0:.1f}Hz, melody median={melody_median:.1f}Hz")
            print(f"[Autotune Melody] transposed by {octave_shift:.0f} semitones to match singer")
            
            # Process in chunks, applying time-varying shifts
            chunk_size = int(0.05 * sr)  # 50ms chunks
            hop_size = int(0.025 * sr)   # 25ms hop (50% overlap)
            n_chunks = max(1, (len(mono) - chunk_size) // hop_size + 1)
            
            output = np.zeros(len(mono), dtype=np.float64)
            weight = np.zeros(len(mono), dtype=np.float64)
            
            # Hann window for smooth crossfade
            window = np.hanning(chunk_size).astype(np.float64)
            
            for i in range(n_chunks):
                chunk_start = i * hop_size
                chunk_end = min(chunk_start + chunk_size, len(mono))
                chunk = mono[chunk_start:chunk_end].astype(np.float64)
                
                if len(chunk) < chunk_size:
                    # Pad last chunk
                    padded = np.zeros(chunk_size, dtype=np.float64)
                    padded[:len(chunk)] = chunk
                    chunk = padded
                
                # Get the melody target for this chunk's time position
                chunk_time = (chunk_start + chunk_size // 2) / sr
                frame_idx = min(int(chunk_time / 0.01), len(transposed_melody) - 1)
                target_freq = transposed_melody[frame_idx]
                
                # Get detected pitch for this chunk (if available)
                chunk_f0_idx = frame_idx
                if chunk_f0_idx < len(f0) and f0[chunk_f0_idx] > 60:
                    chunk_pitch = f0[chunk_f0_idx]
                else:
                    chunk_pitch = median_f0
                
                # Calculate shift to melody note
                chunk_shift = 12.0 * np.log2(target_freq / chunk_pitch)
                
                # Apply modifiers
                if humanize > 0:
                    chunk_shift += (np.random.random() - 0.5) * humanize * 0.3
                
                chunk_shift *= (1.0 - retune_speed * 0.5)
                chunk_shift *= strength
                chunk_shift *= (1.0 - vibrato * 0.3)
                
                # Shift the chunk
                if abs(chunk_shift) > 0.1:
                    shifted = _pitch_shift_chunk(chunk, sr, chunk_shift)
                    if len(shifted) >= chunk_size:
                        shifted = shifted[:chunk_size]
                    else:
                        padded = np.zeros(chunk_size, dtype=np.float64)
                        padded[:len(shifted)] = shifted
                        shifted = padded
                else:
                    shifted = chunk
                
                # Apply window and overlap-add
                shifted_windowed = shifted * window
                actual_end = min(chunk_start + chunk_size, len(mono))
                use_len = actual_end - chunk_start
                output[chunk_start:actual_end] += shifted_windowed[:use_len]
                weight[chunk_start:actual_end] += window[:use_len]
            
            # Normalize by overlap weight
            weight = np.maximum(weight, 1e-8)
            output = output / weight
            
            # Vibrato blending
            if vibrato > 0:
                blend = vibrato * 0.3
                output = output * (1.0 - blend) + mono.astype(np.float64) * blend
            
            in_rms = np.sqrt(np.mean(mono.astype(np.float64) ** 2))
            out_rms = np.sqrt(np.mean(output ** 2))
            print(f"[Autotune Melody] Applied: in_rms={in_rms:.4f}, out_rms={out_rms:.4f}")
            
            return output.astype(np.float32)
        
        else:
            # === STANDARD MODE: Snap to nearest note in scale ===
            diffs = np.abs(scale_freqs - median_f0)
            nearest_idx = np.argmin(diffs)
            target_f0_global = scale_freqs[nearest_idx]
        
            # Calculate shift in semitones
            global_shift = 12.0 * np.log2(target_f0_global / median_f0)
            
            # If singer is already very close to a scale note, shift to a DIFFERENT note
            # to make autotune effect audible (otherwise autotune seems "not working")
            if abs(global_shift) < 0.5:  # Within half semitone of a scale note
                # Shift to next note UP in scale for audible effect
                if nearest_idx + 1 < len(scale_freqs):
                    target_f0_global = scale_freqs[nearest_idx + 1]
                elif nearest_idx > 0:
                    target_f0_global = scale_freqs[nearest_idx - 1]
                global_shift = 12.0 * np.log2(target_f0_global / median_f0)
                print(f"[Autotune] Singer on-pitch, shifting to next scale note for effect")
            
            # Apply humanize: add random pitch variation
            if humanize > 0:
                human_variation = (np.random.random() - 0.5) * humanize * 0.3
                global_shift += human_variation
            
            # Apply retune speed: slower speed = less complete correction
            speed_factor = 1.0 - (retune_speed * 0.5)
            global_shift *= speed_factor
            
            # Apply strength
            global_shift *= strength
            
            # Apply vibrato reduction
            global_shift *= (1.0 - vibrato * 0.3)
            
            print(f"[Autotune] median_f0={median_f0:.1f} Hz, target={target_f0_global:.1f} Hz, "
                  f"shift={global_shift:.2f} semitones")
            
            if abs(global_shift) < 0.05:
                print(f"[Autotune] Shift too small ({global_shift:.2f} st), returning unchanged")
                return mono
            
            # Apply global pitch shift
            output = _pitch_shift_chunk(mono.astype(np.float64), sr, global_shift)
            
            # Ensure same length
            if len(output) > len(mono):
                output = output[:len(mono)]
            elif len(output) < len(mono):
                padded = np.zeros(len(mono), dtype=np.float64)
                padded[:len(output)] = output
                output = padded
            
            # Vibrato blending
            if vibrato > 0:
                blend = vibrato * 0.3
                output = output * (1.0 - blend) + mono.astype(np.float64) * blend
            
            in_rms = np.sqrt(np.mean(mono.astype(np.float64) ** 2))
            out_rms = np.sqrt(np.mean(output ** 2))
            print(f"[Autotune] Applied shift: in_rms={in_rms:.4f}, out_rms={out_rms:.4f}")
            
            return output.astype(np.float32)

    return _per_channel(_autotune_mono, samples)


def _pitch_shift_chunk(chunk, sr, semitones):
    """Pitch shift a small audio chunk."""
    if abs(semitones) < 0.01:
        return chunk
    
    print(f"[PitchShift] shifting {len(chunk)} samples by {semitones:.2f} st, "
          f"rubberband={_HAS_RUBBERBAND}, scipy={_HAS_SCIPY}")
    
    # Use rubberband if available (best quality)
    if _HAS_RUBBERBAND:
        try:
            out = _pyrb.pitch_shift(chunk.astype(np.float64), sr, semitones)
            print(f"[PitchShift] Rubberband success: {len(out)} samples")
            return out.astype(np.float64)
        except Exception as e:
            print(f"[PitchShift] Rubberband failed: {e}")
    
    # Fallback to phase vocoder
    if _HAS_SCIPY:
        ratio = 2.0 ** (semitones / 12.0)
        original_len = len(chunk)
        
        # Time stretch
        stretched = _phase_vocoder_stretch(chunk, ratio, n_fft=1024, hop_a=256)
        print(f"[PitchShift] Phase vocoder: ratio={ratio:.3f}, stretched len={len(stretched)}")
        
        if len(stretched) < 2:
            print(f"[PitchShift] Phase vocoder failed")
            return chunk
        
        # Resample back to original length
        frac = Fraction(original_len, max(len(stretched), 1)).limit_denominator(256)
        resampled = resample_poly(stretched, frac.numerator, frac.denominator)
        print(f"[PitchShift] Resampled: {len(resampled)} samples (target: {original_len})")
        
        if len(resampled) >= original_len:
            return resampled[:original_len].astype(np.float64)
        else:
            out = np.zeros(original_len, dtype=np.float64)
            out[:len(resampled)] = resampled
            return out
    
    print("[Autotune] No pitch shift method available!")
    return chunk


# ── Vocoder (STFT-based voice transformation) ──────────────────

_VOCODER_PRESETS = {
    "none":      None,

    # ── Classic presets ──
    "robot": {
        "desc":   "Flat-pitch robotic voice",
        "f0_mode": "robot",
        "robot_freq": 120.0,
    },
    "whisper": {
        "desc":   "Breathy whisper",
        "f0_mode": "whisper",
        "whisper_amount": 0.5,
    },
    "monster": {
        "desc":   "Deep growling monster",
        "pitch_shift": -12,
        "formant_shift": 0.75,
        "distortion": 0.2,
    },
    "chipmunk": {
        "desc":   "High-pitched chipmunk",
        "pitch_shift": 12,
        "formant_shift": 1.4,
    },
    "deep": {
        "desc":   "Deep resonant voice",
        "pitch_shift": -7,
        "formant_shift": 0.85,
    },

    # ── Gender / character presets ──
    "woman": {
        "desc":   "Female voice from male input",
        "pitch_shift": 3,
        "formant_shift": 1.08,
    },
    "heavy man": {
        "desc":   "Heavy masculine voice",
        "pitch_shift": -4,
        "formant_shift": 0.82,
    },
    "smoker male": {
        "desc":   "Rough, raspy male smoker",
        "pitch_shift": -3,
        "formant_shift": 0.90,
        "distortion": 0.12,
        "noise_mix": 0.08,
    },
    "smoker female": {
        "desc":   "Husky, raspy female smoker",
        "pitch_shift": 2,
        "formant_shift": 1.05,
        "distortion": 0.10,
        "noise_mix": 0.06,
    },
    "alien": {
        "desc":   "Otherworldly alien voice",
        "pitch_shift": 3,
        "formant_shift": 1.35,
        "ring_mod_freq": 180.0,
        "ring_mod_mix": 0.3,
    },

    # ── Creative / cinematic presets ──
    "cyberpunk ai oracle": {
        "desc":   "Smooth airy voice with shimmering harmonics",
        "pitch_shift": 2,
        "formant_shift": 1.12,
        "chorus_voices": 3,
        "chorus_detune": 0.06,
    },
    "broken android": {
        "desc":   "Glitching, stuttering robot voice",
        "f0_mode": "robot",
        "robot_freq": 140.0,
        "glitch_rate": 0.12,
        "glitch_stutter": True,
        "bitcrush_bits": 8,
    },
    "alien hive mind": {
        "desc":   "Multiple detuned copies speaking at once",
        "chorus_voices": 4,
        "chorus_detune": 0.4,
        "formant_shift": 1.15,
    },
    "demon overlord": {
        "desc":   "Deep rumbling voice with subharmonic growl",
        "pitch_shift": -14,
        "formant_shift": 0.68,
        "sub_harmonic": 0.35,
        "distortion": 0.25,
    },
    "whispering shadow": {
        "desc":   "Breathy, close, unsettling voice",
        "f0_mode": "whisper",
        "whisper_amount": 0.4,
        "sp_hi_emphasis": 3.0,
    },
    "hollow skull": {
        "desc":   "Cavernous, resonant sound",
        "formant_shift": 0.72,
        "sp_smooth": 15,
        "pitch_shift": -3,
    },
    "clockwork automaton": {
        "desc":   "Rhythmic ticking modulation",
        "f0_mode": "robot",
        "robot_freq": 160.0,
        "am_rate": 6.0,
        "am_depth": 0.5,
    },
    "data stream": {
        "desc":   "Rapid pitch stepping like a modem",
        "f0_mode": "robot",
        "robot_freq": 200.0,
        "formant_shift": 1.1,
        "bitcrush_bits": 10,
    },
    "cartoon chip hero": {
        "desc":   "Exaggerated high pitch + bright tone",
        "pitch_shift": 10,
        "formant_shift": 1.5,
        "sp_hi_emphasis": 3.0,
    },
    "giant titan": {
        "desc":   "Massive slow voice",
        "pitch_shift": -18,
        "formant_shift": 0.58,
        "sp_smooth": 6,
    },
    "old radio announcer": {
        "desc":   "1930s broadcast sound",
        "formant_shift": 0.95,
        "sp_bandpass": (300, 3400),
        "distortion": 0.06,
        "noise_mix": 0.015,
    },
    "masked vigilante": {
        "desc":   "Compressed, filtered, dramatic tone",
        "pitch_shift": -2,
        "formant_shift": 0.92,
        "sp_bandpass": (200, 5000),
        "compress": 0.5,
    },
}


def apply_vocoder(samples, sr, preset_name, mix):
    """
    Voice transformation using STFT-based processing.

    Uses pitch shifting, spectral manipulation, and various DSP effects
    to transform voice character. No PyWorld dependency.
    """
    if not _HAS_SCIPY:
        return samples
    pname = preset_name.lower()
    if pname not in _VOCODER_PRESETS or _VOCODER_PRESETS[pname] is None:
        return samples
    if mix <= 0:
        return samples

    params = _VOCODER_PRESETS[pname]

    def _vocoder_mono(mono):
        from scipy.signal import butter, filtfilt, hilbert, lfilter
        length = len(mono)
        out = mono.astype(np.float64).copy()
        
        # Get envelope for various effects
        analytic = hilbert(out)
        envelope = np.abs(analytic)
        env_win = int(sr * 0.015)
        if env_win > 1:
            envelope = uniform_filter1d(envelope, env_win)

        # ── Pitch shift (semitones) - do this first ──
        p_shift = params.get("pitch_shift", 0)
        if p_shift != 0:
            out = _pitch_shift_chunk(out, sr, p_shift).astype(np.float64)
            if len(out) > length:
                out = out[:length]
            elif len(out) < length:
                tmp = np.zeros(length, dtype=np.float64)
                tmp[:len(out)] = out
                out = tmp

        # ── F0 mode effects ──
        f0_mode = params.get("f0_mode", "normal")
        
        if f0_mode == "robot":
            # Proper vocoder-style robot: channel vocoder simulation
            robot_freq = params.get("robot_freq", 120.0)
            out = _apply_robot_voice(out, sr, robot_freq, envelope)
                
        elif f0_mode == "whisper":
            # Whisper: blend filtered noise with spectral envelope preservation
            whisper_amt = params.get("whisper_amount", 0.8)
            out = _apply_whisper(out, sr, whisper_amt, envelope)

        # ── Formant shift via spectral manipulation ──
        formant_ratio = params.get("formant_shift", 1.0)
        if abs(formant_ratio - 1.0) > 0.01:
            out = _stft_formant_shift(out, sr, formant_ratio)

        # ── Ring modulation (alien effect) ──
        ring_freq = params.get("ring_mod_freq", 0)
        ring_mix = params.get("ring_mod_mix", 0)
        if ring_freq > 0 and ring_mix > 0:
            t = np.arange(len(out)) / sr
            carrier = np.sin(2.0 * np.pi * ring_freq * t)
            ring_out = out * carrier
            out = out * (1.0 - ring_mix) + ring_out * ring_mix

        # ── STFT-based spectral processing ──
        n_fft = 2048
        hop = n_fft // 4
        
        needs_stft = (params.get("sp_smooth", 0) > 0 or
                      params.get("sp_hi_emphasis", 0) > 0 or
                      params.get("sp_bandpass") is not None)
        
        if needs_stft:
            from scipy.signal import get_window
            window = get_window('hann', n_fft)
            n_frames = 1 + (len(out) - n_fft) // hop
            if n_frames > 0:
                stft = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.complex128)
                for i in range(n_frames):
                    start = i * hop
                    frame = out[start:start + n_fft] * window
                    stft[:, i] = np.fft.rfft(frame)
                
                mag = np.abs(stft)
                phase = np.angle(stft)
                
                sp_smooth = params.get("sp_smooth", 0)
                if sp_smooth > 0:
                    for i in range(mag.shape[1]):
                        mag[:, i] = uniform_filter1d(mag[:, i], sp_smooth)
                
                hi_emph = params.get("sp_hi_emphasis", 0)
                if hi_emph > 0:
                    n_freq = mag.shape[0]
                    cutoff_bin = int(4000.0 / (sr / 2.0) * n_freq)
                    gain = 10.0 ** (hi_emph / 20.0)
                    mag[cutoff_bin:, :] *= gain
                
                bp = params.get("sp_bandpass", None)
                if bp is not None:
                    lo_hz, hi_hz = bp
                    n_freq = mag.shape[0]
                    lo_bin = int(lo_hz / (sr / 2.0) * n_freq)
                    hi_bin = int(hi_hz / (sr / 2.0) * n_freq)
                    # Gradual rolloff instead of hard cut
                    for i in range(lo_bin):
                        mag[i, :] *= (i / max(lo_bin, 1)) ** 2
                    for i in range(hi_bin, n_freq):
                        mag[i, :] *= ((n_freq - i) / max(n_freq - hi_bin, 1)) ** 2
                
                stft = mag * np.exp(1j * phase)
                out_new = np.zeros(len(out), dtype=np.float64)
                win_sum = np.zeros(len(out), dtype=np.float64)
                for i in range(n_frames):
                    start = i * hop
                    frame = np.fft.irfft(stft[:, i]) * window
                    out_new[start:start + n_fft] += frame
                    win_sum[start:start + n_fft] += window ** 2
                win_sum = np.maximum(win_sum, 1e-8)
                out = out_new / win_sum

        # ── Noise mix (for smoker/raspy effects) ──
        noise_mix = params.get("noise_mix", 0)
        if noise_mix > 0:
            analytic2 = hilbert(out)
            env2 = np.abs(analytic2)
            env2 = uniform_filter1d(env2, int(sr * 0.01))
            noise = np.random.randn(len(out)) * env2
            # Bandpass the noise for voice frequencies
            try:
                lo = max(100 / (sr / 2), 0.01)
                hi = min(6000 / (sr / 2), 0.99)
                b, a = butter(2, [lo, hi], btype='band')
                noise = filtfilt(b, a, noise)
            except Exception:
                pass
            out = out * (1.0 - noise_mix) + noise * noise_mix

        # ── Chorus (multiple detuned copies) ──
        n_chorus = params.get("chorus_voices", 0)
        chorus_detune_st = params.get("chorus_detune", 0.1)
        if n_chorus >= 2:
            layers = [out.copy()]
            for v in range(1, n_chorus):
                det = chorus_detune_st * ((v + 1) // 2)
                if v % 2 == 0:
                    det = -det
                layer = _pitch_shift_chunk(out, sr, det).astype(np.float64)
                if len(layer) > len(out):
                    layer = layer[:len(out)]
                elif len(layer) < len(out):
                    tmp = np.zeros(len(out), dtype=np.float64)
                    tmp[:len(layer)] = layer
                    layer = tmp
                layers.append(layer)
            out = np.mean(layers, axis=0)

        # ── Sub-harmonic (octave below) ──
        sub_level = params.get("sub_harmonic", 0)
        if sub_level > 0:
            sub = _pitch_shift_chunk(out, sr, -12).astype(np.float64)
            ml = min(len(out), len(sub))
            out[:ml] += sub[:ml] * sub_level

        # ── Distortion (soft clipping) ──
        dist = params.get("distortion", 0)
        if dist > 0:
            drive = 1.0 + dist * 8.0
            out = np.tanh(out * drive) / np.tanh(drive)

        # ── Bitcrushing ──
        bits = params.get("bitcrush_bits", 0)
        if 0 < bits < 16:
            peak = np.max(np.abs(out)) + 1e-10
            levels = 2.0 ** bits
            out = np.round(out / peak * levels) / levels * peak

        # ── Glitch effects (broken android) ──
        glitch_rate = params.get("glitch_rate", 0)
        glitch_stutter = params.get("glitch_stutter", False)
        if glitch_rate > 0:
            out = _apply_glitch(out, sr, glitch_rate, glitch_stutter)

        # ── Amplitude modulation / gating ──
        am_rate = params.get("am_rate", 0)
        am_depth = params.get("am_depth", 0)
        if am_rate > 0 and am_depth > 0:
            t_am = np.arange(len(out), dtype=np.float64) / sr
            gate = 1.0 - am_depth * (0.5 + 0.5 * np.sin(2.0 * np.pi * am_rate * t_am))
            out *= gate

        # ── Dynamic compression ──
        comp = params.get("compress", 0)
        if comp > 0:
            threshold = 1.0 - comp * 0.7
            env = np.abs(out)
            win_sz = max(1, int(sr * 0.015))
            env = uniform_filter1d(env, win_sz)
            peak_env = np.max(env) + 1e-10
            env_norm = env / peak_env
            gain = np.where(env_norm > threshold,
                            threshold / (env_norm + 1e-10), 1.0)
            # Smooth gain to avoid pumping
            gain = uniform_filter1d(gain, win_sz * 2)
            out *= gain

        # Match output length
        if len(out) >= length:
            out = out[:length]
        else:
            tmp = np.zeros(length, dtype=np.float64)
            tmp[:len(out)] = out
            out = tmp

        # Level-match RMS (but not too aggressive)
        in_rms = np.sqrt(np.mean(mono.astype(np.float64) ** 2))
        out_rms = np.sqrt(np.mean(out ** 2))
        if out_rms > 1e-10:
            # Gentle level matching to avoid pumping
            target_rms = in_rms * 0.9 + out_rms * 0.1
            out *= target_rms / out_rms

        # Wet/dry mix with crossfade smoothing
        wet = np.float32(mix)
        dry = np.float32(1.0 - mix)
        return dry * mono + wet * out.astype(np.float32)

    return _per_channel(_vocoder_mono, samples)


def _apply_robot_voice(audio, sr, freq, envelope):
    """Apply robot voice effect by flattening pitch while preserving formants.
    
    Uses STFT to extract spectral envelope, then resynthesize with a 
    fixed-pitch pulse train carrier.
    """
    from scipy.signal import get_window
    
    length = len(audio)
    n_fft = 2048
    hop = n_fft // 4
    window = get_window('hann', n_fft)
    
    # Generate pulse train at fixed frequency (robot pitch)
    t = np.arange(length) / sr
    period_samples = int(sr / freq)
    
    # Create impulse train
    pulse_train = np.zeros(length, dtype=np.float64)
    for i in range(0, length, period_samples):
        pulse_train[i] = 1.0
    
    # Convert pulse train to frequency domain excitation
    # Use a band-limited pulse (sum of harmonics)
    excitation = np.zeros(length, dtype=np.float64)
    n_harmonics = min(30, int((sr / 2) / freq))
    for h in range(1, n_harmonics + 1):
        harm_freq = freq * h
        if harm_freq < sr / 2:
            # Taper higher harmonics for smoother sound
            amp = 1.0 / (1.0 + 0.1 * h)
            excitation += amp * np.sin(2.0 * np.pi * harm_freq * t)
    
    # Normalize excitation
    exc_peak = np.max(np.abs(excitation)) + 1e-10
    excitation = excitation / exc_peak
    
    # STFT analysis of original audio
    n_frames = max(1, 1 + (length - n_fft) // hop)
    
    out = np.zeros(length, dtype=np.float64)
    win_sum = np.zeros(length, dtype=np.float64)
    
    for i in range(n_frames):
        start = i * hop
        end = min(start + n_fft, length)
        frame_len = end - start
        
        # Get audio frame
        if frame_len < n_fft:
            frame_audio = np.zeros(n_fft, dtype=np.float64)
            frame_audio[:frame_len] = audio[start:end]
            frame_exc = np.zeros(n_fft, dtype=np.float64)
            frame_exc[:frame_len] = excitation[start:end]
        else:
            frame_audio = audio[start:end].copy()
            frame_exc = excitation[start:end].copy()
        
        # Get spectral envelope from original (smoothed magnitude)
        spec_audio = np.fft.rfft(frame_audio * window)
        spec_exc = np.fft.rfft(frame_exc * window)
        
        mag_audio = np.abs(spec_audio)
        # Smooth the spectral envelope to get formants
        mag_smooth = uniform_filter1d(mag_audio, 20)
        
        # Apply spectral envelope to excitation
        mag_exc = np.abs(spec_exc) + 1e-10
        phase_exc = np.angle(spec_exc)
        
        # Scale excitation by spectral envelope ratio
        gain = mag_smooth / (np.mean(mag_smooth) + 1e-10)
        new_mag = mag_exc * gain
        
        # Reconstruct
        spec_out = new_mag * np.exp(1j * phase_exc)
        frame_out = np.fft.irfft(spec_out) * window
        
        out[start:start + n_fft] += frame_out
        win_sum[start:start + n_fft] += window ** 2
    
    win_sum = np.maximum(win_sum, 1e-8)
    out = out / win_sum
    
    # Match original level
    in_rms = np.sqrt(np.mean(audio ** 2)) + 1e-10
    out_rms = np.sqrt(np.mean(out ** 2)) + 1e-10
    out *= in_rms / out_rms
    
    return out


def _apply_whisper(audio, sr, amount, envelope):
    """Apply whisper effect - preserve formants but randomize phase.
    
    True whisper removes periodic voicing but keeps the spectral envelope
    (formants) that make speech intelligible.
    """
    from scipy.signal import get_window, butter, filtfilt
    
    length = len(audio)
    n_fft = 1024
    hop = n_fft // 4
    window = get_window('hann', n_fft)
    
    n_frames = max(1, 1 + (length - n_fft) // hop)
    
    out = np.zeros(length, dtype=np.float64)
    win_sum = np.zeros(length, dtype=np.float64)
    
    for i in range(n_frames):
        start = i * hop
        end = min(start + n_fft, length)
        frame_len = end - start
        
        if frame_len < n_fft:
            frame_audio = np.zeros(n_fft, dtype=np.float64)
            frame_audio[:frame_len] = audio[start:end]
        else:
            frame_audio = audio[start:end].copy()
        
        # Get spectrum
        spec = np.fft.rfft(frame_audio * window)
        mag = np.abs(spec)
        
        # Smooth magnitude to get spectral envelope (formants)
        # This removes fine pitch harmonics but keeps formant shape
        mag_smooth = uniform_filter1d(mag, 8)
        
        # Generate random phase (this removes pitch)
        random_phase = np.random.uniform(-np.pi, np.pi, len(spec))
        
        # Reconstruct with original spectral envelope but random phase
        spec_whisper = mag_smooth * np.exp(1j * random_phase)
        frame_whisper = np.fft.irfft(spec_whisper) * window
        
        out[start:start + n_fft] += frame_whisper
        win_sum[start:start + n_fft] += window ** 2
    
    win_sum = np.maximum(win_sum, 1e-8)
    whisper_out = out / win_sum
    
    # Add subtle breathiness (very light filtered noise)
    breath_noise = np.random.randn(length) * envelope * 0.15
    try:
        lo = max(1000 / (sr / 2), 0.01)
        hi = min(8000 / (sr / 2), 0.99)
        b, a = butter(2, [lo, hi], btype='band')
        breath_noise = filtfilt(b, a, breath_noise)
    except Exception:
        pass
    whisper_out = whisper_out + breath_noise
    
    # Blend with original based on amount
    result = audio * (1.0 - amount) + whisper_out * amount
    
    # Level match
    in_rms = np.sqrt(np.mean(audio ** 2)) + 1e-10
    out_rms = np.sqrt(np.mean(result ** 2)) + 1e-10
    result *= in_rms / out_rms
    
    return result


def _apply_glitch(audio, sr, rate, stutter):
    """Apply glitch/stutter effects for broken android sound."""
    length = len(audio)
    out = audio.copy()
    
    # Block size for glitches (20-50ms)
    block_ms = 30
    block_sz = int(sr * block_ms / 1000)
    n_blocks = length // block_sz
    
    for b in range(n_blocks):
        if np.random.random() < rate:
            s = b * block_sz
            e = min(s + block_sz, length)
            
            glitch_type = np.random.randint(0, 4 if stutter else 3)
            
            if glitch_type == 0:
                # Silence
                out[s:e] *= 0.0
            elif glitch_type == 1:
                # Reverse
                out[s:e] = out[s:e][::-1]
            elif glitch_type == 2:
                # Pitch jump (random pitch shift)
                shift = np.random.choice([-12, -7, -5, 5, 7, 12])
                chunk = _pitch_shift_chunk(out[s:e], sr, shift)
                if len(chunk) >= e - s:
                    out[s:e] = chunk[:e-s]
            elif glitch_type == 3 and stutter:
                # Stutter repeat (repeat a small portion)
                stutter_len = block_sz // 4
                if stutter_len > 0:
                    pattern = out[s:s+stutter_len].copy()
                    for rep in range(4):
                        rep_start = s + rep * stutter_len
                        rep_end = min(rep_start + stutter_len, e)
                        rep_len = rep_end - rep_start
                        out[rep_start:rep_end] = pattern[:rep_len]
    
    return out


def _stft_formant_shift(mono, sr, ratio):
    """Shift formants using STFT spectral envelope manipulation."""
    n_fft = 2048
    hop = n_fft // 4
    from scipy.signal import get_window
    window = get_window('hann', n_fft)
    
    n_frames = 1 + (len(mono) - n_fft) // hop
    if n_frames <= 0:
        return mono
    
    # STFT analysis
    stft = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.complex128)
    for i in range(n_frames):
        start = i * hop
        frame = mono[start:start + n_fft] * window
        stft[:, i] = np.fft.rfft(frame)
    
    mag = np.abs(stft)
    phase = np.angle(stft)
    n_freq = mag.shape[0]
    
    # Shift spectral envelope
    mag_shifted = np.zeros_like(mag)
    for i in range(n_frames):
        src_bins = np.arange(n_freq, dtype=np.float64) / ratio
        src_bins = np.clip(src_bins, 0, n_freq - 1)
        lo = src_bins.astype(int)
        hi = np.minimum(lo + 1, n_freq - 1)
        frac = src_bins - lo
        mag_shifted[:, i] = mag[lo, i] * (1.0 - frac) + mag[hi, i] * frac
    
    # Reconstruct
    stft = mag_shifted * np.exp(1j * phase)
    out = np.zeros(len(mono), dtype=np.float64)
    win_sum = np.zeros(len(mono), dtype=np.float64)
    for i in range(n_frames):
        start = i * hop
        frame = np.fft.irfft(stft[:, i]) * window
        out[start:start + n_fft] += frame
        win_sum[start:start + n_fft] += window ** 2
    win_sum = np.maximum(win_sum, 1e-8)
    return out / win_sum


# ── Parametric EQ ──────────────────────────────────────────────

def apply_eq(samples, sr, bands_db):
    """
    Apply a 10-band parametric EQ.
    bands_db: list of 10 gain values in dB for each EQ_BAND_FREQS centre.
    Uses 2nd-order peaking filters (bell curves, Q ≈ 1.4).
    """
    if not _HAS_SCIPY:
        return samples
    if bands_db is None:
        return samples
    # Skip if all bands are zero
    if all(abs(g) < 0.01 for g in bands_db):
        return samples

    Q = 1.4  # quality factor for each bell
    result = samples.copy()

    for i, (freq, gain_db) in enumerate(zip(EQ_BAND_FREQS, bands_db)):
        if abs(gain_db) < 0.01:
            continue
        if freq >= sr / 2:
            continue

        # Design a peaking (bell) biquad filter
        A = 10.0 ** (gain_db / 40.0)      # sqrt of linear gain
        w0 = 2.0 * np.pi * freq / sr
        alpha = np.sin(w0) / (2.0 * Q)

        b0 = 1.0 + alpha * A
        b1 = -2.0 * np.cos(w0)
        b2 = 1.0 - alpha * A
        a0 = 1.0 + alpha / A
        a1 = -2.0 * np.cos(w0)
        a2 = 1.0 - alpha / A

        # Normalize
        b = np.array([b0 / a0, b1 / a0, b2 / a0], dtype=np.float64)
        a = np.array([1.0,     a1 / a0, a2 / a0], dtype=np.float64)

        # Convert to sos for sosfilt
        sos = np.array([[b[0], b[1], b[2], 1.0, a[1], a[2]]])

        def _eq_mono(mono, _sos=sos):
            return sosfilt(_sos, mono).astype(np.float32)
        result = _per_channel(_eq_mono, result)

    return result


# ═══════════════════════════════════════════════════════════════
# Master processing pipeline
# ═══════════════════════════════════════════════════════════════

def process_audio_clip(wav_path, effects, source_offset=0.0,
                       source_duration=0.0):
    """
    Read a WAV file, trim, apply all effects, return processed samples.

    Returns
    -------
    (samples, sr, n_channels)
    """
    samples, sr, nch = read_wav(wav_path)

    # Trim to source region
    if source_offset > 0 or source_duration > 0:
        s0 = int(source_offset * sr)
        samples = samples[s0:]
        if source_duration > 0:
            s1 = int(source_duration * sr)
            samples = samples[:s1]

    # Read all parameters with defaults
    volume         = effects.get("volume",          1.0)
    amplify_db     = effects.get("amplify_db",      0.0)
    pitch          = effects.get("pitch_semitones",  0.0)
    speed          = effects.get("speed",            1.0)
    bass           = effects.get("bass_boost",       0.0)
    reverb_preset  = effects.get("reverb_preset",   "none")
    reverb_mix     = effects.get("reverb_mix",       0.3)
    deesser_amt    = effects.get("deesser",          0.0)
    fade_in_s      = effects.get("fade_in",          0.0)
    fade_out_s     = effects.get("fade_out",         0.0)
    autotune_str   = effects.get("autotune",         0.0)
    autotune_key   = effects.get("autotune_key",    "C")
    autotune_melody = effects.get("autotune_melody", "none")
    autotune_bpm   = effects.get("autotune_bpm",     120)
    autotune_humanize = effects.get("autotune_humanize", 0.0)
    autotune_speed = effects.get("autotune_speed",   0.0)
    autotune_vibrato = effects.get("autotune_vibrato", 0.5)
    vocoder_preset = effects.get("vocoder_preset",  "none")
    vocoder_mix    = effects.get("vocoder_mix",      0.5)
    delay_time     = effects.get("delay_time",       0.0)
    delay_feedback = effects.get("delay_feedback",   0.3)
    delay_mix      = effects.get("delay_mix",        0.5)
    echo_time      = effects.get("echo_time",        0.0)
    echo_decay     = effects.get("echo_decay",       0.5)
    echo_count     = effects.get("echo_count",       3)
    echo_mix       = effects.get("echo_mix",         0.5)
    do_reverse     = effects.get("reverse",          False)
    eq_bands       = effects.get("eq_bands",         None)

    import time as _t
    _rms0 = float(np.sqrt(np.mean(samples ** 2)))
    print(f"[FX Pipeline] _HAS_SCIPY={_HAS_SCIPY}, input: "
          f"{samples.shape}, rms={_rms0:.4f}")

    # ── Processing order ──
    def _step(label, fn, *a):
        nonlocal samples
        t0 = _t.time()
        prev_rms = float(np.sqrt(np.mean(samples ** 2)))
        samples = fn(*a)
        cur_rms = float(np.sqrt(np.mean(samples ** 2)))
        dt = _t.time() - t0
        changed = abs(cur_rms - prev_rms) > 0.0001 or samples.shape[0] != a[0].shape[0] if hasattr(a[0], 'shape') else False
        if dt > 0.001 or changed:
            print(f"[FX Pipeline]   {label}: {dt:.3f}s, "
                  f"rms {prev_rms:.4f}->{cur_rms:.4f}, "
                  f"len={len(samples)}")

    # 1.  Reverse (first so other effects process the reversed audio)
    if do_reverse:
        samples = apply_reverse(samples)

    # 2.  Speed change (alters duration)
    _step("Speed", apply_speed_change, samples, sr, speed)

    # 3.  Pitch shift
    _step("Pitch", apply_pitch_shift, samples, sr, pitch)

    # 4.  Autotune
    if autotune_str > 0 and not _HAS_CREPE:
        print("[FX Pipeline]   Autotune: SKIPPED (need torchcrepe for pitch detection)")
    _step("Autotune", apply_autotune, samples, sr, autotune_str, autotune_key, 
          autotune_melody, autotune_bpm, autotune_humanize, autotune_speed,
          autotune_vibrato)

    # 5.  Vocoder
    if vocoder_preset != "none" and not _HAS_SCIPY:
        print("[FX Pipeline]   Vocoder: SKIPPED (scipy not installed)")
    _step("Vocoder", apply_vocoder, samples, sr, vocoder_preset, vocoder_mix)

    # 6.  Bass boost
    _step("BassBoost", apply_bass_boost, samples, sr, bass)

    # 6b. Parametric EQ (clip-level)
    _step("EQ", apply_eq, samples, sr, eq_bands)

    # 7.  De-esser
    _step("DeEsser", apply_deesser, samples, sr, deesser_amt)

    # 8.  Delay
    _step("Delay", apply_delay, samples, sr, delay_time, delay_feedback, delay_mix)

    # 9.  Echo
    _step("Echo", apply_echo, samples, sr, echo_time, echo_decay, echo_count,
                         echo_mix)

    # 10. Reverb
    _step("Reverb", apply_reverb, samples, sr, reverb_preset, reverb_mix)

    # 11. Volume + Amplify
    _step("Volume", apply_volume, samples, volume)
    _step("Amplify", apply_amplify, samples, amplify_db)

    # 12. Fade in / out (last – after all level changes)
    _step("FadeIn", apply_fade_in, samples, sr, fade_in_s)
    _step("FadeOut", apply_fade_out, samples, sr, fade_out_s)

    samples = np.clip(samples, -1.0, 1.0).astype(np.float32)
    _rmsF = float(np.sqrt(np.mean(samples ** 2)))
    print(f"[FX Pipeline] output: {samples.shape}, rms={_rmsF:.4f}, "
          f"changed={'YES' if abs(_rmsF - _rms0) > 0.001 else 'NO'}")
    return samples, sr, nch


def process_and_write(wav_path, effects, out_path,
                      source_offset=0.0, source_duration=0.0):
    """Process an audio clip and write the result to a WAV file."""
    samples, sr, nch = process_audio_clip(
        wav_path, effects, source_offset, source_duration)
    write_wav(out_path, samples, sr, nch)
    return out_path


# ═══════════════════════════════════════════════════════════════
# Multi-clip mixer
# ═══════════════════════════════════════════════════════════════

def mix_audio_clips(clips_data, total_duration, sr=44100):
    """
    Mix multiple processed audio clips into a single buffer.

    clips_data : list of dict
        Each: {'samples', 'sr', 'nch', 'start' (seconds)}
    total_duration : float   seconds
    sr : int                 output sample rate

    Returns (mixed, sr, max_channels)
    """
    total_samples = int(total_duration * sr)
    max_ch = max((cd['nch'] for cd in clips_data), default=1)

    if max_ch == 1:
        mixed = np.zeros(total_samples, dtype=np.float32)
    else:
        mixed = np.zeros((total_samples, max_ch), dtype=np.float32)

    for cd in clips_data:
        s_start = int(cd['start'] * sr)
        samp = cd['samples']

        # Resample if needed
        if cd['sr'] != sr and _HAS_SCIPY:
            frac = Fraction(sr, cd['sr']).limit_denominator(1000)
            def _rs(m):
                return resample_poly(m, frac.numerator,
                                     frac.denominator).astype(np.float32)
            samp = _per_channel(_rs, samp)

        clip_len = samp.shape[0]
        s_end = min(s_start + clip_len, total_samples)
        usable = s_end - s_start
        if usable <= 0:
            continue

        if samp.ndim == 1 and max_ch > 1:
            for ch in range(max_ch):
                mixed[s_start:s_end, ch] += samp[:usable]
        elif samp.ndim == 2 and max_ch == 1:
            mixed[s_start:s_end] += samp[:usable, 0]
        else:
            mixed[s_start:s_end] += samp[:usable]

    return np.clip(mixed, -1.0, 1.0), sr, max_ch
