"""Data utilities for audio processing and file I/O."""

import os
import json
import hashlib
from pathlib import Path
from typing import Optional

import numpy as np
import librosa
import soundfile as sf


# --- Emotion label mapping ---
EMOTION_MAP = {"neutral": 0, "angry": 1, "amused": 2, "disgust": 3}
EMOTION_LABELS = list(EMOTION_MAP.keys())
NUM_EMOTIONS = len(EMOTION_MAP)


def load_audio(
    path: str | Path,
    sr: int = 22050,
) -> tuple[np.ndarray, int]:
    """Load audio file and resample to target sample rate.

    Args:
        path: Path to audio file.
        sr: Target sample rate.

    Returns:
        Tuple of (audio array, sample rate).
    """
    audio, orig_sr = librosa.load(str(path), sr=sr, mono=True)
    return audio, sr


def save_audio(
    audio: np.ndarray,
    path: str | Path,
    sr: int = 22050,
) -> None:
    """Save audio array to WAV file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr)


def peak_normalize(audio: np.ndarray, target_db: float = -1.0) -> np.ndarray:
    """Conservative peak normalization (ADR-7).

    Normalizes peak amplitude to target_db. Does NOT apply LUFS loudness
    matching — energy variation across emotions is preserved as signal.

    Args:
        audio: Audio array.
        target_db: Target peak level in dBFS (default: -1.0).

    Returns:
        Normalized audio array.
    """
    peak = np.max(np.abs(audio))
    if peak < 1e-8:
        return audio
    target_linear = 10 ** (target_db / 20.0)
    return audio * (target_linear / peak)


def lufs_normalize(audio: np.ndarray, sr: int, target_lufs: float = -23.0) -> np.ndarray:
    """EBU R128 LUFS normalization for evaluation playback copies only.

    Used ONLY for human eval stimuli — ensures listener fairness.
    Training audio should use peak_normalize() instead.

    Attempts to use ``pyloudnorm`` for accurate K-weighted measurement;
    falls back to an RMS-based approximation when the library is absent.

    Args:
        audio: Audio array.
        sr: Sample rate.
        target_lufs: Target loudness in LUFS (default: -23.0, EBU R128).

    Returns:
        LUFS-normalized audio array.
    """
    if np.max(np.abs(audio)) < 1e-8:
        return audio

    try:
        import pyloudnorm as pyln
        meter = pyln.Meter(sr)
        current_lufs = meter.integrated_loudness(audio)
        if np.isinf(current_lufs):
            return audio
        normalized = pyln.normalize.loudness(audio, current_lufs, target_lufs)
    except ImportError:
        # Fallback: RMS-based approximation
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < 1e-8:
            return audio
        current_lufs_approx = 20 * np.log10(rms) - 0.691  # K-weighted approx
        gain_db = target_lufs - current_lufs_approx
        gain_linear = 10 ** (gain_db / 20.0)
        normalized = audio * gain_linear

    # Clip protection
    peak = np.max(np.abs(normalized))
    if peak > 1.0:
        normalized = normalized / peak * 0.99
    return normalized


def extract_f0(
    audio: np.ndarray,
    sr: int = 22050,
    fmin: float = 75.0,
    fmax: float = 300.0,
    hop_length: int = 256,
    win_length: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract F0 contour using pYIN (via librosa).

    Speaker-aware fmin/fmax should be configured based on core speaker's
    vocal range (S1.1 audit). Masks unvoiced frames.

    Args:
        audio: Audio array.
        sr: Sample rate.
        fmin: Minimum expected F0 (Hz). Male: 75, Female: 100.
        fmax: Maximum expected F0 (Hz). Male: 300, Female: 500.
        hop_length: Hop length for analysis.
        win_length: Window length for analysis.

    Returns:
        Tuple of (f0_contour, voiced_flag).
        f0_contour: F0 values in Hz (NaN for unvoiced frames).
        voiced_flag: Boolean array (True = voiced).
    """
    f0, voiced_flag, _ = librosa.pyin(
        audio,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        hop_length=hop_length,
        win_length=win_length,
    )
    return f0, voiced_flag


def extract_energy(
    audio: np.ndarray,
    hop_length: int = 256,
    win_length: int = 1024,
) -> np.ndarray:
    """Extract frame-level RMS energy.

    Args:
        audio: Audio array.
        hop_length: Hop length.
        win_length: Window length.

    Returns:
        RMS energy contour (1D array, one value per frame).
    """
    energy = librosa.feature.rms(
        y=audio,
        frame_length=win_length,
        hop_length=hop_length,
    )[0]
    return energy


def compute_utterance_prosody_stats(
    f0: np.ndarray,
    energy: np.ndarray,
) -> dict[str, float]:
    """Compute utterance-level prosody statistics.

    These serve as targets for System C (utterance-level auxiliary supervision).
    See ADR-4.

    Args:
        f0: F0 contour (NaN for unvoiced frames).
        energy: RMS energy contour.

    Returns:
        Dict with keys: f0_mean, f0_std, f0_range_low, f0_range_high,
                        energy_mean, energy_std.
    """
    # F0 stats — voiced frames only
    voiced_f0 = f0[~np.isnan(f0)]
    if len(voiced_f0) == 0:
        f0_stats = {
            "f0_mean": 0.0,
            "f0_std": 0.0,
            "f0_range_low": 0.0,
            "f0_range_high": 0.0,
        }
    else:
        f0_stats = {
            "f0_mean": float(np.mean(voiced_f0)),
            "f0_std": float(np.std(voiced_f0)),
            "f0_range_low": float(np.percentile(voiced_f0, 5)),
            "f0_range_high": float(np.percentile(voiced_f0, 95)),
        }

    # Energy stats
    energy_stats = {
        "energy_mean": float(np.mean(energy)),
        "energy_std": float(np.std(energy)),
    }

    return {**f0_stats, **energy_stats}


def compute_speaking_rate(
    audio: np.ndarray,
    sr: int = 22050,
    text: str = "",
) -> float:
    """Estimate speaking rate as syllables per second (approximate).

    Falls back to duration-based estimate if text is empty.
    """
    duration = len(audio) / sr
    if duration < 0.1:
        return 0.0
    if text:
        # Rough syllable count: count vowel groups
        import re
        vowels = re.findall(r'[aeiouy]+', text.lower())
        syllables = max(1, len(vowels))
        return syllables / duration
    return 0.0  # Cannot estimate without text


def file_hash(path: str | Path) -> str:
    """Compute SHA-256 hash of a file for eval freeze manifest."""
    h = hashlib.sha256()
    with open(str(path), "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_canary_texts(path: str | Path = "configs/canary_texts.txt") -> list[dict]:
    """Load canary text set from config file.

    Returns:
        List of dicts with keys: id, text.
    """
    texts = []
    with open(str(path), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("|", 1)
            if len(parts) == 2:
                texts.append({"id": int(parts[0]), "text": parts[1].strip()})
    return texts
