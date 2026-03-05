"""Tests for data pipeline (S1)."""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.data.utils import (
    EMOTION_MAP,
    EMOTION_LABELS,
    peak_normalize,
    lufs_normalize,
    extract_f0,
    extract_energy,
    compute_utterance_prosody_stats,
    load_canary_texts,
    file_hash,
)


class TestEmotionMap:
    def test_four_emotions(self):
        assert len(EMOTION_MAP) == 4

    def test_emotion_labels(self):
        assert set(EMOTION_LABELS) == {"neutral", "angry", "amused", "disgust"}

    def test_emotion_ids_unique(self):
        assert len(set(EMOTION_MAP.values())) == 4

    def test_emotion_ids_sequential(self):
        assert sorted(EMOTION_MAP.values()) == [0, 1, 2, 3]


class TestPeakNormalize:
    def test_normalizes_to_target(self):
        audio = np.random.randn(22050).astype(np.float32) * 0.1
        normalized = peak_normalize(audio, target_db=-1.0)
        peak = np.max(np.abs(normalized))
        expected = 10 ** (-1.0 / 20.0)
        assert abs(peak - expected) < 1e-5

    def test_silent_audio_unchanged(self):
        audio = np.zeros(1000)
        result = peak_normalize(audio)
        assert np.allclose(result, audio)

    def test_preserves_shape(self):
        audio = np.random.randn(44100).astype(np.float32)
        result = peak_normalize(audio)
        assert result.shape == audio.shape

    def test_different_targets(self):
        audio = np.random.randn(22050).astype(np.float32)
        norm_1 = peak_normalize(audio, target_db=-1.0)
        norm_3 = peak_normalize(audio, target_db=-3.0)
        assert np.max(np.abs(norm_1)) > np.max(np.abs(norm_3))


class TestLufsNormalize:
    def test_output_shape(self):
        audio = np.random.randn(22050).astype(np.float32) * 0.1
        result = lufs_normalize(audio, sr=22050)
        assert result.shape == audio.shape

    def test_no_clipping(self):
        audio = np.random.randn(22050).astype(np.float32)
        result = lufs_normalize(audio, sr=22050)
        assert np.max(np.abs(result)) <= 1.0

    def test_silent_unchanged(self):
        audio = np.zeros(1000)
        result = lufs_normalize(audio, sr=22050)
        assert np.allclose(result, audio)


class TestExtractF0:
    def test_output_shape(self):
        # Generate a simple sine wave
        sr = 22050
        t = np.linspace(0, 1, sr)
        audio = np.sin(2 * np.pi * 200 * t).astype(np.float32)

        f0, voiced = extract_f0(audio, sr=sr)
        assert len(f0) == len(voiced)
        assert len(f0) > 0

    def test_sine_wave_f0(self):
        sr = 22050
        freq = 200.0
        t = np.linspace(0, 2, sr * 2)
        audio = np.sin(2 * np.pi * freq * t).astype(np.float32) * 0.5

        f0, voiced = extract_f0(audio, sr=sr, fmin=75, fmax=300)
        voiced_f0 = f0[~np.isnan(f0)]
        if len(voiced_f0) > 0:
            mean_f0 = np.mean(voiced_f0)
            # Should be roughly 200 Hz (±20%)
            assert 160 < mean_f0 < 240, f"Expected ~200 Hz, got {mean_f0:.1f}"

    def test_unvoiced_regions(self):
        audio = np.zeros(22050, dtype=np.float32)
        f0, voiced = extract_f0(audio, sr=22050)
        # Silent audio should be mostly unvoiced
        assert np.sum(voiced) < len(voiced) * 0.5


class TestExtractEnergy:
    def test_output_shape(self):
        audio = np.random.randn(22050).astype(np.float32)
        energy = extract_energy(audio)
        assert len(energy) > 0

    def test_loud_vs_quiet(self):
        loud = np.random.randn(22050).astype(np.float32) * 1.0
        quiet = np.random.randn(22050).astype(np.float32) * 0.01

        e_loud = extract_energy(loud)
        e_quiet = extract_energy(quiet)
        assert np.mean(e_loud) > np.mean(e_quiet)


class TestProsodyStats:
    def test_keys(self):
        f0 = np.array([200, 210, 195, np.nan, 205])
        energy = np.array([0.1, 0.15, 0.12, 0.08, 0.11])

        stats = compute_utterance_prosody_stats(f0, energy)
        expected_keys = {"f0_mean", "f0_std", "f0_range_low", "f0_range_high",
                         "energy_mean", "energy_std"}
        assert expected_keys == set(stats.keys())

    def test_all_nan_f0(self):
        f0 = np.array([np.nan, np.nan, np.nan])
        energy = np.array([0.1, 0.1, 0.1])
        stats = compute_utterance_prosody_stats(f0, energy)
        assert stats["f0_mean"] == 0.0

    def test_values_reasonable(self):
        f0 = np.array([100, 120, 110, 130, 105])
        energy = np.array([0.1, 0.2, 0.15, 0.12, 0.18])
        stats = compute_utterance_prosody_stats(f0, energy)
        assert 100 <= stats["f0_mean"] <= 130
        assert stats["energy_mean"] > 0


class TestCanaryTexts:
    def test_load(self):
        canary_path = Path("configs/canary_texts.txt")
        if canary_path.exists():
            texts = load_canary_texts(canary_path)
            assert len(texts) == 16
            assert all("id" in t and "text" in t for t in texts)

    def test_all_have_text(self):
        canary_path = Path("configs/canary_texts.txt")
        if canary_path.exists():
            texts = load_canary_texts(canary_path)
            assert all(len(t["text"]) > 0 for t in texts)


class TestFileHash:
    def test_deterministic(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"test content")
            f.flush()
            h1 = file_hash(f.name)
            h2 = file_hash(f.name)
            assert h1 == h2
            os.unlink(f.name)

    def test_different_content(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f1:
            f1.write(b"content A")
            f1.flush()
            with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f2:
                f2.write(b"content B")
                f2.flush()
                assert file_hash(f1.name) != file_hash(f2.name)
                os.unlink(f1.name)
                os.unlink(f2.name)
