"""Tests for audio I/O and processing utilities."""

import tempfile
import os

import numpy as np
import pytest

from src.data.utils import load_audio, save_audio, peak_normalize


class TestAudioIO:
    def test_save_and_load_roundtrip(self):
        """Test that save→load preserves audio content."""
        sr = 22050
        audio = np.random.randn(sr * 2).astype(np.float32) * 0.5

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            save_audio(audio, f.name, sr=sr)
            loaded, loaded_sr = load_audio(f.name, sr=sr)

            assert loaded_sr == sr
            assert len(loaded) == len(audio)
            # Allow small floating-point differences from WAV encoding
            assert np.allclose(audio, loaded, atol=1e-4)
            os.unlink(f.name)

    def test_resample(self):
        """Test resampling during load."""
        sr_original = 44100
        sr_target = 22050
        audio = np.random.randn(sr_original).astype(np.float32) * 0.3

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            save_audio(audio, f.name, sr=sr_original)
            loaded, loaded_sr = load_audio(f.name, sr=sr_target)

            assert loaded_sr == sr_target
            # Duration should be roughly preserved
            original_duration = len(audio) / sr_original
            loaded_duration = len(loaded) / loaded_sr
            assert abs(original_duration - loaded_duration) < 0.1
            os.unlink(f.name)

    def test_creates_directory(self):
        """Test that save_audio creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = os.path.join(tmpdir, "a", "b", "c", "test.wav")
            audio = np.random.randn(1000).astype(np.float32)
            save_audio(audio, nested_path, sr=22050)
            assert os.path.exists(nested_path)

    def test_mono_output(self):
        """Test that loaded audio is always mono."""
        sr = 22050
        audio = np.random.randn(sr).astype(np.float32) * 0.3

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            save_audio(audio, f.name, sr=sr)
            loaded, _ = load_audio(f.name, sr=sr)
            assert loaded.ndim == 1
            os.unlink(f.name)


class TestPeakNormalization:
    def test_idempotent(self):
        """Normalizing twice should give the same result."""
        audio = np.random.randn(22050).astype(np.float32) * 0.3
        norm1 = peak_normalize(audio, target_db=-1.0)
        norm2 = peak_normalize(norm1, target_db=-1.0)
        assert np.allclose(norm1, norm2, atol=1e-6)

    def test_preserves_waveform_shape(self):
        """Normalization should only scale, not change shape."""
        audio = np.random.randn(22050).astype(np.float32) * 0.1
        result = peak_normalize(audio)
        # The ratio between any two samples should be preserved
        # (i.e., it's a linear scaling)
        if np.abs(audio[0]) > 1e-8 and np.abs(audio[1]) > 1e-8:
            ratio_orig = audio[0] / audio[1]
            ratio_norm = result[0] / result[1]
            assert abs(ratio_orig - ratio_norm) < 1e-5
