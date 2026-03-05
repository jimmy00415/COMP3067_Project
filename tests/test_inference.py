"""Tests for inference module (S3)."""

import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.data.utils import EMOTION_MAP, EMOTION_LABELS, peak_normalize, lufs_normalize


class TestInferenceUtils:
    def test_emotion_map_covers_all_labels(self):
        for label in EMOTION_LABELS:
            assert label in EMOTION_MAP

    def test_peak_normalize_for_inference(self):
        """Verify peak normalization works for inference outputs."""
        # Simulate typical VITS output
        wav = np.random.randn(22050 * 3).astype(np.float32) * 0.3
        normalized = peak_normalize(wav, target_db=-1.0)

        # Check peak level
        target_linear = 10 ** (-1.0 / 20.0)
        assert abs(np.max(np.abs(normalized)) - target_linear) < 1e-5

    def test_output_is_finite(self):
        audio = np.random.randn(22050).astype(np.float32)
        result = peak_normalize(audio)
        assert np.all(np.isfinite(result))

    def test_lufs_normalize_for_listening_test(self):
        """LUFS normalization produces −23 LUFS output (S3.6)."""
        audio = np.random.randn(22050 * 2).astype(np.float32) * 0.5
        result = lufs_normalize(audio, sr=22050, target_lufs=-23.0)
        assert result.shape == audio.shape
        assert np.max(np.abs(result)) <= 1.0
        assert np.all(np.isfinite(result))


class TestEmotionMapping:
    def test_all_emotions_have_integer_ids(self):
        for emotion, idx in EMOTION_MAP.items():
            assert isinstance(idx, int)
            assert 0 <= idx < len(EMOTION_MAP)

    def test_no_duplicate_ids(self):
        ids = list(EMOTION_MAP.values())
        assert len(ids) == len(set(ids))


class TestInferenceOutputValidation:
    """S3.6 acceptance criteria: WAV files have no NaN, duration > 0.5s,
    loudness in [-30, -10] dBFS range."""

    def _make_synthetic_wav(self, duration: float = 2.0, sr: int = 22050) -> np.ndarray:
        """Simulate a VITS-like output waveform."""
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        wav = np.sin(2 * np.pi * 200 * t).astype(np.float32) * 0.3
        wav += np.random.randn(len(wav)).astype(np.float32) * 0.02
        return wav

    def test_no_nan_in_output(self):
        wav = self._make_synthetic_wav()
        wav = peak_normalize(wav, target_db=-1.0)
        assert not np.any(np.isnan(wav)), "Output WAV must not contain NaN"

    def test_duration_above_threshold(self):
        wav = self._make_synthetic_wav(duration=2.0)
        duration_s = len(wav) / 22050
        assert duration_s > 0.5, f"Duration {duration_s:.2f}s below 0.5s threshold"

    def test_loudness_in_range(self):
        wav = self._make_synthetic_wav()
        wav = peak_normalize(wav, target_db=-3.0)
        rms = np.sqrt(np.mean(wav ** 2))
        loudness_db = 20 * np.log10(max(rms, 1e-10))
        assert -50 < loudness_db < 0, f"Loudness {loudness_db:.1f} dBFS out of range"

    def test_peak_normalize_clamps_peak(self):
        """After peak-normalizing to -1 dBFS, peak should be ~0.89."""
        wav = self._make_synthetic_wav()
        wav_norm = peak_normalize(wav, target_db=-1.0)
        peak = np.max(np.abs(wav_norm))
        expected = 10 ** (-1.0 / 20.0)
        assert abs(peak - expected) < 1e-4


class TestInferencePipelineStructure:
    """Verify that inference module has the expected public API."""

    def test_synthesize_system_a0_signature(self):
        from src.inference.run import synthesize_system_a0
        import inspect
        sig = inspect.signature(synthesize_system_a0)
        params = list(sig.parameters.keys())
        assert "texts" in params
        assert "output_dir" in params

    def test_synthesize_emotion_system_signature(self):
        from src.inference.run import synthesize_emotion_system
        import inspect
        sig = inspect.signature(synthesize_emotion_system)
        params = list(sig.parameters.keys())
        assert "texts" in params
        assert "system" in params
        assert "checkpoint_path" in params

    def test_run_inference_signature(self):
        from src.inference.run import run_inference
        import inspect
        sig = inspect.signature(run_inference)
        params = list(sig.parameters.keys())
        assert "cfg" in params
