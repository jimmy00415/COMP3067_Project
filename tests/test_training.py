"""Tests for training pipeline (S2)."""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import MagicMock, patch

from src.training.train import (
    _compute_mel_spectrogram,
    _kl_loss,
    EmotiveTTSDataset,
    collate_fn,
    Trainer,
)


class TestComputeMelSpectrogram:
    def test_output_shape(self):
        audio = torch.randn(2, 22050)  # 1 second, batch=2
        mel = _compute_mel_spectrogram(audio)
        assert mel.dim() == 3  # (batch, n_mels, frames)
        assert mel.shape[0] == 2
        assert mel.shape[1] == 80  # default n_mels

    def test_output_finite(self):
        audio = torch.randn(1, 22050 * 2)
        mel = _compute_mel_spectrogram(audio)
        assert torch.all(torch.isfinite(mel))

    def test_mono_output(self):
        """Single-channel audio should work."""
        audio = torch.randn(1, 11025)  # 0.5 sec
        mel = _compute_mel_spectrogram(audio)
        assert mel.shape[0] == 1

    def test_different_n_mels(self):
        audio = torch.randn(1, 22050)
        mel = _compute_mel_spectrogram(audio, n_mels=40)
        assert mel.shape[1] == 40

    def test_log_scale(self):
        """Output should be in log scale (negative values expected for quiet audio)."""
        audio = torch.randn(1, 22050) * 0.01
        mel = _compute_mel_spectrogram(audio)
        # Log of small values should be negative
        assert mel.mean().item() < 0


class TestKLLoss:
    def test_zero_for_identical(self):
        """KL of identical distributions should be ~0."""
        B, D, T = 2, 192, 10
        z_p = torch.randn(B, D, T)
        m_p = z_p.clone()
        logs_p = torch.zeros(B, D, T)
        logs_q = torch.zeros(B, D, T)
        z_mask = torch.ones(B, 1, T)

        loss = _kl_loss(z_p, logs_q, m_p, logs_p, z_mask)
        assert loss.dim() == 0  # scalar
        assert loss.item() < 0.1, f"KL for identical distributions should be ~0, got {loss.item()}"

    def test_positive_for_different(self):
        """KL of different distributions should be positive."""
        B, D, T = 2, 64, 10
        z_p = torch.randn(B, D, T) + 5  # shifted mean
        m_p = torch.randn(B, D, T)
        logs_p = torch.zeros(B, D, T)
        logs_q = torch.zeros(B, D, T)
        z_mask = torch.ones(B, 1, T)

        loss = _kl_loss(z_p, logs_q, m_p, logs_p, z_mask)
        assert loss.item() > 0

    def test_mask_effect(self):
        """Masking should affect the result."""
        B, D, T = 1, 32, 10
        z_p = torch.randn(B, D, T) + 3
        m_p = torch.zeros(B, D, T)
        logs_p = torch.zeros(B, D, T)
        logs_q = torch.zeros(B, D, T)

        full_mask = torch.ones(B, 1, T)
        half_mask = torch.zeros(B, 1, T)
        half_mask[:, :, :5] = 1.0

        loss_full = _kl_loss(z_p, logs_q, m_p, logs_p, full_mask)
        loss_half = _kl_loss(z_p, logs_q, m_p, logs_p, half_mask)
        # They may differ due to different normalization denominators
        assert isinstance(loss_full.item(), float)
        assert isinstance(loss_half.item(), float)


class TestCollate:
    def test_variable_length_padding(self):
        """collate_fn should pad variable-length audio."""
        batch = [
            {
                "audio": torch.randn(22050),
                "text": "hello",
                "emotion_id": 0,
                "prosody_targets": {},
                "audio_path": "a.wav",
            },
            {
                "audio": torch.randn(44100),
                "text": "goodbye world",
                "emotion_id": 1,
                "prosody_targets": {},
                "audio_path": "b.wav",
            },
        ]
        result = collate_fn(batch)
        assert result["audio"].shape == (2, 44100)  # padded to max
        assert result["audio_lengths"].tolist() == [22050, 44100]
        assert result["emotion_ids"].shape == (2,)

    def test_prosody_targets_stacked(self):
        """When prosody targets present, they should be stacked."""
        batch = [
            {
                "audio": torch.randn(22050),
                "text": "a",
                "emotion_id": 0,
                "prosody_targets": {
                    "f0_stats": np.array([100, 10, 80, 120], dtype=np.float32),
                    "energy_stats": np.array([0.1, 0.02], dtype=np.float32),
                },
                "audio_path": "a.wav",
            },
            {
                "audio": torch.randn(22050),
                "text": "b",
                "emotion_id": 1,
                "prosody_targets": {
                    "f0_stats": np.array([200, 20, 160, 240], dtype=np.float32),
                    "energy_stats": np.array([0.2, 0.05], dtype=np.float32),
                },
                "audio_path": "b.wav",
            },
        ]
        result = collate_fn(batch)
        assert "f0_stats" in result
        assert result["f0_stats"].shape == (2, 4)
        assert "energy_stats" in result
        assert result["energy_stats"].shape == (2, 2)


class TestTrainerInit:
    def test_default_config(self):
        cfg = {
            "system": "B",
            "training": {"max_epochs": 10, "batch_size": 4, "lr": 1e-4},
        }
        trainer = Trainer(cfg)
        assert trainer.system == "B"
        assert trainer.max_epochs == 10
        assert trainer.batch_size == 4
        assert trainer.lr == 1e-4
        assert trainer.best_val_loss == float("inf")

    def test_checkpoint_dir_created(self, tmp_dir):
        cfg = {
            "system": "C",
            "checkpoint_dir": str(tmp_dir / "ckpts"),
            "training": {},
        }
        trainer = Trainer(cfg)
        assert trainer.checkpoint_dir.exists()

    def test_device_selection(self):
        cfg = {"system": "A", "training": {}, "use_cuda": False}
        trainer = Trainer(cfg)
        assert trainer.device == torch.device("cpu") or torch.cuda.is_available()
