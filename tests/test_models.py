"""Tests for model modules (EmotionVITS, ProsodyHeads, baseline)."""

import pytest
import torch
import torch.nn as nn
import numpy as np

from src.models.prosody_heads import (
    F0StatsHead,
    EnergyStatsHead,
    ProsodyHeads,
    build_prosody_heads,
)
from src.models.emotion_vits import EmotionEmbedding, EmotionVITS


# ---------------------------------------------------------------------------
# ProsodyHeads tests
# ---------------------------------------------------------------------------

class TestF0StatsHead:
    def test_output_shape(self):
        head = F0StatsHead(input_dim=192, hidden_dim=128, output_dim=4)
        x = torch.randn(2, 192)
        out = head(x)
        assert out.shape == (2, 4)

    def test_deterministic(self):
        head = F0StatsHead()
        head.eval()
        x = torch.randn(1, 192)
        out1 = head(x)
        out2 = head(x)
        assert torch.allclose(out1, out2)


class TestEnergyStatsHead:
    def test_output_shape(self):
        head = EnergyStatsHead(input_dim=192, hidden_dim=128, output_dim=2)
        x = torch.randn(3, 192)
        out = head(x)
        assert out.shape == (3, 2)


class TestProsodyHeads:
    def test_forward_returns_dict(self):
        heads = ProsodyHeads(input_dim=192)
        x = torch.randn(2, 192)
        out = heads(x)
        assert "f0_stats" in out
        assert "energy_stats" in out
        assert out["f0_stats"].shape == (2, 4)
        assert out["energy_stats"].shape == (2, 2)

    def test_compute_loss(self):
        heads = ProsodyHeads(input_dim=192)
        x = torch.randn(2, 192)
        preds = heads(x)
        targets = {
            "f0_stats": torch.randn(2, 4),
            "energy_stats": torch.randn(2, 2),
        }
        loss = heads.compute_loss(preds, targets)
        assert loss.dim() == 0  # scalar
        assert loss.item() > 0

    def test_count_parameters(self):
        heads = ProsodyHeads(input_dim=192, hidden_dim=128)
        params = heads.count_parameters()
        assert params["total"] > 0
        assert params["f0_head"] > 0
        assert params["energy_head"] > 0
        assert params["total"] == params["f0_head"] + params["energy_head"]

    def test_build_factory(self):
        heads = build_prosody_heads(input_dim=64, hidden_dim=32)
        x = torch.randn(1, 64)
        out = heads(x)
        assert out["f0_stats"].shape == (1, 4)


# ---------------------------------------------------------------------------
# EmotionEmbedding tests
# ---------------------------------------------------------------------------

class TestEmotionEmbedding:
    def test_output_shape(self):
        emb = EmotionEmbedding(num_emotions=4, embedding_dim=192)
        ids = torch.LongTensor([0, 1, 2, 3])
        out = emb(ids)
        assert out.shape == (4, 192)

    def test_different_emotions_different_embeddings(self):
        emb = EmotionEmbedding(num_emotions=4, embedding_dim=192)
        out = emb(torch.LongTensor([0, 1]))
        # After init the embeddings should be different (initialized with random normal)
        assert not torch.allclose(out[0], out[1])

    def test_init_small_magnitude(self):
        """ADR-2: embeddings initialized with small values to minimize initial impact."""
        emb = EmotionEmbedding(num_emotions=4, embedding_dim=192)
        weight_std = emb.embedding.weight.data.std().item()
        assert weight_std < 0.05, f"Init std {weight_std} too large"


# ---------------------------------------------------------------------------
# EmotionVITS tests (with mock VITS backbone)
# ---------------------------------------------------------------------------

class _MockTextEncoder(nn.Module):
    """Mock text encoder that returns known shapes."""
    def __init__(self, hidden_dim: int = 192):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, x, x_lengths):
        batch, seq_len = x.shape
        x_encoded = torch.randn(batch, self.hidden_dim, seq_len)
        m_p = torch.randn(batch, self.hidden_dim, seq_len)
        logs_p = torch.randn(batch, self.hidden_dim, seq_len)
        x_mask = torch.ones(batch, 1, seq_len)
        return x_encoded, m_p, logs_p, x_mask


class _MockPosteriorEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 192):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, y, y_lengths):
        batch, _, mel_len = y.shape
        z = torch.randn(batch, self.hidden_dim, mel_len)
        m_q = torch.randn(batch, self.hidden_dim, mel_len)
        logs_q = torch.randn(batch, self.hidden_dim, mel_len)
        y_mask = torch.ones(batch, 1, mel_len)
        return z, m_q, logs_q, y_mask


class _MockFlow(nn.Module):
    def forward(self, z, mask, reverse=False):
        return z


class _MockDecoder(nn.Module):
    def forward(self, z):
        batch, hidden, frames = z.shape
        # Produce audio-like output (batch, 1, samples)
        return torch.randn(batch, 1, frames * 256)


class _MockVITS(nn.Module):
    """Minimal mock VITS model with the sub-modules EmotionVITS expects."""
    def __init__(self, hidden_dim: int = 192):
        super().__init__()
        self.enc_p = _MockTextEncoder(hidden_dim)
        self.enc_q = _MockPosteriorEncoder(hidden_dim)
        self.flow = _MockFlow()
        self.dec = _MockDecoder()
        self.dp = nn.Linear(hidden_dim, 1)  # simple stand-in for duration predictor


class TestEmotionVITSForward:
    @pytest.fixture
    def model_b(self):
        vits = _MockVITS(hidden_dim=192)
        return EmotionVITS(
            vits_model=vits,
            use_emotion=True,
            num_emotions=4,
            embedding_dim=192,
        )

    @pytest.fixture
    def model_c(self):
        vits = _MockVITS(hidden_dim=192)
        heads = build_prosody_heads(input_dim=192, hidden_dim=64)
        return EmotionVITS(
            vits_model=vits,
            use_emotion=True,
            num_emotions=4,
            embedding_dim=192,
            use_prosody_heads=True,
            prosody_heads=heads,
        )

    def test_forward_system_b(self, model_b):
        x = torch.LongTensor([[1, 2, 3, 4, 5]])
        x_lengths = torch.LongTensor([5])
        y = torch.randn(1, 80, 20)
        y_lengths = torch.LongTensor([20])
        eid = torch.LongTensor([1])

        outputs = model_b(x, x_lengths, y, y_lengths, emotion_ids=eid)
        assert "model_outputs" in outputs
        assert "z_p" in outputs
        assert "m_p" in outputs

    def test_forward_system_c_with_prosody(self, model_c):
        x = torch.LongTensor([[1, 2, 3]])
        x_lengths = torch.LongTensor([3])
        y = torch.randn(1, 80, 10)
        y_lengths = torch.LongTensor([10])
        eid = torch.LongTensor([2])
        targets = {
            "f0_stats": torch.randn(1, 4),
            "energy_stats": torch.randn(1, 2),
        }

        outputs = model_c(x, x_lengths, y, y_lengths,
                          emotion_ids=eid, prosody_targets=targets)
        assert "prosody_preds" in outputs
        assert "prosody_loss" in outputs
        assert outputs["prosody_loss"].item() > 0

    def test_inject_emotion_additive(self, model_b):
        hidden = torch.randn(2, 192, 10)
        eid = torch.LongTensor([0, 3])
        result = model_b.inject_emotion(hidden, eid)
        assert result.shape == hidden.shape
        # Should differ from input (emotion added)
        assert not torch.allclose(result, hidden)

    def test_no_emotion_passthrough(self):
        vits = _MockVITS()
        model = EmotionVITS(vits_model=vits, use_emotion=False)
        hidden = torch.randn(1, 192, 5)
        eid = torch.LongTensor([0])
        result = model.inject_emotion(hidden, eid)
        assert torch.allclose(result, hidden)

    def test_count_parameters(self, model_b):
        params = model_b.count_parameters()
        assert params["total"] > 0
        assert params["trainable"] >= 0


class TestFreezeStrategies:
    def test_freeze_for_system_a(self):
        vits = _MockVITS()
        model = EmotionVITS(vits_model=vits, use_emotion=False)
        model.freeze_for_system_a()

        # text encoder should be frozen
        for p in model.vits.enc_p.parameters():
            assert not p.requires_grad, "Text encoder should be frozen"

        # duration predictor should be unfrozen
        for p in model.vits.dp.parameters():
            assert p.requires_grad, "Duration predictor should be trainable"

    def test_freeze_for_system_b(self):
        vits = _MockVITS()
        model = EmotionVITS(vits_model=vits, use_emotion=True)
        model.freeze_for_system_b()

        # emotion embedding should be unfrozen
        for p in model.emotion_embedding.parameters():
            assert p.requires_grad, "Emotion embedding should be trainable"

    def test_freeze_for_system_c(self):
        vits = _MockVITS()
        heads = build_prosody_heads(input_dim=192, hidden_dim=64)
        model = EmotionVITS(vits_model=vits, use_emotion=True,
                            use_prosody_heads=True, prosody_heads=heads)
        model.freeze_for_system_c()

        # prosody heads should be unfrozen
        for p in model.prosody_heads.parameters():
            assert p.requires_grad, "Prosody heads should be trainable"
