"""Tests for prosody heads module."""

import torch
import pytest

from src.models.prosody_heads import (
    F0StatsHead,
    EnergyStatsHead,
    ProsodyHeads,
    build_prosody_heads,
)


class TestF0StatsHead:
    def test_output_shape(self):
        head = F0StatsHead(input_dim=192, output_dim=4)
        x = torch.randn(8, 192)
        out = head(x)
        assert out.shape == (8, 4)

    def test_output_dim(self):
        head = F0StatsHead(input_dim=192, output_dim=3)
        x = torch.randn(4, 192)
        out = head(x)
        assert out.shape == (4, 3)


class TestEnergyStatsHead:
    def test_output_shape(self):
        head = EnergyStatsHead(input_dim=192, output_dim=2)
        x = torch.randn(8, 192)
        out = head(x)
        assert out.shape == (8, 2)


class TestProsodyHeads:
    def test_forward(self):
        heads = ProsodyHeads(input_dim=192)
        x = torch.randn(4, 192)
        out = heads(x)
        assert "f0_stats" in out
        assert "energy_stats" in out
        assert out["f0_stats"].shape == (4, 4)
        assert out["energy_stats"].shape == (4, 2)

    def test_loss_computation(self):
        heads = ProsodyHeads(input_dim=192)
        x = torch.randn(4, 192)
        preds = heads(x)

        targets = {
            "f0_stats": torch.randn(4, 4),
            "energy_stats": torch.randn(4, 2),
        }
        loss = heads.compute_loss(preds, targets)
        assert loss.dim() == 0  # Scalar
        assert loss.item() >= 0

    def test_loss_decreases_with_perfect_prediction(self):
        heads = ProsodyHeads(input_dim=192)
        targets = {
            "f0_stats": torch.zeros(4, 4),
            "energy_stats": torch.zeros(4, 2),
        }
        preds_perfect = {
            "f0_stats": torch.zeros(4, 4),
            "energy_stats": torch.zeros(4, 2),
        }
        preds_bad = {
            "f0_stats": torch.ones(4, 4) * 10,
            "energy_stats": torch.ones(4, 2) * 10,
        }
        loss_perfect = heads.compute_loss(preds_perfect, targets)
        loss_bad = heads.compute_loss(preds_bad, targets)
        assert loss_perfect < loss_bad

    def test_parameter_count(self):
        heads = ProsodyHeads(input_dim=192, hidden_dim=128)
        params = heads.count_parameters()
        assert params["total"] > 0
        assert params["f0_head"] > 0
        assert params["energy_head"] > 0
        assert params["total"] == params["f0_head"] + params["energy_head"]

    def test_build_factory(self):
        heads = build_prosody_heads(input_dim=192)
        assert isinstance(heads, ProsodyHeads)


class TestGradientFlow:
    def test_gradients_flow(self):
        """Verify gradients flow through prosody heads."""
        heads = ProsodyHeads(input_dim=192)
        x = torch.randn(4, 192, requires_grad=True)

        preds = heads(x)
        targets = {
            "f0_stats": torch.randn(4, 4),
            "energy_stats": torch.randn(4, 2),
        }
        loss = heads.compute_loss(preds, targets)
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == (4, 192)
        # Check all parameters have gradients
        for name, param in heads.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
