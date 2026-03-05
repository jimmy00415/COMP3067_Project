"""Utterance-level prosody prediction heads for System C (ADR-4).

Architecture:
- F0StatsHead: 2-layer MLP (hidden_dim → 128 → 4)
  Output: [f0_mean, f0_std, f0_range_low, f0_range_high]

- EnergyStatsHead: 2-layer MLP (hidden_dim → 128 → 2)
  Output: [energy_mean, energy_std]

Loss: L1 (MAE) against ground-truth utterance-level stats extracted from
the training audio. Loss weight λ = 0.1 (configurable).

Design rationale (ADR-4):
- Utterance-level (C1) is default: simpler, more stable, sufficient for
  demonstrating that prosody supervision improves emotion expressiveness.
- Frame-level (C2) is a stretch goal (not implemented here).
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class F0StatsHead(nn.Module):
    """Predict utterance-level F0 statistics.

    Input: Pooled encoder representation (batch, hidden_dim)
    Output: (batch, 4) = [f0_mean, f0_std, f0_range_low, f0_range_high]
    """

    def __init__(self, input_dim: int = 192, hidden_dim: int = 128, output_dim: int = 4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Pooled encoder output, shape (batch, input_dim).

        Returns:
            F0 stats predictions, shape (batch, 4).
        """
        return self.mlp(x)


class EnergyStatsHead(nn.Module):
    """Predict utterance-level energy statistics.

    Input: Pooled encoder representation (batch, hidden_dim)
    Output: (batch, 2) = [energy_mean, energy_std]
    """

    def __init__(self, input_dim: int = 192, hidden_dim: int = 128, output_dim: int = 2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Pooled encoder output, shape (batch, input_dim).

        Returns:
            Energy stats predictions, shape (batch, 2).
        """
        return self.mlp(x)


class ProsodyHeads(nn.Module):
    """Combined prosody prediction module (F0 + Energy heads).

    Used in System C as auxiliary supervision. Attached to the
    pooled text encoder output in EmotionVITS.
    """

    def __init__(
        self,
        input_dim: int = 192,
        hidden_dim: int = 128,
        f0_output_dim: int = 4,
        energy_output_dim: int = 2,
    ):
        super().__init__()
        self.f0_head = F0StatsHead(input_dim, hidden_dim, f0_output_dim)
        self.energy_head = EnergyStatsHead(input_dim, hidden_dim, energy_output_dim)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass through both heads.

        Args:
            x: Pooled encoder output, shape (batch, input_dim).

        Returns:
            Dict with 'f0_stats' (batch, 4) and 'energy_stats' (batch, 2).
        """
        return {
            "f0_stats": self.f0_head(x),
            "energy_stats": self.energy_head(x),
        }

    def compute_loss(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute combined L1 loss for prosody prediction.

        Args:
            predictions: Dict from forward() with 'f0_stats', 'energy_stats'.
            targets: Dict with same keys containing ground-truth values.

        Returns:
            Scalar loss tensor.
        """
        loss = torch.tensor(0.0, device=predictions["f0_stats"].device)

        if "f0_stats" in targets and targets["f0_stats"] is not None:
            f0_loss = nn.functional.l1_loss(
                predictions["f0_stats"], targets["f0_stats"]
            )
            loss = loss + f0_loss

        if "energy_stats" in targets and targets["energy_stats"] is not None:
            energy_loss = nn.functional.l1_loss(
                predictions["energy_stats"], targets["energy_stats"]
            )
            loss = loss + energy_loss

        return loss

    def count_parameters(self) -> dict:
        """Prosody heads parameter count."""
        f0_params = sum(p.numel() for p in self.f0_head.parameters())
        energy_params = sum(p.numel() for p in self.energy_head.parameters())
        total = f0_params + energy_params
        return {
            "f0_head": f0_params,
            "energy_head": energy_params,
            "total": total,
        }


def build_prosody_heads(
    input_dim: int = 192,
    hidden_dim: int = 128,
    f0_output_dim: int = 4,
    energy_output_dim: int = 2,
) -> ProsodyHeads:
    """Factory function for ProsodyHeads.

    Args:
        input_dim: Dimension of pooled encoder output (must match VITS hidden dim).
        hidden_dim: MLP hidden layer dimension.
        f0_output_dim: Number of F0 statistics to predict.
        energy_output_dim: Number of energy statistics to predict.

    Returns:
        Initialized ProsodyHeads module.
    """
    heads = ProsodyHeads(input_dim, hidden_dim, f0_output_dim, energy_output_dim)
    params = heads.count_parameters()
    logger.info(f"ProsodyHeads: {params['total']:,} parameters "
                f"(F0: {params['f0_head']:,}, Energy: {params['energy_head']:,})")
    return heads
