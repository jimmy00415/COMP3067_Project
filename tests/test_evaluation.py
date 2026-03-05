"""Tests for evaluation modules (S6.10)."""

import numpy as np
import pandas as pd
import pytest
import tempfile
from pathlib import Path

from src.data.utils import EMOTION_LABELS
from src.evaluation.prosody import (
    analyze_single_file,
    compute_system_emotion_stats,
    test_emotion_differentiation,
    test_causal_attribution,
)


class TestAnalyzeSingleFile:
    def test_valid_audio(self, tmp_dir):
        """Test prosody extraction on a synthetic sine wave."""
        import soundfile as sf

        sr = 22050
        t = np.linspace(0, 2, sr * 2, endpoint=False)
        audio = np.sin(2 * np.pi * 200 * t).astype(np.float32) * 0.5
        path = tmp_dir / "test_sine.wav"
        sf.write(str(path), audio, sr)

        stats = analyze_single_file(str(path), sr=sr)
        assert stats["status"] == "ok"
        assert stats["f0_mean"] > 0
        assert stats["energy_mean"] > 0
        assert stats["duration"] > 0
        assert 0.0 <= stats["voiced_ratio"] <= 1.0

    def test_missing_file(self):
        """Gracefully handle missing file."""
        stats = analyze_single_file("nonexistent.wav")
        assert "error" in stats["status"]

    def test_silent_audio(self, tmp_dir):
        """Handle silent audio without crashing."""
        import soundfile as sf

        sr = 22050
        audio = np.zeros(sr, dtype=np.float32)
        path = tmp_dir / "silent.wav"
        sf.write(str(path), audio, sr)

        stats = analyze_single_file(str(path), sr=sr)
        assert stats["status"] == "ok"
        assert stats["duration"] > 0


class TestComputeSystemEmotionStats:
    def test_aggregate(self):
        """Test aggregation by system × emotion."""
        rows = []
        for sys in ["A0", "A", "B", "C"]:
            for emo in EMOTION_LABELS:
                for _ in range(5):
                    rows.append({
                        "system": sys,
                        "emotion": emo,
                        "f0_mean": np.random.uniform(100, 300),
                        "f0_std": np.random.uniform(10, 50),
                        "f0_range_high": np.random.uniform(200, 400),
                        "energy_mean": np.random.uniform(0.01, 0.2),
                        "energy_std": np.random.uniform(0.001, 0.05),
                        "duration": np.random.uniform(1, 5),
                    })
        df = pd.DataFrame(rows)
        agg = compute_system_emotion_stats(df)
        # 4 systems × 4 emotions = 16 rows
        assert len(agg) == 16
        assert "system" in agg.columns
        assert "emotion" in agg.columns


class TestEmotionDifferentiation:
    def test_significant_difference(self):
        """Kruskal-Wallis should detect clearly different groups."""
        rows = []
        for emo, base_f0 in [("neutral", 150), ("angry", 250), ("amused", 200), ("disgust", 180)]:
            for _ in range(20):
                rows.append({
                    "system": "B",
                    "emotion": emo,
                    "f0_mean": np.random.normal(base_f0, 10),
                })
        df = pd.DataFrame(rows)
        result = test_emotion_differentiation(df, metric="f0_mean", system="B")
        assert result["test"] == "kruskal_wallis"
        assert result["significant"] is True
        assert result["p_value"] < 0.05

    def test_no_difference(self):
        """Same distribution → not significant."""
        rows = []
        for emo in EMOTION_LABELS:
            for _ in range(20):
                rows.append({
                    "system": "A",
                    "emotion": emo,
                    "f0_mean": np.random.normal(180, 10),
                })
        df = pd.DataFrame(rows)
        result = test_emotion_differentiation(df, metric="f0_mean", system="A")
        assert result["test"] == "kruskal_wallis"
        # Not guaranteed but highly likely for same distribution
        # Don't assert significance — just check it runs


class TestCausalAttribution:
    def test_chain_comparison(self):
        """Test A0→A→B→C chain with synthetic data."""
        rows = []
        # Each system shifts f0 upward for angry
        for sys, shift in [("A0", 0), ("A", 10), ("B", 30), ("C", 50)]:
            for _ in range(15):
                rows.append({
                    "system": sys,
                    "emotion": "angry",
                    "f0_mean": np.random.normal(180 + shift, 5),
                })
        df = pd.DataFrame(rows)
        result = test_causal_attribution(df, metric="f0_mean", emotion="angry")
        assert result["metric"] == "f0_mean"
        assert result["emotion"] == "angry"
        assert len(result["comparisons"]) == 3  # A0→A, A→B, B→C

        for comp in result["comparisons"]:
            assert "comparison" in comp
            assert "p_value" in comp or "error" in comp


class TestPlots:
    """Smoke tests for plot functions — verify they don't crash."""

    def test_set_plot_style(self):
        from src.evaluation.plots import set_plot_style
        set_plot_style()  # Should not raise

    def test_plot_f0_by_system_emotion(self, tmp_dir):
        """Plot generation produces a file."""
        from src.evaluation.plots import plot_f0_by_system_emotion
        import matplotlib
        matplotlib.use("Agg")

        rows = []
        for sys in ["A0", "B"]:
            for emo in ["neutral", "angry"]:
                for _ in range(10):
                    rows.append({
                        "system": sys,
                        "emotion": emo,
                        "f0_mean": np.random.uniform(100, 300),
                    })
        df = pd.DataFrame(rows)
        out_path = tmp_dir / "f0_test.png"
        plot_f0_by_system_emotion(df, output_path=str(out_path))
        assert out_path.exists()
