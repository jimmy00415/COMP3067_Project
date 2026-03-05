"""Shared test configuration and fixtures."""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_audio():
    """Generate a short random audio signal (1 second, 22050 Hz)."""
    sr = 22050
    audio = np.random.randn(sr).astype(np.float32) * 0.3
    return audio, sr


@pytest.fixture
def sine_wave():
    """Generate a 200 Hz sine wave (2 seconds, 22050 Hz)."""
    sr = 22050
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = np.sin(2 * np.pi * 200 * t).astype(np.float32) * 0.5
    return audio, sr


@pytest.fixture
def tmp_dir():
    """Provide a temporary directory that is cleaned up after the test."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def device():
    """Return the best available torch device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def canary_texts_path():
    """Path to the canary texts file."""
    p = CONFIGS_DIR / "canary_texts.txt"
    if not p.exists():
        pytest.skip("canary_texts.txt not found")
    return p
