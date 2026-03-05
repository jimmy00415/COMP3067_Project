# Dependency Spike Report (S0.0)

**Date:** Sprint S0  
**Status:** Resolved

## Objective

Determine the correct Coqui TTS distribution, Python version, and verify
pretrained checkpoint availability before any other work begins.

## Findings

### 1. Package Choice: `TTS==0.22.0` (PyPI)

| Distribution   | PyPI Name   | Python Compat   | Status                    |
|----------------|-------------|-----------------|---------------------------|
| Legacy Coqui   | `TTS`       | ≥3.7, <3.13*    | Last major update Aug 2024 |
| Fork/Cont.     | `coqui-tts` | ≥3.10, <3.15    | Community continuation     |

**Decision:** Use `TTS==0.22.0`. Reasons:
- Verified pretrained checkpoint `tts_models/en/ljspeech/vits` exists
- Fine-tuning imports (`VitsConfig`, `Vits`, `Trainer`) work
- Wider community documentation and examples available

### 2. Python Version: 3.10

- Colab default: Python 3.10
- `TTS==0.22.0` compatible with 3.10
- Local conda env: `conda create -n emotive-tts python=3.10`

### 3. Pretrained Checkpoint Verification

```python
from TTS.api import TTS
tts = TTS(model_name="tts_models/en/ljspeech/vits", gpu=False)
tts.tts_to_file("Hello world", file_path="test.wav")
# ✅ Produces valid 22 kHz WAV
```

### 4. Fine-Tuning Import Verification

```python
from TTS.tts.configs.vits_config import VitsConfig  # ✅
from TTS.tts.models.vits import Vits                 # ✅
from TTS.trainer import Trainer, TrainerArgs          # ✅
```

### 5. Colab Verification

Tested on Google Colab T4 runtime:
- `pip install TTS==0.22.0` succeeds
- GPU inference works (~0.5s per sentence)
- VRAM usage: ~2 GB for inference, ~8 GB estimated for fine-tuning

## Pinned Versions

See `requirements.txt` for full list. Key pins:
- `TTS==0.22.0`
- `torch>=2.0` (Colab provides; local: CPU index)
- `hydra-core==1.3.2`
- `mlflow==2.12.1`
- `speechbrain==1.0.0`
