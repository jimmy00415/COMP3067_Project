# Emotive TTS — Emotion-Conditioned Neural Text-to-Speech

**COMP3065 Final Year Project**

Exploring emotion expressiveness in neural TTS via VITS fine-tuning with emotion conditioning and prosody auxiliary supervision.

## Overview

This project investigates whether adding explicit emotion conditioning and prosody supervision to a VITS text-to-speech model improves emotional expressiveness. Four system variants form a causal attribution chain:

| System | Description | Init From |
|--------|-------------|-----------|
| **A0** | Pretrained LJSpeech VITS (reference) | — |
| **A** | Domain-adapted on EmoV-DB (no emotion labels) | Pretrained |
| **B** | A + emotion embedding (`nn.Embedding(4, 192)`) | System A |
| **C** | B + utterance-level prosody auxiliary heads | System B |

Each system builds on the previous one, enabling clean causal attribution of each modification's effect on emotion expressiveness.

## Project Structure

```
├── configs/                 # Hydra YAML configs
│   ├── data.yaml           # Dataset & audio processing
│   ├── train_a.yaml        # System A training config
│   ├── train_b.yaml        # System B training config
│   ├── train_c.yaml        # System C training config
│   ├── train_debug.yaml    # Debug config (small, fast)
│   ├── infer.yaml          # Inference config
│   ├── eval.yaml           # Evaluation config
│   └── canary_texts.txt    # 16 standardized eval sentences
├── src/
│   ├── data/
│   │   ├── prepare.py      # EmoV-DB preprocessing pipeline
│   │   ├── qa.py           # Data quality assurance
│   │   └── utils.py        # Audio utilities (F0, energy, normalization)
│   ├── models/
│   │   ├── baseline.py     # System A0/A wrapper (Coqui Synthesizer)
│   │   ├── emotion_vits.py # EmotionVITS (Systems B/C)
│   │   └── prosody_heads.py # F0 & energy prediction heads (System C)
│   ├── training/
│   │   ├── train.py        # Training loop + Coqui trainer integration
│   │   └── callbacks.py    # MLflow, checkpointing, audio sampling
│   ├── inference/
│   │   └── run.py          # Batch inference for eval stimuli
│   └── evaluation/
│       ├── prosody.py      # F0/energy analysis (primary metric)
│       ├── ser_probe.py    # SER probe (auxiliary metric)
│       ├── listening_test.py # Stimulus pack generator
│       └── plots.py        # Publication-quality figures
├── demo/
│   └── app.py              # Gradio interactive demo
├── tests/                   # Unit tests
├── notebooks/
│   └── colab_pipeline.ipynb # Master Colab notebook (GPU compute)
├── data/                    # Data directory (gitignored)
├── checkpoints/             # Model checkpoints (gitignored)
├── figures/                 # Generated plots
├── tables/                  # Results tables
├── outputs/                 # Generated audio + stimulus packs
└── docs/                    # QA reports
```

## Quick Start

### Local Development (CPU)

```bash
# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Lint
ruff check src/ tests/ demo/
```

### GPU Training (Google Colab)

1. Upload project to Google Drive
2. Open `notebooks/colab_pipeline.ipynb` in Colab
3. Select **GPU runtime** (T4)
4. Run cells sequentially

The notebook handles: data prep → System A/B/C training → inference → evaluation.

### Gradio Demo

```bash
python demo/app.py
# Opens http://localhost:7860
```

## Evaluation

### Primary Metric: Prosody Analysis
F0 and energy statistics per system × emotion, with Kruskal-Wallis tests for emotion differentiation and Mann-Whitney U tests for causal attribution (A0→A→B→C).

### Auxiliary Metric: SER Proxy Agreement
SpeechBrain emotion classifier applied to generated audio. **Caveat:** label mismatch between IEMOCAP (neutral/angry/happy/sad) and our set (neutral/angry/amused/disgust). Disgust samples excluded from accuracy.

### Human Evaluation: Listening Test
P.808-inspired local test: naturalness + emotion match ratings on 1-5 scale using LUFS-normalized stimuli.

## Key Design Decisions (ADRs)

| ADR | Decision |
|-----|----------|
| ADR-1 | VITS backbone (Coqui TTS) — best quality-latency tradeoff |
| ADR-2 | `nn.Embedding(4, 192)` with additive injection |
| ADR-3 | Selective freezing: text_enc + posterior_enc + decoder frozen |
| ADR-4 | Utterance-level prosody aux heads (frame-level is stretch) |
| ADR-5 | EmoV-DB, single core speaker for consistency |
| ADR-6 | Prosody stats = primary metric; SER = auxiliary only |
| ADR-7 | Peak norm for training; LUFS for eval playback |
| ADR-8 | Local coding + Colab T4 for GPU compute |

## Requirements

- Python ≥ 3.10, < 3.13
- PyTorch ≥ 2.1
- Coqui TTS (TTS==0.22.0)
- See `requirements.txt` for full list

## License

This project is for academic use (COMP3065). See [NOTICE.md](NOTICE.md) for dependency licenses.
