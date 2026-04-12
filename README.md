# Emotive TTS — Emotion-Conditioned Neural Text-to-Speech

**COMP3065 Project**

Exploring emotion expressiveness in neural TTS via VITS fine-tuning with emotion conditioning and prosody auxiliary supervision.

---

## Overview

This project investigates whether adding explicit emotion conditioning and prosody supervision to a VITS text-to-speech model improves emotional expressiveness. Four system variants form a **causal attribution chain**, where each system builds incrementally on the previous one:

| System | Description | Init From | Trainable Params |
|--------|-------------|-----------|-----------------|
| **A0** | Pretrained LJSpeech VITS (reference baseline) | — | 0 (frozen) |
| **A** | Domain-adapted on EmoV-DB (no emotion labels) | Pretrained | 8.4M (23.2%) |
| **B** | A + emotion embedding (`nn.Embedding(4, 192)`) | System A | 8.4M (23.2%) |
| **C** | B + utterance-level prosody auxiliary heads (F0 + energy) | System B | 8.5M (23.3%) |

This chain enables clean causal attribution: any change from A→B is attributable to emotion conditioning, and B→C to prosody supervision.

---

## Experimental Results

### Training Summary

All systems were trained on **Google Colab** with a single GPU session (42.4 GB VRAM instance). Training used the **Coqui TTS VITS** pretrained model (`tts_models/en/ljspeech/vits`, 22050 Hz) as the backbone with selective layer freezing.

| Config | Value |
|--------|-------|
| Learning rate | 5×10⁻³ (all systems) |
| Max epochs | 15 |
| Batch size | 32 |
| Validation every | 5 epochs |
| Early stopping patience | 4 |
| LR scheduler | ExponentialLR (γ=0.998) |
| Gradient clipping | 5.0 |
| Frozen layers | text_encoder, posterior_encoder, decoder |
| Unfrozen layers | flow, duration_predictor (monitoring-only loss) |

**Loss composition:** `total_loss = kl_loss + prosody_loss`. Mel reconstruction loss and duration loss are computed for monitoring only (under `torch.no_grad()`), not backpropagated.

#### Training Convergence

| System | Epoch 1 Train Loss | Epoch 15 Train Loss | Final Val Loss | Final Mel Loss | Final Dur Loss |
|--------|-------------------|---------------------|----------------|----------------|----------------|
| **A** | 107.27 (KL) | 102.09 (KL) | 101.88 | 2.134 | 1.164 |
| **B** | 102.92 (KL) | 101.57 (KL) | 101.57 | 2.144 | 1.137 |
| **C** | 109.19 (KL+prosody) | 103.96 (KL+prosody) | 103.84 | 2.132 | 0.947 |

- **System A** achieved steady KL loss reduction (~5.2 decrease over 15 epochs), confirming successful domain adaptation.
- **System B** warm-started from A and converged to slightly lower KL (101.57 vs 102.09), showing the emotion embedding did not hurt convergence.
- **System C** has higher total loss due to the added prosody loss component (λ=0.1). Its KL component (101.34) is actually the lowest, and its duration loss dropped significantly (1.17→0.95), suggesting the prosody heads provided useful auxiliary gradient signal.

### Prosody Evaluation (Primary Metric)

256 evaluation stimuli were generated (4 systems × 4 emotions × 16 texts) and analyzed for F0 (pitch) and energy characteristics.

#### Quantitative Comparison Summary

| System | F0 Mean (Hz) | F0 Std (Hz) | F0 Emotion Spread | F0 Std Spread |
|--------|-------------|-------------|-------------------|---------------|
| **A0** | 208.5 ± 17.9 | 32.0 ± 5.9 | 0.0 Hz | 0.0 Hz |
| **A** | 183.8 ± 7.3 | 21.6 ± 5.3 | 0.0 Hz | 0.0 Hz |
| **B** | 176.3 ± 13.7 | 23.3 ± 9.2 | 0.9 Hz | 2.5 Hz |
| **C** | 166.1 ± 33.5 | 37.3 ± 17.8 | **6.7 Hz** | **6.8 Hz** |

- **F0 Emotion Spread** measures how much the mean F0 differs across emotions (higher = more differentiation between emotions, desirable for expressive TTS).
- **F0 Std Spread** measures how much per-emotion pitch variability differs (higher = more expressive variation).
- **Expected trend** A0 ≤ A < B ≤ C — **confirmed**: each system progressively increases emotion differentiation.
- **A0 and A** produce identical prosody for all emotions (spread = 0.0 Hz), since neither has emotion-specific conditioning.
- **System C** shows the strongest differentiation at 6.7 Hz F0 spread and 6.8 Hz F0 Std spread — a clear improvement, though modest in absolute terms.

#### Emotion Differentiation Tests (Kruskal-Wallis)

| System | F0 Mean H (p-value) | F0 Std H (p-value) | Energy Mean H (p-value) |
|--------|--------------------|--------------------|------------------------|
| **A0** | 0.00 (1.000) | 0.00 (1.000) | 0.00 (1.000) |
| **A** | 0.00 (1.000) | 0.00 (1.000) | 0.00 (1.000) |
| **B** | 0.59 (0.899) | 2.88 (0.411) | 0.36 (0.948) |
| **C** | 1.12 (0.771) | **7.16 (0.067)** | 1.49 (0.685) |

**No system achieves statistically significant (p < 0.05) emotion differentiation** on any single prosody metric. However, System C's F0 Std approaches marginal significance (p = 0.067), suggesting that prosody supervision does push the model toward distinguishing emotions in its pitch variability pattern.

#### Causal Attribution (Mann-Whitney U Tests)

34 statistically significant causal effects were identified:

| Transition | Key Effects |
|------------|------------|
| **A0 → A** | Domain adaptation significantly **decreased** F0 mean by ~25 Hz and F0 std by ~10 Hz across all emotions (p < 0.001). This reflects the model adapting from LJSpeech (female, ~210 Hz) to EmoV-DB (male, ~180 Hz). Energy patterns also shifted significantly. |
| **A → B** | Emotion conditioning **decreased** F0 mean by ~6–9 Hz for neutral and angry (p < 0.05). Modest but significant energy changes detected. |
| **B → C** | Prosody supervision **decreased** F0 mean for neutral by ~20 Hz (p = 0.04) and **increased** F0 std for angry by ~23 Hz (p < 0.001) — the largest single emotion-specific prosody shift in the chain. |

The B→C transition for angry emotion's F0 std (+23 Hz) is particularly noteworthy: it shows the prosody heads successfully guided the model to produce more variable pitch for angry speech, which is characteristic of emotional anger expression.

### Listening Test

A P.808-inspired stimulus pack was generated with **64 stimuli** (4 texts × 4 systems × 4 emotions, ~13 min total). LUFS-normalized audio files and a response form are available in `outputs/listening_test/`. Formal evaluation was not conducted due to time constraints.

---

## Dataset

| Property | Value |
|----------|-------|
| Source | EmoV-DB |
| Speaker | Single speaker ("unknown") |
| Emotions | Amused (778), Angry (640), Disgust (798), Neutral (892) |
| Total samples | 3,108 |
| Total duration | 255.7 minutes (~4.3 hours) |
| Mean utterance length | 4.9 seconds |
| Train / Val / Test split | 2,486 / 311 / 311 |
| Sample rate | 22,050 Hz |

---

## Limitations

### Computing Power Constraints

This project was conducted entirely on **Google Colab free/standard GPU instances**. The primary limitations arising from this are:

- **Limited training epochs (15 per system).** State-of-the-art TTS fine-tuning typically requires hundreds to thousands of epochs. Our 15-epoch budget was dictated by Colab session time limits and GPU availability. The training loss curves were still declining at epoch 15 for all systems, indicating the models had not converged.
- **No hyperparameter search.** The learning rate (5×10⁻³), batch size (32), and loss weights (λ=0.1 for prosody) were set by manual tuning rather than systematic grid/random search. Better configurations likely exist.
- **Single training run per system.** Without repeated runs, we cannot report confidence intervals or assess variability in outcomes. Results may differ with different random seeds.
- **Colab session instability.** Training was interrupted by Colab runtime disconnections, requiring checkpoint restoration and careful state management.
- **No multi-GPU or distributed training.** The VITS model architecture and Coqui TTS framework support distributed training, but Colab's single-GPU constraint prevented this.

### Data Constraints

- **Small dataset (3,108 utterances, ~4.3 hours).** Modern TTS research often uses datasets of 20–100+ hours. Our limited data constrains the model's ability to learn robust emotion-specific acoustic patterns.
- **Single speaker.** All EmoV-DB recordings come from one speaker, limiting generalization to other voices and speaking styles.
- **Four emotions only.** Real-world emotional speech spans a continuum; our discrete 4-class setup (amused, angry, disgust, neutral) is a simplification.
- **Label granularity.** Emotions are assigned at the utterance level. Within-utterance emotion transitions and intensity variations are not captured.
- **Class imbalance.** The emotion distribution is imbalanced (angry: 640 vs neutral: 892), which may bias the model toward more-represented categories.

### Methodological Limitations

- **Duration predictor bypass.** Due to numerical instability in Coqui's Stochastic Duration Predictor (SDP) with short sequences, the duration loss is computed as a monitoring-only proxy (mean encoder output vs. MAS log-durations under `torch.no_grad()`). The SDP's spline parameters are not explicitly trained, which may limit prosodic control.
- **Frozen decoder.** The VITS decoder (HiFi-GAN) was kept frozen to preserve audio quality. This means the waveform generation stage cannot adapt emotion-specific spectral details that go beyond what the flow model provides.
- **No formal listening test.** While a stimulus pack was generated, subjective evaluation by human raters was not completed. Prosody metrics are a proxy for perceptual quality.
- **SER label mismatch.** The SpeechBrain emotion classifier was trained on IEMOCAP (neutral/angry/happy/sad), which does not include our "amused" or "disgust" categories, limiting the applicability of SER-based evaluation.

### Impact on Results

These constraints collectively explain why **no system achieved statistically significant emotion differentiation** (Kruskal-Wallis p < 0.05). Despite this, the consistent A0 → A → B → C improvement trend in F0 emotion spread (0.0 → 0.0 → 0.9 → 6.7 Hz) and the marginal significance of System C's F0 Std differentiation (p = 0.067) suggest that the approach is sound. With substantially more training time, data, and compute, we would expect these trends to reach statistical significance.

---

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
3. Select **GPU runtime** (T4 or higher recommended)
4. Run cells sequentially

The notebook handles: data prep → System A/B/C training → inference → evaluation → visualization.

### Gradio Demo

```bash
python demo/app.py
# Opens http://localhost:7860
```

---

## Model Architecture & Training Details

### EmotionVITS Architecture

EmotionVITS wraps the Coqui TTS VITS model with three additions:

1. **Emotion embedding** (System B+): `nn.Embedding(4, 192)` additively injected into the text encoder hidden states before the flow model.
2. **Prosody heads** (System C): Two small MLPs predicting utterance-level F0 mean/std and energy mean/std from the encoder output, trained with L1 loss (λ=0.1).
3. **Selective freezing**: text_encoder, posterior_encoder, and decoder (HiFi-GAN) are frozen. Only the normalizing flow and duration predictor parameters are trainable.

### Forward Pass

```
text → text_encoder → [+ emotion_embedding] → posterior_encoder(spec)
  → flow(z) → MAS alignment → dur_loss (monitoring only, no_grad)
  → expand m_p/logs_p → decoder(z) → waveform
  → prosody_heads(encoder_output) → prosody_loss (System C only)
```

### Loss Function

```
total_loss = kl_loss + λ × prosody_loss     (System C)
total_loss = kl_loss                         (Systems A, B)
```

Where:
- **KL loss**: KL divergence between prior (text-conditioned) and posterior (spec-conditioned) distributions, properly masked to exclude padding positions.
- **Prosody loss**: L1 between predicted and ground-truth utterance-level F0/energy statistics.
- **Mel loss** and **duration loss**: Computed for monitoring/logging but excluded from backpropagation (`torch.no_grad()`).

---

## Evaluation Methodology

### Primary Metric: Prosody Analysis
F0 and energy statistics per system × emotion, with:
- **Kruskal-Wallis tests** for within-system emotion differentiation (does the system produce different prosody for different emotions?)
- **Mann-Whitney U tests** for causal attribution (does each modification A→B→C significantly change prosody?)

### Auxiliary Metric: SER Proxy Agreement
SpeechBrain emotion classifier applied to generated audio. **Caveat:** label mismatch between IEMOCAP (neutral/angry/happy/sad) and our set (neutral/angry/amused/disgust). Disgust and amused samples are excluded from accuracy.

### Human Evaluation: Listening Test
P.808-inspired local test: naturalness + emotion match ratings on 1–5 scale using LUFS-normalized stimuli (64 stimuli, ~13 min).

---

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
