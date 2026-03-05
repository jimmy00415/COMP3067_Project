# Sprint-Milestones Plan: Emotionally Controlled TTS

**Project:** Compute-Efficient Emotionally Controlled TTS  
**Version:** 2.1 — March 5, 2026  
**Timeline:** 10 sprints × 7 days = 70 days (Mar 5 → May 13, 2026)  
**Sprint cadence:** Weekly, with mid-sprint checkpoint on Wednesday  
**Changelog v2.1:** Adopt "local coding + Google Colab compute" workflow. All code is written and tested locally; a master Colab notebook (`notebooks/colab_pipeline.ipynb`) orchestrates all GPU-heavy tasks (training, batch inference, evaluation). Colab T4 (15 GB VRAM) replaces local RTX 2050 as primary GPU.  
**Changelog v2.0:** Fixes unfair baseline (A0/A/B/C hierarchy), single-speaker-first data policy, utterance-level prosody as default, SER label mismatch, P.808 wording, dependency spike, loudness policy, canary set spec, human eval duration, and quality gate sharpening.

---

## Part I — Engineering Trade-Off Analysis

### 1. Cross-Document Tension Map

The PRD and Deep-Research Report agree on many points but diverge on several critical engineering decisions. Below is a structured conflict analysis with resolved decisions.

| # | Dimension | PRD Position | Research Report Position | Tension | **Decision & Rationale** |
|---|---|---|---|---|---|
| T1 | **TTS Backbone** | "Use Coqui TTS as the main framework" (FR-2) | "StyleTTS2 or VITS as backbone; Coqui not mentioned as primary" | Direct conflict: PRD mandates Coqui; Report favors StyleTTS2 | **Use Coqui TTS (VITS-based) as primary backbone.** PRD is the spec-of-record. Coqui wraps VITS and provides fine-tuning CLI, so we satisfy both docs. StyleTTS2 is dropped to avoid complexity and DDP issues on single-GPU. |
| T2 | **System hierarchy** | Explicitly requires 3 systems (A=baseline, B=+emotion embedding, C=+prosody aux loss) with ablations | Proposes "Method A/B/C" as alternative control strategies (reference / interpolation / tokens), not a progressive hierarchy | Structural mismatch: PRD's A→B→C is incremental; Report's A/B/C are alternatives | **Extend PRD hierarchy to A0/A/B/C.** A0 = raw pretrained LJSpeech VITS (demo only). A = fine-tuned on EmoV-DB *without* emotion labels (domain-adapted baseline). B = A + emotion embedding. C = B + prosody auxiliary loss. This fixes the **causal attribution problem**: without A, you cannot tell whether B/C gains come from emotion conditioning or just domain adaptation. The PRD's 3-system requirement is satisfied by A/B/C; A0 is an additional reference row. |
| T3 | **Config / Tracking stack** | Mandates Hydra + MLflow + DVC (FR-5) | Mentions them but deprioritizes relative to "just get results" | PRD is more prescriptive on infra | **Implement Hydra + MLflow. DVC for dataset manifest checksums only (not full model versioning).** Full DVC model versioning adds ceremony a student team won't maintain. Manifest checksums satisfy the acceptance criterion. If a DVC remote is available, use it; otherwise, `dvc add data/raw/` plus hash files is sufficient. |
| T4 | **Training vs. inference-only** | Assumes fine-tuning for System B and C (FR-2 acceptance criteria require training scripts) | Strongly warns against training; prefers inference-only reference-based methods | Direct conflict on whether to train at all | **Fine-tune lightly for A, B, and C.** PRD acceptance criteria explicitly require "train script runs end-to-end." System A fine-tune is scientifically necessary as the domain-adapted baseline. We use Coqui fine-tuning with frozen encoder layers + low LR + early stopping. Fallback: if training fails → reference-based emotion injection as degraded System B. |
| T5 | **Evaluation tooling** | Requires P.808 toolkit + optional MUSHRA | Agrees on P.808; adds SER-probe as objective proxy | Aligned with caveats | **Use structured local ACR form (P.808-inspired, not P.808-aligned) + prosody statistics as primary automatic metric + SER probe as auxiliary-only weak proxy.** The SpeechBrain wav2vec2-IEMOCAP model uses labels (neu/ang/hap/sad) that do not match our classes (neutral/angry/amused/disgust); "accuracy" is ill-defined for disgust and only loosely approximated for amused≈happy. Human emotion recognition is the primary scientific evidence. Skip AMT crowdsourcing (unrealistic). Skip MUSHRA (overkill for 4-system comparison). |
| T6 | **Demo framework** | "Local web UI or CLI" (FR-6), no specific tool | Recommends Gradio or Streamlit | Compatible | **Use Gradio.** Lower boilerplate than Streamlit for audio I/O; single-file demo possible. |
| T7 | **Emotions scope** | Doesn't specify which emotions | EmoV-DB has: neutral, sleepiness, anger, disgust, amused | Need to fix early; speaker coverage is uneven | **4 emotions: neutral, angry, amused, disgust. Single speaker first (see T11).** Drop sleepiness (low perceptual distinctiveness, hard for listeners to label). 4 classes give a tractable confusion matrix (4×4) and enough contrast for F0/energy plots. |
| T8 | **Primary training dataset** | EmoV-DB or ESD (pick 1) | EmoV-DB first, ESD as stretch | Aligned | **EmoV-DB only.** No ESD license request overhead. Single dataset simplifies pipeline and removes a schedule risk entirely. |
| T9 | **Secondary eval dataset** | CREMA-D or RAVDESS (pick 1) | RAVDESS for eval references | Aligned with caveat | **RAVDESS as sanity/reference benchmark only, not as trap/control stimuli.** CC BY-NC-SA 4.0. If RAVDESS natural speech is used as "trap" against our synthetic outputs, listeners may partly react to dataset/domain differences rather than the intended test construct. Trap/control samples should come from our own synthesis pipeline (e.g., System A natural-sounding output or held-out EmoV-DB recordings). |
| T10 | **Intensity control** | "Optional" throughout | "Bonus" trajectory/interpolation | Both mark as optional | **Implement as stretch in Sprint 8 only if C is stable.** Do not let it delay core A0/A/B/C. |
| T11 | **Speaker strategy** | Doesn't specify | EmoV-DB has 4 speakers, but coverage is uneven (speaker Jsh lacks angry/disgust) | Under-specified; naive multi-speaker split is fragile | **Single speaker with complete 4-emotion coverage as core path.** Multi-speaker only as stretch after B is stable. If multi-speaker is attempted, speaker ID must be modeled explicitly (e.g., speaker embedding), not just metadata. |

### 2. Architecture Decision Records (ADRs)

#### ADR-1: Backbone = Coqui TTS (VITS-based)

- **Context:** Need a pretrained TTS with fine-tuning support on single GPU.
- **Decision:** Coqui TTS `tts_models/en/ljspeech/vits` as starting checkpoint.
- **Consequences:** 22 kHz output (not 24 kHz like StyleTTS2). VITS architecture gives us access to posterior encoder, duration predictor, and text encoder — all points where we can inject emotion embeddings. Coqui's `TrainerConfig` handles checkpointing + logging.
- **Package status (critical):** The Coqui ecosystem is currently split across multiple distributions:
  - **Legacy `TTS`** (PyPI): Last significant GitHub activity Aug 2024. Documented as Python ≥3.7, <3.11 (for 0.22.0).
  - **`coqui-tts`** (PyPI fork/continuation): Python ≥3.10, <3.15.
  - These differ in API surface, model zoo access, and dependency pins.
- **Risk:** Choosing the wrong distribution can cause silent import failures or missing pretrained checkpoints. **Mitigation:** S0 dependency spike (S0.0) must resolve this before any other work begins. Pin exact package + version + Python in `requirements.txt` only after spike passes.

#### ADR-2: Baseline hierarchy = A0 / A / B / C (fair ablation chain)

- **Context:** The original plan used "raw pretrained LJSpeech VITS" as System A. Since B/C are fine-tuned on EmoV-DB, any improvement could come from domain adaptation rather than emotion conditioning. This is a **causal attribution problem**.
- **Decision:** Four-system hierarchy:
  - **A0:** Raw pretrained Coqui VITS on canary texts (demo/reference only, not a scientific baseline).
  - **A:** Fine-tuned on EmoV-DB **without** emotion labels (all utterances treated as single unlabeled class). This isolates the domain-adaptation effect.
  - **B:** A + learnable emotion embedding (`nn.Embedding(4, hidden_dim)`) added to text encoder hidden states.
  - **C:** B + utterance-level prosody auxiliary losses (F0 + energy).
- **Ablation chain this enables:**
  - A0 → A = domain adaptation effect (expected large improvement from LJSpeech → EmoV-DB speaker/style)
  - A → B = **explicit emotion-conditioning effect** (the claim we want to make)
  - B → C = **prosody supervision effect**
- **Rationale:** Without System A, the report would have a confound. With A, every gain is attributable.
- **Consequences:** System A requires its own fine-tuning run (~4–8 GPU-hours). This is worth the cost.

#### ADR-3: Emotion conditioning = Learnable embedding table + concat/add to text encoder

- **Context:** System B needs emotion conditioning. Options: (a) prepend text tokens, (b) learnable embedding added to encoder, (c) reference audio style transfer.
- **Decision:** (b) Learnable embedding table of size `[num_emotions × hidden_dim]`, added to text encoder hidden states before flow/decoder.
- **Rationale:** More principled than token hacking (a); doesn't require reference audio at inference (c). Clean ablation: remove embedding → System A.
- **Consequences:** Requires modifying Coqui VITS model class (~50 lines). Must freeze most pretrained weights and only train embedding + adapter layers.

#### ADR-4: Prosody auxiliary loss = Utterance-level F0 + energy (frame-level as stretch)

- **Context:** System C needs prosody-aware supervision (PRD says "pitch/energy/duration, choose ≥2").
- **Decision (revised):** Two implementation tiers:
  - **C1 (main path, default):** Utterance-level prosody targets — predict mean log-F0, F0 std/range, mean log-energy, energy std from the pooled text-encoder output via lightweight MLP heads. L1 loss against ground-truth statistics. **This does not require frame-level alignment.**
  - **C2 (stretch, only if C1 is stable):** Frame-level log-F0 and log-energy prediction using VITS's MAS alignment. Higher fidelity but requires aligning frame-level prosody to text-encoder timesteps — exactly the kind of integration that burns 10–14 days in a student project.
- **Rationale:** Duration prediction is already in VITS (stochastic duration predictor). F0 + energy are the two most emotion-discriminative prosody features. Utterance-level targets are trivially extractable, produce clean ablation results, and are scientifically defensible ("prosody-aware" supervision at utterance granularity). Frame-level upgrades the fidelity but is not required for the core claim.
- **Consequences:** C1 needs only aggregated statistics per utterance (already computed in S1). No alignment complexity. MLP heads are ~20 lines of code.

#### ADR-5: Evaluation = P.808-inspired local listening test + Prosody stats (primary) + SER proxy (auxiliary)

- **Context:** P.808 crowdsourcing on AMT is unrealistic for students. The SpeechBrain wav2vec2-IEMOCAP emotion model has a **label mismatch** with our task.
- **Decision:**
  - **Human eval (primary scientific evidence):** Google Forms–based structured ACR test, **P.808-inspired** (not "P.808-aligned" — we do not implement the full P.808 crowdsourcing protocol with environment checks, trapping, and AMT integration). 5-point MOS for naturalness, forced-choice emotion identification, confidence Likert. Recruit 15-20 peers. **12–18 minute session per listener** (10–12 test stimuli + 2 controls + demographics/device check). Trap/control stimuli drawn from our own pipeline outputs (not cross-domain RAVDESS recordings).
  - **Prosody diagnostics (primary automatic metric):** F0 (mean, std, range), energy (mean, std), speaking rate. Violin/box plots by emotion × system.
  - **SER probe (auxiliary only, weak proxy):** SpeechBrain `emotion-recognition-wav2vec2-IEMOCAP`. Labels: neu/ang/hap/sad. Our classes: neutral/angry/amused/disgust. Mapping: neutral→neu, angry→ang, amused≈hap (loose), disgust→**unmapped**. Report as `ser_proxy_agreement` (not "accuracy"). Document label incompatibility explicitly in code, README, and report. The model card warns performance on other datasets is not warranted.
  - **Optional: ASR WER intelligibility check** on generated text as an additional automatic metric.
- **Consequences:** Human eval requires ~1 week for collection + analysis. Shorter form means more complete responses and less fatigue contamination.

#### ADR-6: Infrastructure tiers (Must / Should / Could)

| Tier | Component | Justification |
|---|---|---|
| **Must** | Git + Hydra configs + MLflow tracking + seed control | PRD FR-5 acceptance criteria |
| **Must** | Unit tests for data pipeline + audio I/O | PRD Section 9.1 |
| **Should** | DVC dataset manifest checksums (+ remote if available) | Lightweight compliance with FR-5 DVC req |
| **Should** | GitHub Actions CI (lint + smoke test) | High report/presentation impact, low effort |
| **Could** | Docker for inference | Only if time in Sprint 9+ |
| **Could** | Full DVC model versioning / experiment lineage | Adds ceremony; skip unless team will maintain it |

#### ADR-7: Data policy — single speaker, conservative normalization, separate eval stimuli

- **Context:** EmoV-DB speaker coverage is uneven (speaker Jsh lacks angry/disgust). Loudness normalization can erase energy variation that is part of the emotional signal. Training texts must be separated from evaluation texts.
- **Decisions:**
  - **Speaker:** Choose one EmoV-DB speaker with complete coverage across all 4 target emotions. Multi-speaker is stretch only and requires explicit speaker-ID conditioning.
  - **Loudness normalization:** Training audio receives **conservative** normalization only (peak normalization to -1 dBFS, no aggressive LUFS loudness matching). Energy variation across emotions is preserved as a training signal. For human evaluation playback, a separate copy of stimuli is loudness-normalized (e.g., -23 LUFS per EBU R128) to ensure listener fairness.
  - **Data split policy:** Explicitly distinguish (a) training splits from (b) evaluation stimulus design. The frozen human-eval set must use only held-out test texts never seen during training or model debugging (i.e., not the canary set if it was used for iterative development).
- **Consequences:** One speaker simplifies the pipeline dramatically. Conservative normalization preserves a signal we care about for diagnostics.

#### ADR-8: Compute strategy — Local coding + Google Colab for GPU tasks

- **Context:** Local machine has an RTX 2050 (4 GB VRAM) — insufficient for VITS fine-tuning. Google Colab provides free T4 GPUs (15 GB VRAM) or paid A100 access.
- **Decision:** Two-environment workflow:
  - **Local (VS Code):** All code authoring, unit tests, linting, config editing, data QA (CPU), Gradio demo development. Code lives in the Git repo as normal Python modules (`src/`, `configs/`, `tests/`).
  - **Google Colab (`notebooks/colab_pipeline.ipynb`):** A single master notebook that: (1) mounts Google Drive, (2) clones/pulls the repo, (3) installs dependencies, (4) runs GPU-heavy tasks in sequence (data preprocessing, training A/B/C, batch inference, evaluation). Each compute phase is a clearly labeled notebook section with markdown cells explaining the step.
- **Data flow:** Raw data (EmoV-DB) stored on Google Drive. Processed data, checkpoints, and generated audio saved back to Drive. Final outputs (samples, figures, tables) synced to local repo for report writing.
- **Rationale:** Avoids VRAM bottleneck. Colab's free T4 (15 GB) easily fits VITS fine-tuning. Notebook format provides a self-documenting execution log (with cell outputs preserved) which strengthens the "completeness" rubric.
- **Consequences:** Must ensure all `src/` code is importable from notebook (proper `sys.path` setup or `pip install -e .`). Training configs still managed via Hydra YAML files, invoked from notebook cells. Checkpoints must be saved to Drive (not ephemeral Colab disk).

### 3. Compute Budget Estimation

| Phase | GPU-hours (est.) | Runs on | Notes |
|---|---|---|---|
| Data preprocessing | 0 (CPU) | Local or Colab | Audio resampling + F0/energy extraction |
| System A0 inference | ~0.5 | Colab T4 (or CPU) | Just load pretrained + generate canary texts |
| System A fine-tuning (no emotion labels) | 4–8 | **Colab T4** (15 GB) | Same data as B but no emotion embedding; establishes domain baseline |
| System B fine-tuning | 8–16 | **Colab T4** (15 GB) | Freeze encoder, train embedding + adapter, ~50 epochs on single-speaker EmoV-DB |
| System C fine-tuning | 8–16 | **Colab T4** (15 GB) | Same as B + utterance-level prosody heads; warm-start from B |
| Inference (all systems, full eval set) | ~1 | Colab T4 | Batch generate ~200+ samples |
| SER probe | ~0.5 | Colab T4 | Inference-only wav2vec2; auxiliary metric only |
| **Total** | **~22–42 GPU-hours** | | Fits in free-tier Colab sessions (reset every ~4h); use Drive checkpointing to resume |

---

## Part II — Sprint-Milestones Plan

### Sprint Overview

| Sprint | Dates | Theme | Key Deliverable |
|---|---|---|---|
| **S0** | Mar 5–11 | Foundation & Dependency Spike | Env proven, repo scaffold, data downloaded |
| **S1** | Mar 12–18 | Data Pipeline + QA + System A0 | FR-1 complete, single-speaker data validated, A0 samples |
| **S2** | Mar 19–25 | System A (domain baseline) + B Scaffolding | A fine-tuned, EmotionVITS forward pass works |
| **S3** | Mar 26–Apr 1 | System B Training + Midpoint | System B converged, **Midpoint Report submitted** |
| **S4** | Apr 2–8 | System C: Utterance-Level Prosody | Prosody heads + aux loss integrated (utterance-level) |
| **S5** | Apr 9–15 | System C Stabilization + Ablations | C converged, ablation chain A0→A→B→C complete, eval set frozen |
| **S6** | Apr 16–22 | Evaluation Harness (Automatic) | Prosody stats + SER proxy + plots pipeline |
| **S7** | Apr 23–29 | Human Evaluation | Listening test launched + collected |
| **S8** | Apr 30–May 6 | Analysis, Demo & Stretch | Gradio demo, intensity stretch, all figures |
| **S9** | May 7–13 | Final Report & Presentation | Report submitted, slides done, repo polished |

---

### Sprint S0: Foundation & Dependency Spike (Mar 5–11)

**Goal:** Resolve the Coqui distribution question first (S0.0), then build a zero-to-runnable dev environment. Every team member can clone, install, and run pretrained Coqui TTS inference.

| ID | Task | Owner | Acceptance Criteria | Est. Hours |
|---|---|---|---|---|
| S0.0 | **Dependency spike** (DO THIS FIRST) | Lead | (1) Decide `TTS` vs `coqui-tts` PyPI package; (2) Lock Python version (3.10 recommended for Colab compatibility); (3) Verify `tts_models/en/ljspeech/vits` pretrained checkpoint exists in chosen distribution; (4) Verify minimal fine-tuning script imports run without error (e.g., `from TTS.tts.configs.vits_config import VitsConfig`); (5) **Test on Google Colab** — create a quick spike notebook, install package, run inference. (6) Only then write `requirements.txt` with pinned versions. Document findings in `docs/dependency_spike.md` | 4 |
| S0.1 | Create repo with standard layout (see below) | Lead | `git clone` + structure matches spec | 2 |
| S0.2 | Write `requirements.txt` (from spike results) | Lead | `pip install -r requirements.txt` succeeds on team machines; versions pinned | 1 |
| S0.3 | Install & verify Coqui TTS pretrained model | All | `tts --text "Hello" --model_name tts_models/en/ljspeech/vits --out_path test.wav` produces audio | 1 |
| S0.4 | Download EmoV-DB dataset | Data | All emotion folders present; file count matches published stats; raw speaker inventory documented | 3 |
| S0.5 | Download RAVDESS dataset (reference/benchmark only) | Data | Audio files + metadata CSV extracted. Noted: **not for trap/control stimuli** | 1 |
| S0.6 | Set up Hydra config scaffold | Lead | `configs/` has `data.yaml`, `train_a.yaml`, `train_b.yaml`, `train_c.yaml`, `infer.yaml`, `eval.yaml` with placeholder values | 2 |
| S0.7 | Set up MLflow local tracking server | Lead | `mlflow ui` launches; test experiment logged | 1 |
| S0.8 | Write `NOTICE.md` (licenses) | Lead | All deps listed with license + purpose | 1 |
| S0.9 | DVC init + dataset manifest checksums | Lead | `dvc init; dvc add data/raw/` produces `.dvc` files; if DVC remote available, configure push | 1 |
| S0.10 | Define "canary text set" (16 sentences) | All | `configs/canary_texts.txt` committed. Must cover: (a) short (≤6 words) vs long (≥15 words); (b) declarative, interrogative, exclamatory, imperative; (c) emotionally compatible text (e.g., "I can't believe you did that!") vs emotionally neutral/conflicting text (e.g., "The quarterly report is ready."); (d) varied phoneme coverage (include fricatives, plosives, nasals, liquids). Purpose: avoid demo overfitting to a few friendly sentences | 2 |
| S0.11 | **Create master Colab notebook** `notebooks/colab_pipeline.ipynb` | Lead | Skeleton notebook with section headers for each compute phase: (0) Setup + Drive mount, (1) Data prep, (2) System A training, (3) System B training, (4) System C training, (5) Inference + sample generation, (6) Evaluation. Each section has install cells, import cells, and execution cells calling `src/` modules. Verify: notebook runs the setup section on Colab without errors | 3 |

**Repo structure to create:**
```
COMP3065_Project/
├── configs/
│   ├── data.yaml
│   ├── train_a.yaml       # System A: domain fine-tune, no emotion
│   ├── train_b.yaml       # System B: + emotion embedding
│   ├── train_c.yaml       # System C: + prosody aux loss
│   ├── train_debug.yaml   # Small-run debug config
│   ├── infer.yaml
│   └── eval.yaml
├── data/
│   ├── raw/               # EmoV-DB + RAVDESS (git-ignored, DVC-tracked)
│   ├── processed/         # After preprocessing
│   │   ├── train/         # Training audio (conservative normalization)
│   │   └── eval_stimuli/  # Playback-normalized copies for human eval
│   └── manifests/         # CSV/JSONL splits
├── docs/
│   ├── dependency_spike.md
│   └── data_qa_report.md
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── prepare.py     # FR-1 pipeline
│   │   ├── qa.py          # Data QA: clip/text length histograms, bad files, noise check
│   │   └── utils.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline.py    # System A0/A wrapper
│   │   ├── emotion_vits.py   # System B/C model
│   │   └── prosody_heads.py  # System C aux heads (utterance-level default)
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train.py
│   │   └── callbacks.py
│   ├── inference/
│   │   ├── __init__.py
│   │   └── run.py         # FR-3
│   └── evaluation/
│       ├── __init__.py
│       ├── prosody.py      # F0/energy analysis (primary automatic metric)
│       ├── ser_probe.py    # SpeechBrain wrapper (AUXILIARY proxy only)
│       ├── listening_test.py   # Generate stimulus packs
│       └── plots.py        # Visualization
├── notebooks/
│   └── colab_pipeline.ipynb  # Master Colab notebook: all GPU tasks
├── demo/
│   └── app.py             # Gradio demo
├── tests/
│   ├── test_data.py
│   ├── test_audio_io.py
│   └── test_inference.py
├── outputs/               # Generated audio (git-ignored)
├── figures/               # Plots for report
├── tables/                # CSV results for report
├── NOTICE.md
├── README.md
├── requirements.txt
├── setup.py
└── .github/
    └── workflows/
        └── ci.yml
```

**Exit criteria for S0:**
- [ ] Dependency spike completed: exact package + version + Python documented and proven
- [ ] Every team member can run pretrained Coqui VITS and produce `test.wav`
- [ ] **Colab spike passes**: notebook installs deps, loads pretrained model, generates audio on Colab T4
- [ ] EmoV-DB raw data is on disk (and uploaded to Google Drive); file count verified; speaker inventory logged
- [ ] `mlflow ui` shows the workspace; one dummy run logged
- [ ] Canary text set has 16 sentences covering all required categories
- [ ] Master Colab notebook skeleton committed with working setup section
- [ ] Repo pushed to Git with structure above

---

### Sprint S1: Data Pipeline + QA + System A0 (Mar 12–18)

**Goal:** Complete FR-1 (data pipeline) with single-speaker-first policy, Coqui-recommended data QA, and System A0 (raw pretrained) inference.

| ID | Task | Acceptance Criteria | Est. Hours |
|---|---|---|---|
| S1.1 | **Identify core speaker** | Audit EmoV-DB: list per-speaker emotion coverage and sample counts. Select one speaker with complete coverage across neutral, angry, amused, disgust. Document: which speakers are missing which emotions (e.g., Jsh lacks angry/disgust). Save `docs/data_qa_report.md` | 3 |
| S1.2 | Implement `src/data/prepare.py` for EmoV-DB | Reads raw EmoV-DB, filters to chosen speaker, resamples to 22050 Hz, applies **conservative peak normalization** (-1 dBFS only, no aggressive LUFS matching — energy is part of the emotional signal), outputs `data/processed/train/` | 6 |
| S1.3 | Build emotion label mapping | `emotion_map.json`: `{neutral: 0, angry: 1, amused: 2, disgust: 3}` | 1 |
| S1.4 | Transcript normalization | Lowercase, strip punctuation variants, verify against CMU Arctic text files | 3 |
| S1.5 | **Data QA (Coqui-recommended checks)** | Implement `src/data/qa.py`: (a) clip length histogram, (b) transcript length histogram, (c) bad/corrupted file detection, (d) annotation-audio consistency check, (e) noise/spectrogram visual inspection on random subset, (f) per-emotion class counts. Save all to `docs/data_qa_report.md` and `figures/data_qa/` | 4 |
| S1.6 | Create frozen train/val/test splits | 80/10/10 stratified by emotion (single speaker). Compute class counts **before** splitting to verify balance. Save as `manifests/train.csv`, `val.csv`, `test.csv`. Additionally create `manifests/eval_holdout.csv`: a set of texts **never used in training or canary debugging** — reserved exclusively for human evaluation stimulus generation | 3 |
| S1.7 | Compute & store split statistics | Print and save: samples per emotion per split, total duration. Save `tables/dataset_stats.csv` | 2 |
| S1.8 | Extract F0 contours (pYIN via librosa) | For all processed audio. **Speaker-aware configuration:** set `fmin`/`fmax` based on core speaker's vocal range (e.g., male: 75–300 Hz, female: 100–500 Hz). Mask unvoiced frames in output. Save as `.npy` files alongside audio. Note: default pYIN frame length at 22050 Hz is ~93 ms; acceptable for diagnostics but avoid over-interpreting brief fluctuations | 4 |
| S1.9 | Extract energy contours + utterance-level stats | Frame-level RMS energy saved as `.npy`. Also compute and save per-utterance summary statistics: mean F0, F0 std, F0 range, mean energy, energy std → appended to manifest CSV. These will serve as prosody targets for System C (utterance-level) | 3 |
| S1.10 | System A0: Coqui pretrained inference wrapper | `src/models/baseline.py` wraps Coqui's `Synthesizer`; takes text → wav | 3 |
| S1.11 | System A0: Generate canary text samples | `outputs/system_a0/` with 16 canary texts from raw pretrained LJSpeech VITS, manifest CSV. These are reference-only (not a scientific baseline) | 2 |
| S1.12 | Unit tests for data pipeline | `tests/test_data.py`: test resampling, label mapping, split reproducibility, F0 extraction on known WAV | 3 |
| S1.13 | Acceptance gate: `python -m src.data.prepare --config configs/data.yaml` | Deterministic output on repeated runs | 1 |

**Key metric to log in MLflow:** Dataset stats (samples per emotion, total hours, core speaker ID).

**Quality Gate G1 (end of S1): Data QA must be complete before any model modification begins.** If data QA reveals coverage gaps or quality issues, fix them before proceeding.

**Exit criteria for S1:**
- [ ] Core speaker selected with documented rationale; coverage verified for all 4 emotions
- [ ] `data/processed/train/` contains conservatively normalized WAVs, F0 `.npy`, energy `.npy`
- [ ] Utterance-level prosody stats appended to manifests (for System C targets)
- [ ] `data/manifests/` contains frozen splits with correct counts + separate `eval_holdout.csv`
- [ ] Data QA report in `docs/data_qa_report.md` with histograms in `figures/data_qa/`
- [ ] `outputs/system_a0/` has 16 canary sentences as WAV + manifest CSV
- [ ] `python -m pytest tests/test_data.py` passes
- [ ] Summary stats saved to `tables/dataset_stats.csv`

---

### Sprint S2: System A (Domain Baseline) + System B Scaffolding (Mar 19–25)

**Goal:** Fine-tune System A (domain-adapted baseline, no emotion labels) and scaffold EmotionVITS for System B. This sprint is scientifically critical: without System A, every B/C improvement is confounded by domain adaptation.

| ID | Task | Acceptance Criteria | Est. Hours |
|---|---|---|---|
| S2.1 | **System A: domain baseline fine-tuning** (code) | Write training script / config for fine-tuning pretrained VITS on EmoV-DB core-speaker data **without any emotion labels** (all utterances treated as a single class). Use Coqui's standard fine-tuning recipe. Freeze text encoder + decoder; unfreeze duration predictor + flow. LR=5e-5, early stopping patience=10 | 6 |
| S2.2 | System A training config YAML | `configs/train_a.yaml`: no emotion embedding, standard VITS loss only. Document: "this is the domain-adapted baseline to isolate conditioning effect" | 2 |
| S2.3 | System A debug run (local CPU) | Small-run config: 50 samples, 5 epochs, completes in <5 min on CPU. Validates code correctness before Colab | 2 |
| S2.4 | Launch full System A training (**Colab**) | Run System A training in Colab notebook Section 2; loss decreasing after epoch 5. Checkpoint saved to Google Drive | 3 |
| S2.5 | Fork/modify Coqui VITS model class for System B | New class `EmotionVITS` in `src/models/emotion_vits.py` with `nn.Embedding(4, hidden_dim)` injected into text encoder output | 8 |
| S2.6 | Design embedding injection point | Emotion embedding broadcast-added to text encoder hidden states before posterior encoder / flow. Document decision with architecture diagram | 3 |
| S2.7 | Implement Coqui-compatible dataset class | Extends Coqui's dataset to include emotion label from manifest CSV | 4 |
| S2.8 | System B debug config | `configs/train_debug.yaml`: 2 emotions, 50 samples, 5 epochs. Must complete in <5 min on CPU | 2 |
| S2.9 | MLflow integration for training | Each run logs: config hash, epoch losses, val losses, checkpoint path, git commit, system_id tag (A/B/C) | 3 |

**Quality Gate G2 (Wednesday Mar 22):** If EmotionVITS forward pass is still blocked after one focused attempt (~10h), immediately switch to **Fallback B-alt:** Use Coqui's speaker embedding mechanism to "overload" speaker IDs as emotion IDs. Set `num_speakers = 4` (one per emotion) and fine-tune. Less elegant but requires zero model surgery. Do not burn a whole sprint proving you can surgically edit internals.

**Exit criteria for S2:**
- [ ] System A training completed or nearly converged
- [ ] System A checkpoint saved and logged in MLflow with tag `system=A`
- [ ] `EmotionVITS` class instantiates and runs forward pass with emotion labels (or fallback B-alt activated)
- [ ] Debug training for B completes without errors (even if quality is poor)
- [ ] MLflow has logged System A run + B debug run

---

### Sprint S3: System B Training + Midpoint Report (Mar 26–Apr 1)

**Goal:** System B produces emotion-differentiated audio. System A inference samples generated. Midpoint Report submitted.

| ID | Task | Acceptance Criteria | Est. Hours |
|---|---|---|---|
| S3.1 | System A inference on canary texts | `outputs/system_a/` has 16 canary text samples (no emotion variation — domain baseline). Compare audio quality vs A0 | 2 |
| S3.2 | Configure System B training | `configs/train_b.yaml`: Initialize from **System A checkpoint** (not raw pretrained). LR=1e-4, batch_size=16, max_epochs=100, early_stopping patience=10, gradient clipping. Freeze: text encoder, posterior encoder, decoder. Unfreeze: emotion embedding, duration predictor, flow layers | 2 |
| S3.3 | Launch + complete System B training (**Colab**) | Run in Colab notebook Section 3. Training converged (val loss plateaued for ≥5 epochs) or hit max epochs. Checkpoint saved to Drive | 8 (monitoring) |
| S3.4 | System B inference on canary texts × 4 emotions | `outputs/system_b/` has 64 samples (16 texts × 4 emotions) + manifest | 2 |
| S3.5 | Prosody comparison A0 vs A vs B | F0 box plots: A0 (pretrained neutral) vs A (domain-adapted neutral) vs B (4 emotions). This is the first test of the ablation chain. Save to `figures/` | 3 |
| S3.6 | Integration test: full pipeline smoke test | `tests/test_inference.py`: load System B checkpoint → generate 1 sample → assert WAV is valid (no NaN, duration > 0.5s, loudness in range) | 3 |
| S3.7 | **Write Midpoint Report** | Sections below | 12 |
| S3.8 | Midpoint slides (if required) | 5–8 slides covering same content | 4 |

**Midpoint Report content outline (revised):**
1. **Intro/Motivation** (1 page): Emotional TTS challenge, one-to-many mapping, prosody coupling. Cite [1][2][5].
2. **Problem & Data** (1 page): EmoV-DB description, 4 emotions, single-speaker rationale, preprocessing pipeline, split stats table, data QA summary.
3. **Progress** (2 pages):
   - System A0: raw pretrained VITS inference (reference).
   - System A: domain-adapted baseline, training curve. Explain why this is scientifically necessary (isolate conditioning vs domain adaptation).
   - System B: emotion embedding architecture diagram, training curve, F0 comparison A0 → A → B.
   - Table: "Open-source components vs. our contributions."
4. **Evaluation Plan** (1 page): ACR protocol (**P.808-inspired local listening test**), prosody stats as primary automatic metric, SER proxy with label-mismatch caveat, ablation matrix (A0→A→B→C).
5. **Future Plan & Risks** (0.5 page): System C utterance-level prosody heads, human eval timeline, compute risk mitigation, frame-level stretch.

**Exit criteria for S3:**
- [ ] System A and B checkpoints saved and logged in MLflow
- [ ] System B produces audibly different output for different emotions (informal team check)
- [ ] Comparative Figure (A0 vs A vs B F0) in `figures/`
- [ ] Midpoint Report PDF submitted
- [ ] All code committed; repo in clean state

---

### Sprint S4: System C — Utterance-Level Prosody Losses (Apr 2–8)

**Goal:** Add utterance-level F0 + energy prediction heads to System B → System C. Frame-level alignment is **explicitly out of scope** for this sprint (stretch only later).

| ID | Task | Acceptance Criteria | Est. Hours |
|---|---|---|---|
| S4.1 | Implement `src/models/prosody_heads.py` (utterance-level) | Two MLP heads: `F0StatsHead(hidden_dim → 4)` predicting [mean_f0, f0_std, f0_range_low, f0_range_high] and `EnergyStatsHead(hidden_dim → 2)` predicting [mean_energy, energy_std]. Each is 2-layer MLP with ReLU. Input: pooled (mean) text-encoder hidden states | 4 |
| S4.2 | Integrate heads into `EmotionVITS` forward | Heads receive mean-pooled encoder output. Predictions returned alongside VITS outputs | 3 |
| S4.3 | Implement prosody loss computation | L1 loss between predicted and ground-truth utterance-level stats (from S1.9 manifest). Weight: `λ_prosody = 0.1` relative to main VITS loss. **No frame-level alignment needed** | 3 |
| S4.4 | Training config for System C | `configs/train_c.yaml`: same as B but with prosody loss enabled. Initialize from **System B checkpoint** (warm start) | 2 |
| S4.5 | Debug training run | Small-run config completes; prosody losses logged separately in MLflow (`loss_f0_stats`, `loss_energy_stats`) | 2 |
| S4.6 | Launch full System C training (**Colab**) | Run in Colab notebook Section 4. Training starts; total loss + prosody losses tracked. Checkpoint to Drive | 2 |
| S4.7 | Hyperparameter note: λ sensitivity | Log runs with `λ_prosody ∈ {0.05, 0.1, 0.2}` if compute allows. Otherwise, fix at 0.1 and note in report as limitation | 3 |

**Quality Gate G3 (end of S4): If utterance-level C is stable, ship it.** Do not let frame-level alignment destroy the schedule. Frame-level C2 is stretch-only, to be attempted in S8 if time permits.

**Exit criteria for S4:**
- [ ] System C forward pass runs without errors (debug config)
- [ ] MLflow logs show separate `loss_f0_stats`, `loss_energy_stats`, `loss_total`
- [ ] Full training is running or queued

---

### Sprint S5: System C Stabilization + Ablations + Eval Freeze (Apr 9–15)

**Goal:** System C converged. Full ablation chain A0→A→B→C exercised. Evaluation sample set frozen.

| ID | Task | Acceptance Criteria | Est. Hours |
|---|---|---|---|
| S5.1 | Complete System C training (**Colab**) | Val loss plateau or max epochs reached. Final checkpoint saved to Drive | 8 (monitoring) |
| S5.2 | System C inference on canary texts × 4 emotions | `outputs/system_c/` has 64 samples (16 texts × 4 emotions) + manifest | 2 |
| S5.3 | **Ablation chain: A0 → A → B → C** | Generate same test sentences through all 4 systems. Compare prosody stats. This is the core scientific result: (a) A0→A = domain adaptation; (b) A→B = emotion conditioning; (c) B→C = prosody supervision | 4 |
| S5.4 | **Ablation: C without F0 loss** (energy-only prosody) | Retrain from B with only energy stats head; generate samples | 4 |
| S5.5 | **Ablation: C without energy loss** (F0-only prosody) | Retrain from B with only F0 stats head; generate samples | 4 |
| S5.6 | **Freeze evaluation sample set** | Select 5 eval-holdout texts (from `eval_holdout.csv`, NOT canary set, NOT training texts) × 4 emotions × 4 systems (A0/A/B/C) = 80 samples. Additionally generate 4 system-A outputs as control/trap stimuli. Lock all file hashes in `manifests/eval_frozen_manifest.csv`. Prepare a second copy in `data/processed/eval_stimuli/` with **playback loudness normalization** (e.g., -23 LUFS) for listener fairness — training copies remain conservatively normalized | 3 |
| S5.7 | Model regression test | All frozen samples pass: no NaN, duration 0.5–15s, loudness within expected range | 2 |
| S5.8 | Generate spectrograms and waveform plots | Same-text multi-emotion panels for 3 selected sentences × systems A0/A/B/C. Save to `figures/` | 3 |
| S5.9 | Log all checkpoints + configs in MLflow | Each system's final checkpoint has: run_id, config, val_loss, epoch, system_id tag | 1 |

**Critical milestone:** After S5.6, the evaluation sample set is **frozen**. No regeneration after this point. All human eval stimuli come from `eval_stimuli/` (loudness-normalized copies). All automatic analyses can run on either copy.

**Exit criteria for S5:**
- [ ] Systems A0, A, B, C all have final checkpoints and samples generated
- [ ] Ablation chain results documented (at minimum: prosody stats table across A0→A→B→C)
- [ ] Frozen eval set of 80+ samples with hash manifest
- [ ] Eval-stimuli copies loudness-normalized for playback
- [ ] All spectrogram/waveform figures in `figures/`

---

### Sprint S6: Evaluation Harness — Automatic (Apr 16–22)

**Goal:** Complete automatic evaluation pipeline. Prosody statistics as primary automatic metric. SER probe as auxiliary-only. All diagnostic figures and tables generated.

| ID | Task | Acceptance Criteria | Est. Hours |
|---|---|---|---|
| S6.1 | Implement `src/evaluation/prosody.py` | Extract F0 (mean, std, range), energy (mean, std), speaking rate for any WAV file. Speaker-aware `fmin`/`fmax` for pYIN. Returns dict. This is the **primary automatic metric** | 5 |
| S6.2 | Run prosody analysis on all frozen samples | Output: `tables/prosody_stats.csv` with columns: `system, emotion, text_id, f0_mean, f0_std, f0_range, energy_mean, energy_std, speaking_rate` | 2 |
| S6.3 | Implement `src/evaluation/plots.py` | Functions: `plot_f0_distribution()`, `plot_energy_distribution()`, `plot_confusion_matrix()`, `plot_prosody_contour()`, `plot_ablation_chain()` | 6 |
| S6.4 | Generate all prosody distribution figures | Violin/box plots: F0 by emotion (per system), energy by emotion (per system). Ablation chain figure: A0→A→B→C prosody stats progression. Save to `figures/` | 3 |
| S6.5 | Implement `src/evaluation/ser_probe.py` (**AUXILIARY ONLY**) | Wrapper around SpeechBrain wav2vec2 emotion model. **Prominently document label mismatch:** model labels are neu/ang/hap/sad; our classes are neutral/angry/amused/disgust. Mapping: neutral→neu (exact), angry→ang (exact), amused→hap (loose approximation), disgust→**unmapped** (no IEMOCAP equivalent). Output column is `ser_proxy_agreement` (NOT "accuracy"). Docstring, README, and report must all state: "proxy only; model card warns performance on other datasets is not warranted" | 4 |
| S6.6 | Run SER probe on all frozen samples | Output: `tables/ser_proxy_results.csv` with columns: `system, emotion_target, ser_predicted_label, ser_confidence, label_compatible` (boolean: is mapping valid?) | 1 |
| S6.7 | Generate SER proxy visualizations | Per-system heatmaps showing predicted-vs-target mapping. Label "proxy" prominently. Save to `figures/` | 2 |
| S6.8 | Optional: ASR intelligibility check | Run a pretrained ASR (e.g., Whisper-small) on generated samples; compute WER against input text. Report as intelligibility check | 3 |
| S6.9 | Implement one-command evaluation script | `python -m src.evaluation.run --config configs/eval.yaml` produces all `tables/*.csv` and `figures/*.png` | 3 |
| S6.10 | Write evaluation unit tests | `tests/test_evaluation.py`: test prosody extraction on a known WAV, test plot generation | 2 |

**Exit criteria for S6:**
- [ ] `python -m src.evaluation.run` produces all tables and figures without error
- [ ] `tables/` contains: `prosody_stats.csv`, `ser_proxy_results.csv`
- [ ] `figures/` contains: F0 distributions, energy distributions, SER proxy heatmaps (A0/A/B/C), ablation chain figure, prosody contour panels
- [ ] SER label mismatch documented in at least 3 places (docstring, README, report draft notes)
- [ ] Tests pass

---

### Sprint S7: Human Evaluation (Apr 23–29)

**Goal:** Design, launch, collect, and analyze a **P.808-inspired local listening test** (not P.808-aligned — we do not implement the full crowdsourcing protocol). Short form to avoid listener fatigue.

| ID | Task | Acceptance Criteria | Est. Hours |
|---|---|---|---|
| S7.1 | Design listening test protocol | Document: test structure (ACR naturalness + forced-choice emotion), randomization strategy, control stimuli **drawn from our own pipeline** (NOT cross-domain RAVDESS recordings), participant instructions, device/headphone check question. Target: **12–18 minutes per listener** | 4 |
| S7.2 | Create Google Forms listening test | Form with: audio playback embeds (hosted on Google Drive, using loudness-normalized eval copies from `eval_stimuli/`), 5-point MOS scale per sample, emotion identification dropdown (neutral/angry/amused/disgust/none-of-above), confidence Likert. **10–12 test stimuli + 2 controls per listener** (balanced design: each listener hears all 4 systems on different texts/emotions via Latin-square rotation) | 6 |
| S7.3 | Pilot test (2–3 team members) | Run through entire form; identify bugs, unclear instructions, audio playback issues. Time the session — must be ≤18 min | 2 |
| S7.4 | Recruit 15–20 listeners | Peers/classmates. Send form link. Set deadline | 1 |
| S7.5 | Monitor responses mid-week | Check: enough responses? Any patterns of confusion? | 1 |
| S7.6 | Close collection + export data | Download responses as CSV. Map coded sample IDs back to (system, emotion, text) | 2 |
| S7.7 | Compute MOS per system | Mean ± 95% CI. Kruskal-Wallis test for significance (non-parametric, appropriate for ordinal Likert data). Save to `tables/mos_results.csv` | 3 |
| S7.8 | Compute emotion recognition accuracy | Per-system accuracy and confusion matrix. Save to `tables/emotion_accuracy.csv` and `figures/human_confusion_*.png` | 3 |
| S7.9 | Compute inter-rater agreement | Fleiss' kappa or Krippendorff's alpha for emotion identification task | 2 |
| S7.10 | Write "Fail cases" log | Identify samples where listeners consistently misidentified emotion or rated very low. Qualitative analysis for report Discussion section | 2 |

**Listening test design spec (revised — short form):**

```
Per listener (randomized order, 12-18 min total):
├── 2 control samples (high-quality System A outputs, expected MOS ≥3.5)
│   NOT drawn from RAVDESS (avoids cross-domain confound)
├── 10-12 test stimuli balanced across:
│   - 4 systems (A0, A, B, C) — each heard ≥2 times
│   - 4 emotions — each heard ≥2 times
│   - Latin-square rotation so different listeners get different combos
│   (each individual stimulus is ~3-8 seconds)
├── 1 demographics/device check question at start
└── Total: 12-14 stimuli → ~12-18 min session

Response per sample:
  1. Naturalness (1-5 ACR scale: Bad/Poor/Fair/Good/Excellent)
  2. Perceived emotion (forced choice: neutral/angry/amused/disgust/none-of-above)
  3. Confidence in emotion judgment (1-5 Likert)
```

**Exit criteria for S7:**
- [ ] ≥15 valid listener responses collected
- [ ] `tables/mos_results.csv` and `tables/emotion_accuracy.csv` generated
- [ ] Human evaluation confusion matrices in `figures/`
- [ ] Statistical tests documented (p-values, CIs)
- [ ] Protocol correctly described as "P.808-inspired local listening test"

---

### Sprint S8: Demo, Analysis & Stretch Goals (Apr 30–May 6)

**Goal:** Working Gradio demo. Stretch: intensity control and/or frame-level prosody. Deep analysis for report.

| ID | Task | Acceptance Criteria | Est. Hours |
|---|---|---|---|
| S8.1 | Implement Gradio demo `demo/app.py` | Input: text box + emotion dropdown + (optional) intensity slider. Output: audio player + F0 contour plot. Runs on CPU (slower OK) | 6 |
| S8.2 | Demo: "same text, switch emotion" mode | Button to regenerate same text with all 4 emotions side-by-side | 3 |
| S8.3 | Demo acceptance test | Demo runs from fresh clone with documented setup. Response time <30s on CPU for one sentence | 2 |
| S8.4 | **Stretch A: Intensity embedding** | If System C is stable: add intensity scalar (0.0–1.0) as multiplicative scaling on emotion embedding. Re-train briefly. Generate intensity-graded samples | 6 |
| S8.5 | **Stretch B: Frame-level prosody (C2)** | If C1 utterance-level is stable and time permits: attempt frame-level F0/energy prediction using VITS MAS alignment. This is the original ADR-3 fallback path, now attempted as upside only | 6 |
| S8.6 | Deep analysis: write "Results" narrative | Interpret all tables and figures. Key questions: Does A→B show emotion conditioning? Does B→C shift prosody distributions? How large is the A0→A domain-adaptation effect? Does SER proxy agree or disagree with human listeners (and why)? | 6 |
| S8.7 | Failure analysis | Document: (a) which emotions are hardest, (b) naturalness vs. emotion fidelity trade-off, (c) where SER proxy diverges from human judgments (expected due to label mismatch) | 3 |
| S8.8 | Generate final "figure panel" for report | Publication-quality multi-panel figure: same text in 4 emotions from System C, showing waveform + F0 + spectrogram | 3 |
| S8.9 | Polish repo: README + inline docs | README has: project description, setup, quick start, reproduction commands, license table | 3 |
| S8.10 | GitHub Actions CI | Workflow: `pip install` → lint (ruff) → `pytest tests/` → smoke inference (1 sample from System A0) | 3 |

**Exit criteria for S8:**
- [ ] `python demo/app.py` launches Gradio interface; audio plays
- [ ] All figures for final report are in `figures/` (publication quality)
- [ ] Analysis narrative drafted
- [ ] README is comprehensive
- [ ] CI passes on push

---

### Sprint S9: Final Report & Presentation (May 7–13)

**Goal:** Submit final report, slides, source code package. Everything polished.

| ID | Task | Acceptance Criteria | Est. Hours |
|---|---|---|---|
| S9.1 | Write Final Report: Abstract + Intro | 1.5 pages. Frame contribution: emotion control layer + prosody supervision + rigorous ablation chain (A0/A/B/C) + fair evaluation | 4 |
| S9.2 | Write Final Report: Problem Definition | 1 page. Formal problem statement, notation, emotion classes, dataset description, single-speaker rationale | 3 |
| S9.3 | Write Final Report: Method | 2–3 pages. Architecture diagram, System A0/A/B/C descriptions, training procedure (freeze strategy, warm-start chain), utterance-level prosody supervision | 5 |
| S9.4 | Write Final Report: Experiments | 3–4 pages. All tables (MOS, emotion accuracy, ablation chain A0→A→B→C, prosody stats), all figures (distributions, confusion matrices, prosody panels), statistical tests. SER clearly labeled as auxiliary proxy with label mismatch caveat | 6 |
| S9.5 | Write Final Report: Discussion + Conclusion | 1.5 pages. Limitations (single speaker, utterance-level prosody, SER incompatibility, small listener pool), failure analysis, future work (multi-speaker, frame-level prosody, larger datasets), broader impact | 3 |
| S9.6 | Write Final Report: References | IEEE-style, ≥15 refs including all from PRD Section 14 | 2 |
| S9.7 | Write Final Report: Author Contributions + Code Provenance | Table with each member's specific contributions. **Separately label: (a) open-source reused as-is, (b) open-source modified, (c) original code, (d) original experiment design.** This directly addresses implementation-quality scoring | 2 |
| S9.8 | Internal report review + revision | Full read-through, fix consistency, check all figures referenced, verify P.808 wording is "P.808-inspired" throughout, verify SER is called "proxy" throughout | 3 |
| S9.9 | Prepare presentation slides | 12–15 slides: motivation → fair baseline design (A0/A/B/C) → method → results → demo video/live → limitations → conclusion. 15-min target | 5 |
| S9.10 | Record/rehearse demo for presentation | Screen recording as backup if live demo fails | 2 |
| S9.11 | Package source code | `make package` or zip: code + configs + README + NOTICE + requirements. Exclude data + checkpoints (too large) but include DVC manifest files | 2 |
| S9.12 | Final repo cleanup | Remove debug files, ensure `.gitignore` is correct, tag release `v1.0` | 1 |

**Final Report target structure (revised):**
```
1. Abstract (200 words)
2. Introduction (1.5 pp) — problem, motivation, contribution summary
3. Related Work (1 pp) — VITS, StyleTTS2, emotional TTS, evaluation standards
4. Problem Definition (1 pp) — formal, with notation
5. Method (2.5 pp)
   5.1 System A0: Raw pretrained VITS (reference)
   5.2 System A: Domain-adapted baseline (no emotion labels)
   5.3 System B: Emotion embedding conditioning
   5.4 System C: Utterance-level prosody-aware auxiliary supervision
   5.5 Training procedure (freeze strategy, warm-start chain, hyperparameters)
6. Experiments (3.5 pp)
   6.1 Experimental setup (data, single-speaker rationale, metrics, fair baselines)
   6.2 Ablation chain: A0 → A → B → C
   6.3 Automatic evaluation results (prosody stats — primary; SER proxy — auxiliary)
   6.4 Human evaluation results (MOS, emotion recognition, P.808-inspired protocol)
   6.5 Qualitative analysis (spectrograms, prosody contours)
7. Discussion (1 pp) — limitations, failure modes, SER label mismatch, future work
8. Conclusion (0.5 pp)
9. Code Provenance (open-source reused / modified / original code / original experiment design)
10. Author Contributions
11. References (≥15)
Total: ~12–14 pages
```

**Exit criteria for S9:**
- [ ] Final report PDF submitted
- [ ] Slides PDF/PPTX submitted
- [ ] Source code package submitted
- [ ] Demo runs on presentation machine (tested)
- [ ] Git tag `v1.0` pushed

---

## Part III — Cross-Cutting Concerns

### 1. Risk Registry with Sprint-Level Triggers

| Risk | Probability | Impact | Detection Sprint | Trigger Condition | Mitigation Action |
|---|---|---|---|---|---|
| R1: Coqui package distribution confusion | **High** | High | **S0 (Day 1)** | Import failures, missing checkpoints, Python version mismatch between `TTS` and `coqui-tts` | S0.0 dependency spike resolves this before anything else. Document findings in `docs/dependency_spike.md` |
| R2: Coqui VITS model modification too complex | Medium | High | S2 (Wed) | Cannot inject embedding into forward pass after 10h effort | Activate Fallback B-alt: overload speaker embedding as emotion embedding (zero model surgery) |
| R3: Training does not converge (System A or B) | Medium | High | S2/S3 (Mon) | Val loss not decreasing after 30 epochs | Lower LR to 5e-5; unfreeze more layers; reduce to 2 emotions |
| R4: EmoV-DB speaker coverage gap | Medium | Medium | S1 | Chosen speaker lacks one emotion, or sample count too low (<100 per emotion) | Switch to alternative speaker; if none have full coverage, merge 2 closest speakers and add speaker conditioning |
| R5: Training takes too long (GPU hours) | Medium | High | S2/S4 | Single Colab session times out (4h free tier) before convergence | Save checkpoints to Drive every N epochs; resume in next session. Use fp16; reduce epochs; subset data further. Consider Colab Pro if free tier insufficient |
| R6: Not enough human eval participants | Medium | Medium | S7 (Mon-Wed) | <10 responses by Wednesday | Extend deadline; reduce stimuli per listener further; add incentive |
| R7: SER probe gives random results | **Expected** | Low | S6 | Proxy agreement ≈ chance | Expected due to label mismatch. This is why it's auxiliary-only. Rely on human eval + prosody stats as primary evidence |
| R8: Audio quality collapse in fine-tuned models | Medium | High | S2/S3/S5 | MOS-proxy < 2.0 or unintelligible output | Freeze more layers; reduce fine-tuning epochs; increase LR warmup; A0 always available as quality reference |
| R9: EmoV-DB download issues | Low | High | S0 | Mirror links broken | Use alternative OpenSLR mirror; manually download from archived sources |
| R10: Loudness normalization erases emotional energy signal | Low | Medium | S1 | Training audio has flat energy across emotions | Confirm conservative normalization policy is applied (peak-norm only, not LUFS). If energy variation is absent in raw data, note as dataset limitation |

### 2. Quality Gates (Go/No-Go per Sprint) — Sharpened

| Sprint | Gate | Pass Condition | Fail Action |
|---|---|---|---|
| S0 | **Dependency gate** (BLOCKING) | Dependency spike passes: chosen package imports, pretrained model loads, fine-tuning imports work | Cannot proceed AT ALL until this passes. If `TTS` fails, try `coqui-tts`. If both fail, escalate to finding alternative VITS implementation |
| S0 | Env gate | Pretrained inference produces audio | Debug environment |
| S1 | **Data QA gate** (BLOCKING) | Data QA complete (coverage verified, clip/text histograms, no bad files). Core speaker selected with 4-emotion coverage | **Do not start model modification until this gate passes.** Data problems cascade into every downstream sprint |
| S2 | **Model gate** (Wed checkpoint) | EmotionVITS forward pass works with emotion embedding | **Immediately** switch to Fallback B-alt (speaker-embedding overload) or external conditioning wrapper. Do not burn more than 1 additional day |
| S3 | Training gate | System A and B loss decreasing; B audio quality acceptable | Reduce scope to 2 emotions; increase freeze ratio |
| S3 | **Midpoint gate** | Report submitted | Hard deadline — submit whatever is ready |
| S4 | **Prosody gate** (end of sprint) | Utterance-level C is stable and loss decreasing | **If stable: ship it. Do not start frame-level alignment.** Frame-level C2 is stretch-only (S8) |
| S5 | **Eval freeze gate** (BLOCKING) | 80+ samples locked with hash manifest. Loudness-normalized eval copies created. Texts are from `eval_holdout.csv` (not training, not canary) | No proceeding to human eval without frozen, properly sourced samples |
| S7 | **Human eval gate** | ≥15 responses collected | Extend by 3 days; reduce sample count if needed |
| S9 | **Submission gate** | Report + code + slides packaged | Hard deadline |

### 3. Definition of Done (per task)

A task is "done" when:
1. Code is committed to a feature branch and merged to main
2. Relevant tests pass (`pytest tests/`)
3. MLflow run logged (for training/inference tasks)
4. Output artifacts exist in expected locations
5. README updated if new setup steps needed

### 4. Key Metrics Dashboard (tracked in MLflow)

| Metric | Tracked From | Target | Alert If |
|---|---|---|---|
| `train/loss_total` (System A) | S2 | Decreasing | Plateau before epoch 10 |
| `train/loss_total` (System B) | S3 | Decreasing | Plateau before epoch 20 |
| `train/loss_f0_stats` | S4 | Decreasing | > 2× initial value (diverging) |
| `train/loss_energy_stats` | S4 | Decreasing | > 2× initial value |
| `val/loss_total` | S2/S3/S4 | Within 20% of train loss | Val >> Train (overfitting) |
| `eval/prosody_f0_separation` | S6 | F0 distributions differ across emotions (KS test p<0.05) | All emotions have same F0 distribution (conditioning not working) |
| `eval/ser_proxy_agreement` | S6 | Informational only (auxiliary) | Do not use as primary go/no-go metric |
| `eval/mos_mean` | S7 | > 3.0 for Systems A/B/C | < 2.5 (quality collapse) |
| `eval/emotion_acc_human` | S7 | > 50% for System C | < 30% |

### 5. Dependency Installation Order (revised)

**Local environment (coding + CPU tests):**
```bash
# Step 1: Core environment
conda create -n emotive-tts python=3.10 -y
conda activate emotive-tts

# Step 2: PyTorch CPU (for local debug runs and unit tests)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Step 3: Coqui TTS (version from spike)
pip install TTS==0.22.0   # OR: pip install coqui-tts==X.Y.Z

# Step 4: Infrastructure
pip install hydra-core==1.3.2 mlflow==2.12.1 dvc==3.50.0

# Step 5: Evaluation
pip install speechbrain librosa matplotlib seaborn pandas

# Step 6: Demo
pip install gradio==4.44.0

# Step 7: Dev tools
pip install pytest ruff pre-commit
```

**Google Colab (GPU compute) — first cell of `colab_pipeline.ipynb`:**
```python
# Cell 1: Setup
from google.colab import drive
drive.mount('/content/drive')

# Clone/pull repo
!git clone https://github.com/<YOUR_ORG>/COMP3065_Project.git /content/project 2>/dev/null || \
    (cd /content/project && git pull)

# Install dependencies (Colab already has PyTorch + CUDA)
!pip install -q TTS==0.22.0 hydra-core==1.3.2 mlflow speechbrain librosa

# Add project to path
import sys
sys.path.insert(0, '/content/project')

# Verify
from TTS.api import TTS
print('TTS OK')
from TTS.tts.configs.vits_config import VitsConfig
print('VitsConfig OK')

# Set checkpoint/data paths on Drive
DRIVE_ROOT = '/content/drive/MyDrive/COMP3065'
DATA_DIR = f'{DRIVE_ROOT}/data'
CKPT_DIR = f'{DRIVE_ROOT}/checkpoints'
OUTPUT_DIR = f'{DRIVE_ROOT}/outputs'
```

### 6. Weekly Standup Template

```markdown
## Sprint S[N] Standup — [Date]

### Completed since last standup
- [task ID]: [brief description]

### In progress
- [task ID]: [status, blockers]

### Blocked
- [task ID]: [blocker description, mitigation attempted]

### Risk updates
- [any new risks or status changes to existing risks]

### Quality gate status
- [gate name]: [pass/pending/blocked]

### Metrics snapshot
- Training loss: [value] (system: [A/B/C])
- Samples generated: [count]
- Tests passing: [count/total]
```

---

## Part IV — Summary Decision Table (revised)

| Decision | Choice | Key Rationale |
|---|---|---|
| Backbone | Coqui TTS (VITS), exact distribution resolved in S0.0 spike | PRD mandates it; fine-tuning CLI; pretrained 22kHz model. Distribution choice (TTS vs coqui-tts) empirically resolved before Day 2 |
| **Baseline hierarchy** | **A0 (raw pretrained) / A (domain fine-tuned, no emotion) / B (+emotion embedding) / C (+prosody loss)** | **Fixes causal attribution problem.** Without A, B/C gains are confounded by domain adaptation |
| Emotions | neutral, angry, amused, disgust (4) | Perceptually distinct; tractable confusion matrix; covers valence range |
| Training dataset | EmoV-DB, **single speaker with complete 4-emotion coverage** | No license friction; avoids speaker-coverage confound; multi-speaker is stretch only |
| Eval dataset | RAVDESS (reference/benchmark only, NOT for trap/control stimuli) | CC BY-NC-SA; validated labels. Cross-domain traps would confound listener judgments |
| Emotion conditioning | Learnable embedding + add to encoder | Clean ablation; no reference audio needed at inference |
| **Prosody supervision** | **Utterance-level F0 + energy stats (C1, default)** | No frame-level alignment complexity. Frame-level (C2) is stretch in S8 |
| Freeze strategy | Freeze encoder + decoder; train embedding + duration predictor + flow | Balance quality preservation vs. new capability |
| **Loudness normalization** | **Conservative peak-norm (-1 dBFS) for training; EBU R128 (-23 LUFS) for eval playback copies** | Energy is part of the emotional signal — aggressive training normalization erases it |
| **Human eval** | P.808-**inspired** local listening test (Google Forms, 15-20 peers, 12-18 min) | Realistic for students; shorter form avoids fatigue; wording protects against methodological criticism |
| **Primary auto eval** | **Prosody statistics (F0/energy distributions, speaking rate)** | Directly measures what we claim to control |
| **SER probe** | **Auxiliary-only weak proxy (`ser_proxy_agreement`, NOT "accuracy")** | IEMOCAP labels (neu/ang/hap/sad) ≠ our classes; disgust unmapped; amused≈hap is loose. Model card warns against cross-domain use |
| Config | Hydra | PRD mandate; CLI overrides for ablations |
| Tracking | MLflow | PRD mandate; lightweight local server |
| Data versioning | DVC (manifest checksums only; remote if available) | Satisfies FR-5 without ceremony the team won't maintain |
| Demo | Gradio | Low boilerplate; audio I/O built-in |
| Intensity control | Stretch goal (Sprint 8) | Don't let it delay core A0/A/B/C |
| Frame-level prosody (C2) | Stretch goal (Sprint 8) | Ship utterance-level C1 first; frame-level only if stable |
| MUSHRA | Skip | 4-system comparison doesn't need it; ACR + emotion test sufficient |
| Docker | Skip unless time in S9 | Low priority vs. report quality |
| **Compute environment** | **Local coding + Google Colab T4 for GPU** | RTX 2050 too small for VITS; Colab T4 (15 GB) is free; notebook preserves execution log for rubric |
