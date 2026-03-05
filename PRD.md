Project Requirements Document (PRD)

Project: Compute-Efficient Emotionally Controlled TTS (Student-Feasible)
Version: 1.0 (Mar 5, 2026)

0) Executive summary

You will deliver an emotionally controlled Text-to-Speech system that is feasible on limited student compute by adapting a pretrained open TTS backbone (not training from scratch). The core contribution is lightweight emotion + intensity conditioning plus prosody-aware supervision, with rigorous evaluation (human + diagnostic) to maximize rubric score.

This PRD specifies: scope, functional/non-functional requirements, architecture, experiment plan, testing, deployment, open-source dependencies (code + data), and IEEE-style references.

1) Problem statement and motivation

Plain TTS can sound natural but often fails at emotion fidelity and controllability. Emotional expressiveness is tightly coupled with prosody (F0 / energy / duration), and naive “prompt only” control is often ambiguous. Your project targets a reproducible and compute-efficient approach that can still produce defensible experimental evidence and compelling visualizations.

2) Goals and success criteria
2.1 Primary goals (must hit)

Emotion controllability: Same text can be synthesized across emotions (and optionally intensity levels).

Emotion fidelity: Human listeners can identify the intended emotion above baseline.

Naturalness: MOS/ACR does not collapse compared with baseline.

Reproducibility: Experiments re-runnable from configs with logged artifacts and metrics.

2.2 Success metrics (reportable)

Human

MOS / ACR (speech naturalness) using ITU-T P.808 style crowdsourcing or structured local listening test.

Emotion recognition accuracy + confusion matrix (listener labels vs target).

Intensity rating (Likert) if intensity control is implemented.

Diagnostic (automatic)

F0 statistics by emotion; energy and duration statistics; distribution plots.

Consistency check: intelligibility proxy (ASR WER optional) or manual “failed cases” log.

3) Scope
3.1 In-scope deliverables

Three systems (A/B/C) + ablations + evaluation + demo + reports

System A (Baseline): pretrained TTS backbone → waveform (no explicit emotion control).

System B (Emotion-conditioned): text + emotion label embedding → waveform.

System C (Emotion + prosody-aware): System B + prosody auxiliary supervision (pitch/energy/duration) + optional intensity embedding.

Required artifacts

Dataset pipeline scripts + manifests

Training + inference scripts

Evaluation toolkit (listening test + plots + tables)

Demo (local web UI or CLI batch generation)

Midpoint report + Final report + Slides + Source code package

3.2 Out of scope (explicitly)

Training StyleTTS2 / diffusion / codec-LM from scratch (compute-heavy).

Large-scale dataset collection.

Claiming new SOTA.

End-to-end “human-level TTS” positioning (leave to paper discussion).

4) Users / stakeholders

Primary users: your coding agents + report/presentation lead

Evaluators: course instructor/TAs and peers in listening tests

Secondary: anyone reproducing your repo

5) Functional requirements
FR-1 Dataset ingestion & preprocessing

Description: Provide a deterministic pipeline to download/organize/preprocess the dataset(s).
Must include

Standard audio format (e.g., WAV; consistent sample rate)

Transcript normalization + label mapping

Train/val/test split frozen early

Output metadata (CSV/JSONL): audio_path, text, emotion, intensity(optional), speaker_id(optional)

Acceptance criteria

python -m data.prepare --config configs/data.yaml produces the same file lists on repeated runs

Summary stats by split/emotion printed and saved

Recommended tools

(Optional) Montreal Forced Aligner for alignment or trimming non-speech segments.

FR-2 Model training pipeline (A/B/C)

Description: Train and fine-tune using pretrained open backbone to reduce compute.
Backbone requirement

Use Coqui TTS as the main framework (stable fine-tuning workflows, pretrained models).

System A requirements

Load pretrained model and run inference on held-out text set.

System B requirements

Implement emotion embedding conditioning (discrete classes)

Condition the acoustic model on emotion

System C requirements

Add at least one prosody-related auxiliary objective:

pitch (F0), energy, duration (choose ≥2 if possible)

Acceptance criteria

For each system:

Train script runs end-to-end on one GPU (or CPU-fallback for debug)

Generates a fixed “demo pack” (same texts × emotions)

Logs config + metrics + checkpoint paths

Compute feasibility note
Coqui explicitly recommends fine-tuning pretrained TTS models for faster convergence and “reasonable results with only a couple of hours of data.”

FR-3 Inference & sample generation

Description: Provide deterministic synthesis for evaluation and demo.
Must include

Batch inference for a list of sentences

“Same text, different emotion” generation mode

Output naming convention and metadata export

Acceptance criteria

python -m inference.run --config configs/infer.yaml produces outputs/samples/* with a manifest CSV

FR-4 Evaluation harness

Description: Support both human and diagnostic evaluation.

FR-4.1 MOS/ACR (Naturalness)

Implement either:

ITU-T P.808-style crowdsourcing toolkit (if you can use AMT), or

a structured local ACR form that mirrors P.808 test principles.

Microsoft provides an open P.808 toolkit implementation (ACR/DCR/CCR).

FR-4.2 Emotion recognition test

Listeners choose perceived emotion among your class set

Export confusion matrix and accuracy

FR-4.3 Prosody diagnostics

Plot F0/energy contours and distributions by emotion

Compare A vs B vs C

Optional FR-4.4 MUSHRA
If you need finer discrimination among multiple systems, support MUSHRA using ITU-R BS.1534 guidance and/or webMUSHRA tooling.

Acceptance criteria

One command produces: tables/*.csv and figures/*.png from generated samples.

FR-5 Experiment tracking & reproducibility

Description: Every run must be traceable to data + code + config.

Must include

Config system (Hydra) with CLI overrides for sweeps.

Run tracking (MLflow) for metrics/artifacts.

Data/model versioning (DVC) for dataset + checkpoints (or at minimum, dataset hash manifests).

Seed control and determinism guidelines per PyTorch reproducibility notes.

Acceptance criteria

For each reported result, you can point to:
git commit + dvc revision + mlflow run id + config file.

FR-6 Demo requirement

Description: A working demonstration for presentation day.

Minimum demo

Local web UI or CLI that:

accepts text + emotion (+ intensity)

generates waveform(s)

lets users switch emotions for the same text

Acceptance criteria

Demo runs on a laptop CPU (slower is OK) or a single GPU machine with documented setup.

6) Non-functional requirements
NFR-1 Compute constraints

Must run training/fine-tuning on single consumer GPU or university lab GPU

Provide “small-run config” for quick iteration (e.g., 1 speaker, 2 emotions, few epochs)

NFR-2 Reproducibility

Follow PyTorch reproducibility guidance: seeds, deterministic ops when needed.

NFR-3 Licensing & attribution compliance

All open-source code and data must be cited and license-checked.

Repo must clearly label:

“Upstream open-source code used as-is”

“Modified open-source components”

“Original code written by your team”

NFR-4 Maintainability

Standard repo layout + lint/format + unit tests for utilities

Minimal “one-command” scripts

7) System architecture
7.1 High-level components

Data pipeline → normalized audio + transcripts + labels

Training pipeline → A/B/C models

Inference pipeline → sample packs + demo outputs

Evaluation pipeline → listening tests + diagnostics + plots

Tracking layer → Hydra + MLflow + DVC

7.2 Recommended tech stack (best practice)

Core TTS framework: Coqui TTS (training + fine-tuning)

Configuration: Hydra

Tracking: MLflow

Versioning: DVC

Alignment (optional): Montreal Forced Aligner

Listening tests: Microsoft P.808 toolkit (ACR/CCR/DCR)

MUSHRA (optional): ITU-R BS.1534 + webMUSHRA

8) Experiment design (what you must run)
8.1 Baselines & proposed system

A: pretrained / fine-tuned baseline (no explicit emotion)

B: + emotion embedding

C: + prosody supervision (and optional intensity embedding)

8.2 Required ablations (minimum)

B vs C (remove prosody loss)

C with/without intensity embedding (if implemented)

8.3 Reporting requirements

Tables:

MOS/ACR by system

Emotion recognition accuracy by system

Plots:

confusion matrix

F0/energy distributions by emotion

same-text multi-emotion waveform/prosody examples (figure panel)

9) Testing strategy
9.1 Code tests

Unit tests: config parsing, metadata generation, audio IO

Integration tests: one mini-run (few samples) training/inference end-to-end

9.2 Model regression tests

Keep a frozen “canary text set” and compare:

audio duration bounds

loudness bounds

sanity check: no NaNs in generated audio

9.3 Evaluation validation

Ensure the listening test randomizes sample order and includes controls (hidden reference / traps if using MUSHRA/P.808 patterns).

10) Deployment & packaging
Minimal (required)

Local run instructions + environment file

Optional Docker for inference only (CPU)

Bonus (if time)

Host inference demo on a simple local web server for presentation.

11) Risks & mitigations (score-oriented)

Emotion not perceptible → prioritize strong emotion sets; add prosody supervision; show same-text comparisons.

Naturalness drops → freeze more layers; lower LR; shorten training.

Dataset license friction → choose a dataset with clean access (see Section 12).

Evaluation too weak → lock in MOS + emotion recognition early and run pilot tests.

12) Necessary open-source code (with links)

Below are the required and optional open-source codebases you should use.

Required (recommended)

Coqui TTS (training/fine-tuning, pretrained backbones) — MPL-2.0

Hydra (config & sweeps) — MIT

MLflow (experiment tracking) — Apache-2.0

DVC (data/model versioning, experiments) — Apache-2.0

PyTorch reproducibility guidance (engineering best practice)

Strongly recommended (evaluation)

Microsoft P.808 toolkit (crowdsourced MOS/ACR/DCR/CCR workflows) — MIT

Optional (only if needed)

webMUSHRA (web-based MUSHRA framework; check its license terms)

Montreal Forced Aligner (alignment / trimming / segmentation) — MIT

HiFi-GAN (vocoder experiments; keep fixed in main comparisons) — MIT

StyleTTS2 (stretch goal only; training complexity) — MIT

Reference VITS repo (if implementing outside Coqui)

Copy-paste link bundle (URLs in code block)
Coqui TTS:              https://github.com/coqui-ai/TTS
Coqui fine-tuning docs: https://docs.coqui.ai/en/latest/finetuning.html
Hydra:                  https://github.com/facebookresearch/hydra
MLflow:                 https://github.com/mlflow/mlflow
DVC:                    https://github.com/iterative/dvc
PyTorch reproducibility:https://docs.pytorch.org/docs/stable/notes/randomness.html
P.808 toolkit:          https://github.com/microsoft/P.808
webMUSHRA:              https://github.com/audiolabs/webMUSHRA
MFA:                    https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner
HiFi-GAN:               https://github.com/jik876/hifi-gan
StyleTTS2:              https://github.com/yl4579/StyleTTS2
VITS paper (ICML):      https://proceedings.mlr.press/v139/kim21f.html
13) Necessary open-source data (with access + license notes)
Dataset priority (best practice)

Choose one primary training dataset + one secondary evaluation dataset.

A) Recommended primary training dataset (pick 1)

Option A1: EmoV-DB (OpenSLR SLR115)

Pros: smaller, emotion-focused, easier for student fine-tuning

License: non-commercial purposes (research/teaching/publication/personal experimentation)

Download: OpenSLR mirrors + repo instructions

Option A2: ESD (Emotional Speech Dataset)

Pros: parallel utterances, multi-speaker, 5 emotions; good for controllability experiments

Important: distributed for research purpose and requires completing a license form / contacting the authors

B) Recommended secondary evaluation dataset (pick 1)

Option B1: CREMA-D

Pros: large, well-validated emotion labels; provides intensity levels

License: Open Database License (ODbL) + Database Contents License (DbCL) per repo

Access: official Git LFS repo; TFDS loader also exists

Option B2: RAVDESS

Pros: validated emotion + intensity; easy to cite; common benchmark

License: CC BY-NC-SA 4.0 (per Zenodo record)

Copy-paste data link bundle
EmoV-DB (OpenSLR): https://openslr.org/115/
EmoV-DB repo:      https://github.com/numediart/EmoV-DB
ESD dataset repo:  https://github.com/HLTSingapore/Emotional-Speech-Data
ESD landing page:  https://hltsingapore.github.io/ESD/index.html
CREMA-D repo:      https://github.com/CheyneyComputerScience/CREMA-D
CREMA-D paper:     https://pubmed.ncbi.nlm.nih.gov/25653738/
RAVDESS (Zenodo):  https://zenodo.org/records/1188976
14) Paper references (IEEE style)

Use these in your Final Report “References” section.

Core TTS / Vocoder

[1] J. Kim, J. Kong, and J. Son, “Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech,” in Proc. ICML, 2021, pp. 5530–5540.

[2] Y. A. Li, C. Han, V. S. Raghavan, G. Mischler, and N. Mesgarani, “StyleTTS 2: Towards Human-Level Text-to-Speech through Style Diffusion and Adversarial Training with Large Speech Language Models,” arXiv preprint arXiv:2306.07691, 2023.

[3] J. Kong, J. Kim, and J. Bae, “HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis,” arXiv preprint arXiv:2010.05646, 2020.

Datasets

[4] K. Zhou, B. Şişman, R. Liu, and H. Li, “Seen and Unseen Emotional Style Transfer for Voice Conversion with A New Emotional Speech Dataset,” in Proc. IEEE ICASSP, 2021, pp. 920–924, doi: 10.1109/ICASSP39728.2021.9413391.

[5] A. Adigwe, N. Tits, K. El Haddad, S. Ostadabbas, and T. Dutoit, “The Emotional Voices Database: Towards Controlling the Emotion Dimension in Voice Generation Systems,” arXiv preprint arXiv:1806.09514, 2018.

[6] H. Cao, D. G. Cooper, M. K. Keutmann, R. C. Gur, A. Nenkova, and R. Verma, “CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset,” IEEE Trans. Affective Computing, vol. 5, no. 4, pp. 377–390, 2014, doi: 10.1109/TAFFC.2014.2336244.

[7] S. R. Livingstone and F. A. Russo, “The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English,” PLOS ONE, vol. 13, no. 5, p. e0196391, 2018, doi: 10.1371/journal.pone.0196391.

Evaluation standards & tooling

[8] ITU-T Recommendation P.808, “Subjective evaluation of speech quality with a crowdsourcing approach,” Jun. 2021.

[9] ITU-R Recommendation BS.1534-3, “Method for the subjective assessment of intermediate quality level of audio systems (MUSHRA),” Oct. 2015.

[10] B. Naderi and R. Cutler, “An Open Source Implementation of ITU-T Recommendation P.808 with Validation,” in Proc. INTERSPEECH, 2020.

[11] M. Schöffler et al., “webMUSHRA—A Comprehensive Framework for Web-based Listening Tests,” Journal of Open Research Software, vol. 6, no. 1, p. 8, 2018, doi: 10.5334/jors.187.

Alignment / infrastructure

[12] M. McAuliffe, M. Socolof, S. Mihuc, M. Wagner, and M. Sonderegger, “Montreal Forced Aligner: Trainable Text-Speech Alignment Using Kaldi,” in Proc. INTERSPEECH, 2017, pp. 498–502, doi: 10.21437/Interspeech.2017-1386.

[13] DVC, “DVC Experiments Overview,” documentation, accessed 2026.

[14] MLflow, “MLflow Tracking Quickstart,” documentation, accessed 2026.

[15] PyTorch, “Reproducibility,” documentation, updated Oct. 2025.