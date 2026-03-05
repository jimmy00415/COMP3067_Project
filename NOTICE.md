# NOTICE — Third-Party Licenses

This project uses the following open-source software:

| Component | License | Purpose | Modified? |
|---|---|---|---|
| [Coqui TTS](https://github.com/coqui-ai/TTS) | MPL-2.0 | VITS backbone, pretrained model, fine-tuning | Yes — EmotionVITS extends VITS with emotion embedding + prosody heads |
| [PyTorch](https://pytorch.org/) | BSD-3-Clause | Deep learning framework | No |
| [torchaudio](https://pytorch.org/audio/) | BSD-2-Clause | Audio I/O and transforms | No |
| [librosa](https://librosa.org/) | ISC | F0 extraction (pYIN), audio analysis | No |
| [SpeechBrain](https://speechbrain.github.io/) | Apache-2.0 | SER probe (wav2vec2-IEMOCAP, auxiliary only) | No |
| [Hydra](https://hydra.cc/) | MIT | Configuration management | No |
| [MLflow](https://mlflow.org/) | Apache-2.0 | Experiment tracking | No |
| [DVC](https://dvc.org/) | Apache-2.0 | Data version control (manifest checksums) | No |
| [Gradio](https://gradio.app/) | Apache-2.0 | Demo web UI | No |
| [matplotlib](https://matplotlib.org/) | PSF-based | Plotting | No |
| [seaborn](https://seaborn.pydata.org/) | BSD-3-Clause | Statistical visualization | No |
| [pandas](https://pandas.pydata.org/) | BSD-3-Clause | Data manipulation | No |
| [soundfile](https://python-soundfile.readthedocs.io/) | BSD-3-Clause | Audio file I/O | No |

## Datasets

| Dataset | License | Usage |
|---|---|---|
| [EmoV-DB](https://github.com/numediart/EmoV-DB) | Open (research) | Primary training dataset |
| [RAVDESS](https://zenodo.org/record/1188976) | CC BY-NC-SA 4.0 | Reference/benchmark only (NOT for trap/control stimuli) |
| [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) | Public Domain | Pretrained VITS checkpoint (via Coqui) |

## Code Provenance

| Category | Description |
|---|---|
| **Open-source reused as-is** | Coqui TTS training loop, SpeechBrain SER inference, librosa pYIN |
| **Open-source modified** | Coqui VITS model class → EmotionVITS (emotion embedding injection, prosody heads) |
| **Original code** | Data pipeline, EmotionVITS modifications, prosody heads, evaluation harness, Colab notebook, demo |
| **Original experiment design** | A0/A/B/C ablation hierarchy, utterance-level prosody supervision, P.808-inspired listening test |
