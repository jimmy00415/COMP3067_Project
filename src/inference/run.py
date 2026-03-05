"""Inference module for batch audio generation (FR-3).

Generates speech from canary texts × all emotions for each system variant.
Supports Systems A0, A, B, C. Outputs to data/processed/eval_stimuli/.

Usage:
    python -m src.inference.run --config configs/infer.yaml

Or from Colab:
    from src.inference.run import run_inference
    run_inference(cfg)
"""

import logging
import time
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import soundfile as sf

from src.data.utils import (
    EMOTION_MAP,
    EMOTION_LABELS,
    load_canary_texts,
    peak_normalize,
    lufs_normalize,
    file_hash,
)
from src.models.baseline import BaselineSynthesizer, create_system_a0, create_system_a
from src.models.emotion_vits import EmotionVITS, build_emotion_vits
from src.models.prosody_heads import build_prosody_heads

logger = logging.getLogger(__name__)


def synthesize_system_a0(
    texts: list[dict],
    output_dir: str | Path,
    use_cuda: bool = False,
) -> list[dict]:
    """Generate audio with System A0 (pretrained LJSpeech, no fine-tuning).

    A0 has no emotion conditioning — generates neutral speech for all
    emotion conditions. This is the reference baseline.

    Args:
        texts: List of canary text dicts with 'id' and 'text' keys.
        output_dir: Output directory for WAV files.
        use_cuda: Whether to use GPU.

    Returns:
        List of generation metadata dicts.
    """
    output_dir = Path(output_dir) / "system_a0"
    output_dir.mkdir(parents=True, exist_ok=True)

    synth = create_system_a0()
    synth.use_cuda = use_cuda
    synth.load()

    results = []
    for text_info in texts:
        text_id = text_info["id"]
        text = text_info["text"]

        # A0 generates the same audio regardless of emotion label
        # but we save copies for each emotion to enable fair comparison
        wav, sr = synth.synthesize(text)
        wav = np.array(wav)

        for emotion in EMOTION_LABELS:
            fname = f"a0_text{text_id:02d}_{emotion}.wav"
            fpath = output_dir / fname

            # Apply peak normalization
            wav_norm = peak_normalize(wav, target_db=-1.0)
            sf.write(str(fpath), wav_norm, sr)

            # LUFS version for listening test
            lufs_dir = output_dir / "lufs"
            lufs_dir.mkdir(exist_ok=True)
            wav_lufs = lufs_normalize(wav, sr, target_lufs=-23.0)
            sf.write(str(lufs_dir / fname), wav_lufs, sr)

            results.append({
                "system": "A0",
                "text_id": text_id,
                "text": text,
                "emotion": emotion,
                "file_path": str(fpath),
                "lufs_path": str(lufs_dir / fname),
                "sha256": file_hash(fpath),
            })

    logger.info(f"System A0: Generated {len(results)} audio files")
    return results


def synthesize_system_a(
    texts: list[dict],
    checkpoint_path: str,
    config_path: str,
    output_dir: str | Path,
    use_cuda: bool = False,
) -> list[dict]:
    """Generate audio with System A (fine-tuned, no emotion labels).

    Like A0, System A has no emotion conditioning. The only difference
    is domain adaptation on EmoV-DB data.

    Args:
        texts: Canary text list.
        checkpoint_path: Path to System A checkpoint.
        config_path: Path to model config.
        output_dir: Output directory.
        use_cuda: GPU flag.

    Returns:
        Generation metadata list.
    """
    output_dir = Path(output_dir) / "system_a"
    output_dir.mkdir(parents=True, exist_ok=True)

    synth = create_system_a(checkpoint_path, config_path, use_cuda=use_cuda)
    synth.load()

    results = []
    for text_info in texts:
        text_id = text_info["id"]
        text = text_info["text"]

        wav, sr = synth.synthesize(text)
        wav = np.array(wav)

        for emotion in EMOTION_LABELS:
            fname = f"a_text{text_id:02d}_{emotion}.wav"
            fpath = output_dir / fname

            wav_norm = peak_normalize(wav, target_db=-1.0)
            sf.write(str(fpath), wav_norm, sr)

            lufs_dir = output_dir / "lufs"
            lufs_dir.mkdir(exist_ok=True)
            wav_lufs = lufs_normalize(wav, sr, target_lufs=-23.0)
            sf.write(str(lufs_dir / fname), wav_lufs, sr)

            results.append({
                "system": "A",
                "text_id": text_id,
                "text": text,
                "emotion": emotion,
                "file_path": str(fpath),
                "lufs_path": str(lufs_dir / fname),
                "sha256": file_hash(fpath),
            })

    logger.info(f"System A: Generated {len(results)} audio files")
    return results


def synthesize_emotion_system(
    texts: list[dict],
    system: str,
    checkpoint_path: str,
    output_dir: str | Path,
    use_cuda: bool = False,
    num_emotions: int = 4,
    embedding_dim: int = 192,
    noise_scale: float = 0.667,
    length_scale: float = 1.0,
) -> list[dict]:
    """Generate audio with System B or C (emotion-conditioned).

    Args:
        texts: Canary text list.
        system: "B" or "C".
        checkpoint_path: Path to checkpoint.
        output_dir: Output directory.
        use_cuda: GPU flag.
        noise_scale: VITS noise scale.
        length_scale: VITS duration scale.

    Returns:
        Generation metadata list.
    """
    system = system.upper()
    output_dir = Path(output_dir) / f"system_{system.lower()}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build model
    prosody_heads = None
    if system == "C":
        prosody_heads = build_prosody_heads(embedding_dim)

    model = build_emotion_vits(
        system=system,
        checkpoint_path=checkpoint_path,
        use_cuda=use_cuda,
        num_emotions=num_emotions,
        embedding_dim=embedding_dim,
        prosody_heads=prosody_heads,
    )
    model.eval()

    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")

    results = []
    for text_info in texts:
        text_id = text_info["id"]
        text = text_info["text"]

        for emotion in EMOTION_LABELS:
            emotion_id = EMOTION_MAP[emotion]
            fname = f"{system.lower()}_text{text_id:02d}_{emotion}.wav"
            fpath = output_dir / fname

            try:
                # Tokenize text using the VITS tokenizer
                if hasattr(model.vits, 'tokenizer'):
                    tokens = model.vits.tokenizer.text_to_ids(text)
                else:
                    # Fallback: character-level tokenization
                    tokens = [ord(c) for c in text.lower()]

                x = torch.LongTensor([tokens]).to(device)
                emotion_tensor = torch.LongTensor([emotion_id]).to(device)

                with torch.no_grad():
                    wav_tensor = model.infer(
                        x=x,
                        emotion_ids=emotion_tensor,
                        noise_scale=noise_scale,
                        length_scale=length_scale,
                    )

                wav = wav_tensor.squeeze().cpu().numpy()
                sr = 22050  # VITS default

                # Peak normalize
                wav_norm = peak_normalize(wav, target_db=-1.0)
                sf.write(str(fpath), wav_norm, sr)

                # LUFS version
                lufs_dir = output_dir / "lufs"
                lufs_dir.mkdir(exist_ok=True)
                wav_lufs = lufs_normalize(wav, sr, target_lufs=-23.0)
                sf.write(str(lufs_dir / fname), wav_lufs, sr)

                results.append({
                    "system": system,
                    "text_id": text_id,
                    "text": text,
                    "emotion": emotion,
                    "file_path": str(fpath),
                    "lufs_path": str(lufs_dir / fname),
                    "sha256": file_hash(fpath),
                })

            except Exception as e:
                logger.error(f"Failed to synthesize {fname}: {e}")
                results.append({
                    "system": system,
                    "text_id": text_id,
                    "text": text,
                    "emotion": emotion,
                    "error": str(e),
                })

    logger.info(f"System {system}: Generated {len(results)} audio files")
    return results


def run_inference(cfg: dict) -> dict:
    """Run full inference pipeline for all systems.

    Args:
        cfg: Inference configuration dict (from infer.yaml).

    Returns:
        Dict with all generation results and manifest path.
    """
    import pandas as pd

    infer_cfg = cfg.get("inference", cfg)

    # Accept both "systems" (list) and "system" (string) from config
    systems = infer_cfg.get("systems", None)
    if systems is None:
        s = infer_cfg.get("system", "A0")
        systems = [s] if isinstance(s, str) else list(s)
    # Always generate for all 4 systems for evaluation
    if len(systems) == 1 and infer_cfg.get("batch", {}).get("enabled", False):
        systems = ["A0", "A", "B", "C"]

    output_dir = infer_cfg.get("output_dir", "data/processed/eval_stimuli")
    canary_path = infer_cfg.get("canary_texts", "configs/canary_texts.txt")
    use_cuda = infer_cfg.get("use_cuda", torch.cuda.is_available())

    # Load canary texts
    texts = load_canary_texts(canary_path)
    logger.info(f"Loaded {len(texts)} canary texts")

    all_results = []

    for system in systems:
        system = system.upper()
        logger.info(f"--- Generating for System {system} ---")

        if system == "A0":
            results = synthesize_system_a0(texts, output_dir, use_cuda=use_cuda)

        elif system == "A":
            ckpt = infer_cfg.get("system_a_checkpoint", "checkpoints/system_a/best.pth")
            config = infer_cfg.get("system_a_config", "checkpoints/system_a/config.json")
            results = synthesize_system_a(texts, ckpt, config, output_dir, use_cuda=use_cuda)

        elif system in ("B", "C"):
            ckpt_key = f"system_{system.lower()}_checkpoint"
            ckpt = infer_cfg.get(ckpt_key, f"checkpoints/system_{system.lower()}/best.pth")
            results = synthesize_emotion_system(
                texts, system, ckpt, output_dir,
                use_cuda=use_cuda,
                noise_scale=infer_cfg.get("noise_scale", 0.667),
                length_scale=infer_cfg.get("length_scale", 1.0),
            )
        else:
            logger.warning(f"Unknown system: {system}")
            continue

        all_results.extend(results)

    # Save eval manifest with file hashes (eval freeze — ADR-7)
    manifest_df = pd.DataFrame(all_results)
    manifest_path = Path(output_dir) / "eval_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    # Save as JSON too for notebook consumption
    json_path = Path(output_dir) / "eval_manifest.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"Inference complete. {len(all_results)} total files. Manifest: {manifest_path}")
    return {
        "total_files": len(all_results),
        "manifest_path": str(manifest_path),
        "systems": systems,
    }


def main():
    """CLI entry point."""
    import yaml
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Run EmotionVITS inference")
    parser.add_argument("--config", type=str, default="configs/infer.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    run_inference(cfg)


if __name__ == "__main__":
    main()
