"""EmoV-DB data preparation pipeline (FR-1).

Reads raw EmoV-DB, filters to core speaker, resamples, applies conservative
peak normalization (ADR-7), extracts F0/energy contours, computes utterance-level
prosody stats, and creates train/val/test splits.

Usage:
    python -m src.data.prepare --config configs/data.yaml

Or from Colab notebook:
    from src.data.prepare import prepare_dataset
    prepare_dataset(cfg)
"""

import os
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.data.utils import (
    EMOTION_MAP,
    EMOTION_LABELS,
    load_audio,
    save_audio,
    peak_normalize,
    lufs_normalize,
    extract_f0,
    extract_energy,
    compute_utterance_prosody_stats,
)

logger = logging.getLogger(__name__)


# --- EmoV-DB directory name to our emotion label mapping ---
EMOVDB_EMOTION_MAP = {
    "Neutral": "neutral",
    "Angry": "angry",
    "Amused": "amused",
    "Disgust": "disgust",
    "Sleepiness": None,  # Dropped — low perceptual distinctiveness
}

# Known EmoV-DB speakers
EMOVDB_SPEAKERS = ["bea", "jenie", "josh", "sam"]


def scan_emovdb(
    raw_dir: str | Path,
    target_emotions: list[str] | None = None,
) -> pd.DataFrame:
    """Scan EmoV-DB raw directory and build a manifest DataFrame.

    Args:
        raw_dir: Path to raw EmoV-DB directory.
        target_emotions: List of target emotions (default: our 4).

    Returns:
        DataFrame with columns: file_path, speaker, emotion, text
    """
    if target_emotions is None:
        target_emotions = EMOTION_LABELS

    raw_dir = Path(raw_dir)
    records = []

    for emotion_dir in sorted(raw_dir.iterdir()):
        if not emotion_dir.is_dir():
            continue
        dir_name = emotion_dir.name
        # Map EmoV-DB folder names to our labels
        our_emotion = EMOVDB_EMOTION_MAP.get(dir_name)
        if our_emotion is None or our_emotion not in target_emotions:
            logger.info(f"Skipping emotion dir: {dir_name}")
            continue

        for wav_file in sorted(emotion_dir.glob("*.wav")):
            # Parse speaker from filename
            # EmoV-DB naming: e.g., "bea_angry_001.wav" or similar
            fname = wav_file.stem.lower()
            speaker = None
            for spk in EMOVDB_SPEAKERS:
                if spk in fname:
                    speaker = spk
                    break
            if speaker is None:
                speaker = "unknown"

            # Try to find matching text file
            txt_file = wav_file.with_suffix(".txt")
            text = ""
            if txt_file.exists():
                text = txt_file.read_text(encoding="utf-8").strip()

            records.append({
                "file_path": str(wav_file),
                "speaker": speaker,
                "emotion": our_emotion,
                "text": text,
                "filename": wav_file.name,
            })

    df = pd.DataFrame(records)
    logger.info(f"Scanned {len(df)} files from {raw_dir}")
    return df


def audit_speaker_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Audit per-speaker emotion coverage (S1.1).

    Returns a pivot table: speakers × emotions with sample counts.
    """
    pivot = df.pivot_table(
        index="speaker",
        columns="emotion",
        values="file_path",
        aggfunc="count",
        fill_value=0,
    )
    return pivot


def select_core_speaker(
    df: pd.DataFrame,
    required_emotions: list[str] | None = None,
    min_per_emotion: int = 50,
    preferred_speaker: str | None = None,
) -> str:
    """Select core speaker with complete emotion coverage (ADR-7).

    Args:
        df: Full manifest DataFrame.
        required_emotions: Emotions that must be present.
        min_per_emotion: Minimum samples per emotion.
        preferred_speaker: If specified and valid, use this speaker.

    Returns:
        Speaker name string.

    Raises:
        ValueError: If no speaker has complete coverage.
    """
    if required_emotions is None:
        required_emotions = EMOTION_LABELS

    coverage = audit_speaker_coverage(df)
    logger.info(f"Speaker coverage:\n{coverage}")

    # Check each speaker
    valid_speakers = []
    for speaker in coverage.index:
        has_all = all(
            coverage.loc[speaker].get(e, 0) >= min_per_emotion
            for e in required_emotions
        )
        if has_all:
            total = sum(coverage.loc[speaker].get(e, 0) for e in required_emotions)
            valid_speakers.append((speaker, total))

    if not valid_speakers:
        # Relax min_per_emotion constraint
        logger.warning(
            f"No speaker has >={min_per_emotion} samples per emotion. "
            "Trying with min_per_emotion=1..."
        )
        for speaker in coverage.index:
            has_all = all(
                coverage.loc[speaker].get(e, 0) >= 1
                for e in required_emotions
            )
            if has_all:
                total = sum(coverage.loc[speaker].get(e, 0) for e in required_emotions)
                valid_speakers.append((speaker, total))

    if not valid_speakers:
        raise ValueError(
            f"No speaker has complete coverage for {required_emotions}. "
            f"Coverage:\n{coverage}"
        )

    if preferred_speaker and any(s == preferred_speaker for s, _ in valid_speakers):
        return preferred_speaker

    # Pick speaker with most total samples
    valid_speakers.sort(key=lambda x: x[1], reverse=True)
    chosen = valid_speakers[0][0]
    logger.info(f"Selected core speaker: {chosen} ({valid_speakers[0][1]} samples)")
    return chosen


def process_audio_file(
    input_path: str | Path,
    output_path: str | Path,
    sr: int = 22050,
    peak_db: float = -1.0,
    f0_fmin: float = 75.0,
    f0_fmax: float = 300.0,
    hop_length: int = 256,
    win_length: int = 1024,
) -> dict:
    """Process a single audio file: load, normalize, extract features.

    Args:
        input_path: Raw audio file path.
        output_path: Where to save processed audio.
        sr: Target sample rate.
        peak_db: Peak normalization target.
        f0_fmin/f0_fmax: Speaker-aware F0 range.

    Returns:
        Dict with prosody stats and metadata.
    """
    # Load and resample
    audio, _ = load_audio(input_path, sr=sr)

    # Conservative peak normalization (ADR-7)
    audio = peak_normalize(audio, target_db=peak_db)

    # Save processed audio
    save_audio(audio, output_path, sr=sr)

    # Extract F0
    f0, voiced = extract_f0(
        audio, sr=sr, fmin=f0_fmin, fmax=f0_fmax,
        hop_length=hop_length, win_length=win_length,
    )

    # Extract energy
    energy = extract_energy(audio, hop_length=hop_length, win_length=win_length)

    # Save feature arrays alongside audio
    out_dir = Path(output_path).parent
    stem = Path(output_path).stem
    np.save(out_dir / f"{stem}_f0.npy", f0)
    np.save(out_dir / f"{stem}_energy.npy", energy)
    np.save(out_dir / f"{stem}_voiced.npy", voiced)

    # Compute utterance-level stats (System C targets)
    stats = compute_utterance_prosody_stats(f0, energy)
    stats["duration"] = len(audio) / sr
    stats["num_frames"] = len(f0)
    stats["voiced_ratio"] = float(np.sum(voiced) / max(len(voiced), 1))

    return stats


def create_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    stratify_col: str = "emotion",
    eval_holdout_texts: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create stratified train/val/test splits + eval holdout.

    Args:
        df: Manifest DataFrame.
        train_ratio, val_ratio, test_ratio: Split ratios.
        seed: Random seed.
        stratify_col: Column to stratify by.
        eval_holdout_texts: Number of unique texts to reserve for eval.

    Returns:
        (train_df, val_df, test_df, eval_holdout_df)
    """
    from sklearn.model_selection import train_test_split

    # First, reserve eval holdout texts (never used in train/val/test)
    unique_texts = df["text"].unique()
    rng = np.random.RandomState(seed)
    if len(unique_texts) > eval_holdout_texts and eval_holdout_texts > 0:
        holdout_texts = rng.choice(unique_texts, size=eval_holdout_texts, replace=False)
        eval_holdout_df = df[df["text"].isin(holdout_texts)].copy()
        df_remaining = df[~df["text"].isin(holdout_texts)].copy()
    else:
        eval_holdout_df = pd.DataFrame(columns=df.columns)
        df_remaining = df.copy()

    # Stratified split
    adjusted_val = val_ratio / (train_ratio + val_ratio + test_ratio)
    adjusted_test = test_ratio / (train_ratio + val_ratio + test_ratio)

    train_df, temp_df = train_test_split(
        df_remaining,
        test_size=adjusted_val + adjusted_test,
        random_state=seed,
        stratify=df_remaining[stratify_col],
    )

    relative_test = adjusted_test / (adjusted_val + adjusted_test)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test,
        random_state=seed,
        stratify=temp_df[stratify_col],
    )

    logger.info(
        f"Splits: train={len(train_df)}, val={len(val_df)}, "
        f"test={len(test_df)}, eval_holdout={len(eval_holdout_df)}"
    )
    return train_df, val_df, test_df, eval_holdout_df


def prepare_dataset(cfg: dict) -> dict:
    """Full data preparation pipeline (FR-1).

    Args:
        cfg: Configuration dict (from data.yaml).

    Returns:
        Dict with summary statistics.
    """
    dataset_cfg = cfg.get("dataset", cfg)
    audio_cfg = cfg.get("audio", {})
    splits_cfg = cfg.get("splits", {})

    raw_dir = dataset_cfg.get("raw_dir", "data/raw/EmoV-DB")
    processed_dir = dataset_cfg.get("processed_dir", "data/processed/train")
    manifests_dir = dataset_cfg.get("manifests_dir", "data/manifests")
    core_speaker = dataset_cfg.get("core_speaker")
    sr = audio_cfg.get("sample_rate", 22050)
    peak_db = audio_cfg.get("peak_db", -1.0)
    f0_fmin = audio_cfg.get("f0_fmin", 75.0)
    f0_fmax = audio_cfg.get("f0_fmax", 300.0)
    hop_length = audio_cfg.get("hop_length", 256)
    win_length = audio_cfg.get("win_length", 1024)
    seed = splits_cfg.get("seed", 42)

    # Ensure output dirs exist
    Path(processed_dir).mkdir(parents=True, exist_ok=True)
    Path(manifests_dir).mkdir(parents=True, exist_ok=True)

    # Step 1: Scan raw data
    logger.info("Step 1: Scanning raw EmoV-DB...")
    full_df = scan_emovdb(raw_dir)

    if full_df.empty:
        raise RuntimeError(f"No audio files found in {raw_dir}")

    # Step 2: Speaker audit & selection
    logger.info("Step 2: Speaker audit...")
    coverage = audit_speaker_coverage(full_df)
    logger.info(f"Speaker × Emotion coverage:\n{coverage}")

    if core_speaker is None or core_speaker == "null":
        core_speaker = select_core_speaker(full_df)
    logger.info(f"Core speaker: {core_speaker}")

    # Filter to core speaker
    df = full_df[full_df["speaker"] == core_speaker].copy()
    logger.info(f"After filtering to speaker '{core_speaker}': {len(df)} files")

    # Step 3: Process all audio files
    logger.info("Step 3: Processing audio files...")
    all_stats = []
    for idx, row in df.iterrows():
        emotion = row["emotion"]
        fname = Path(row["file_path"]).stem
        out_path = Path(processed_dir) / emotion / f"{fname}.wav"

        stats = process_audio_file(
            row["file_path"],
            out_path,
            sr=sr,
            peak_db=peak_db,
            f0_fmin=f0_fmin,
            f0_fmax=f0_fmax,
            hop_length=hop_length,
            win_length=win_length,
        )
        stats["processed_path"] = str(out_path)
        stats["original_path"] = row["file_path"]
        stats["emotion"] = emotion
        stats["emotion_id"] = EMOTION_MAP[emotion]
        stats["text"] = row["text"]
        stats["speaker"] = core_speaker
        all_stats.append(stats)

    # Build processed manifest
    processed_df = pd.DataFrame(all_stats)
    df = df.reset_index(drop=True)
    for col in processed_df.columns:
        if col not in df.columns:
            df[col] = processed_df[col].values

    # Step 4: Create splits
    logger.info("Step 4: Creating splits...")
    train_df, val_df, test_df, eval_holdout_df = create_splits(
        df,
        train_ratio=splits_cfg.get("train_ratio", 0.8),
        val_ratio=splits_cfg.get("val_ratio", 0.1),
        test_ratio=splits_cfg.get("test_ratio", 0.1),
        seed=seed,
    )

    # Save manifests
    train_df.to_csv(Path(manifests_dir) / "train.csv", index=False)
    val_df.to_csv(Path(manifests_dir) / "val.csv", index=False)
    test_df.to_csv(Path(manifests_dir) / "test.csv", index=False)
    eval_holdout_df.to_csv(Path(manifests_dir) / "eval_holdout.csv", index=False)

    # Save full manifest with prosody stats
    df.to_csv(Path(manifests_dir) / "full_manifest.csv", index=False)

    # Step 5: Summary statistics
    summary = {
        "core_speaker": core_speaker,
        "total_files": len(df),
        "train_files": len(train_df),
        "val_files": len(val_df),
        "test_files": len(test_df),
        "eval_holdout_files": len(eval_holdout_df),
    }

    # Per-emotion counts
    for emotion in EMOTION_LABELS:
        summary[f"count_{emotion}"] = int((df["emotion"] == emotion).sum())

    # Duration stats
    if "duration" in df.columns:
        summary["total_duration_min"] = float(df["duration"].sum() / 60)
        summary["mean_duration_sec"] = float(df["duration"].mean())

    # Save summary
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv("tables/dataset_stats.csv", index=False)

    logger.info(f"Dataset preparation complete. Summary:\n{summary}")
    return summary


def main():
    """CLI entry point."""
    import yaml
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Prepare EmoV-DB dataset")
    parser.add_argument(
        "--config", type=str, default="configs/data.yaml",
        help="Path to data config YAML",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    prepare_dataset(cfg)


if __name__ == "__main__":
    main()
