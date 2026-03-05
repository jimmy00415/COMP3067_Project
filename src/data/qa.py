"""Data Quality Assurance module (S1.5).

Performs Coqui-recommended checks on the processed dataset:
- Clip length histogram
- Transcript length histogram
- Bad/corrupted file detection
- Annotation-audio consistency
- Per-emotion class counts
- Noise/spectrogram inspection (random subset)

Outputs: docs/data_qa_report.md + figures/data_qa/*.png

Usage:
    python -m src.data.qa --config configs/data.yaml
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.utils import load_audio, EMOTION_LABELS

logger = logging.getLogger(__name__)


def check_clip_lengths(
    df: pd.DataFrame,
    output_dir: str | Path = "figures/data_qa",
) -> dict:
    """Generate clip length histogram and flag outliers."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    durations = df["duration"].values if "duration" in df.columns else []
    if len(durations) == 0:
        logger.warning("No duration column found — computing from audio files...")
        durations = []
        for _, row in df.iterrows():
            try:
                audio, sr = load_audio(row.get("processed_path", row.get("file_path")))
                durations.append(len(audio) / sr)
            except Exception as e:
                logger.error(f"Cannot load {row.get('file_path')}: {e}")
                durations.append(0)
        df = df.copy()
        df["duration"] = durations
        durations = np.array(durations)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(durations, bins=50, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Duration (seconds)")
    ax.set_ylabel("Count")
    ax.set_title("Clip Length Distribution")
    ax.axvline(np.mean(durations), color="red", linestyle="--", label=f"Mean: {np.mean(durations):.2f}s")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "clip_lengths.png", dpi=150)
    plt.close(fig)

    stats = {
        "min_duration": float(np.min(durations)),
        "max_duration": float(np.max(durations)),
        "mean_duration": float(np.mean(durations)),
        "std_duration": float(np.std(durations)),
        "outliers_short": int(np.sum(durations < 0.5)),
        "outliers_long": int(np.sum(durations > 15.0)),
    }
    return stats


def check_transcript_lengths(
    df: pd.DataFrame,
    output_dir: str | Path = "figures/data_qa",
) -> dict:
    """Generate transcript length histogram."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if "text" not in df.columns:
        logger.warning("No text column found in manifest.")
        return {"has_transcripts": False}

    char_lengths = df["text"].str.len().values
    word_counts = df["text"].str.split().str.len().values

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(char_lengths, bins=50, edgecolor="black", alpha=0.7)
    axes[0].set_xlabel("Character count")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Transcript Length (chars)")

    axes[1].hist(word_counts, bins=30, edgecolor="black", alpha=0.7, color="green")
    axes[1].set_xlabel("Word count")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Transcript Length (words)")

    fig.tight_layout()
    fig.savefig(output_dir / "transcript_lengths.png", dpi=150)
    plt.close(fig)

    return {
        "has_transcripts": True,
        "mean_chars": float(np.mean(char_lengths)),
        "mean_words": float(np.mean(word_counts)),
        "empty_transcripts": int(np.sum(char_lengths == 0)),
    }


def check_corrupted_files(df: pd.DataFrame) -> list[str]:
    """Detect bad/corrupted audio files."""
    bad_files = []
    file_col = "processed_path" if "processed_path" in df.columns else "file_path"

    for _, row in df.iterrows():
        fpath = row[file_col]
        try:
            audio, sr = load_audio(fpath)
            if len(audio) == 0:
                bad_files.append({"file": fpath, "issue": "empty audio"})
            elif np.any(np.isnan(audio)):
                bad_files.append({"file": fpath, "issue": "contains NaN"})
            elif np.max(np.abs(audio)) < 1e-6:
                bad_files.append({"file": fpath, "issue": "silent/near-zero"})
        except Exception as e:
            bad_files.append({"file": fpath, "issue": str(e)})

    if bad_files:
        logger.warning(f"Found {len(bad_files)} corrupted/problematic files")
    else:
        logger.info("No corrupted files detected")
    return bad_files


def check_class_balance(
    df: pd.DataFrame,
    output_dir: str | Path = "figures/data_qa",
) -> dict:
    """Generate per-emotion class counts and balance chart."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    counts = df["emotion"].value_counts()

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = sns.color_palette("husl", len(counts))
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="black")
    ax.set_xlabel("Emotion")
    ax.set_ylabel("Sample Count")
    ax.set_title("Class Balance")
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                str(val), ha="center", va="bottom", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "class_balance.png", dpi=150)
    plt.close(fig)

    return {f"count_{e}": int(counts.get(e, 0)) for e in EMOTION_LABELS}


def check_prosody_distributions(
    df: pd.DataFrame,
    output_dir: str | Path = "figures/data_qa",
) -> None:
    """Plot F0 and energy distributions per emotion (pre-training diagnostic)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if "f0_mean" not in df.columns:
        logger.info("No prosody stats in manifest — skipping distribution plots")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.boxplot(data=df, x="emotion", y="f0_mean", ax=axes[0], palette="husl")
    axes[0].set_title("F0 Mean by Emotion (raw data)")
    axes[0].set_ylabel("F0 Mean (Hz)")

    sns.boxplot(data=df, x="emotion", y="energy_mean", ax=axes[1], palette="husl")
    axes[1].set_title("Energy Mean by Emotion (raw data)")
    axes[1].set_ylabel("Energy Mean")

    fig.tight_layout()
    fig.savefig(output_dir / "prosody_distributions.png", dpi=150)
    plt.close(fig)


def generate_qa_report(
    manifest_path: str | Path,
    output_dir: str | Path = "figures/data_qa",
    report_path: str | Path = "docs/data_qa_report.md",
) -> None:
    """Generate full Data QA report (S1.5).

    Args:
        manifest_path: Path to full manifest CSV.
        output_dir: Directory for QA figures.
        report_path: Path to output Markdown report.
    """
    logger.info("Starting Data QA...")
    df = pd.read_csv(manifest_path)

    # Run all checks
    clip_stats = check_clip_lengths(df, output_dir)
    transcript_stats = check_transcript_lengths(df, output_dir)
    bad_files = check_corrupted_files(df)
    class_stats = check_class_balance(df, output_dir)
    check_prosody_distributions(df, output_dir)

    # Write report
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)

    report_lines = [
        "# Data QA Report",
        "",
        f"**Generated from:** `{manifest_path}`",
        f"**Total samples:** {len(df)}",
        "",
        "## 1. Clip Length Statistics",
        "",
        f"- Min duration: {clip_stats['min_duration']:.2f}s",
        f"- Max duration: {clip_stats['max_duration']:.2f}s",
        f"- Mean duration: {clip_stats['mean_duration']:.2f}s ± {clip_stats['std_duration']:.2f}s",
        f"- Outliers <0.5s: {clip_stats['outliers_short']}",
        f"- Outliers >15s: {clip_stats['outliers_long']}",
        "",
        "![Clip lengths](../figures/data_qa/clip_lengths.png)",
        "",
        "## 2. Transcript Statistics",
        "",
    ]

    if transcript_stats.get("has_transcripts"):
        report_lines.extend([
            f"- Mean character length: {transcript_stats['mean_chars']:.1f}",
            f"- Mean word count: {transcript_stats['mean_words']:.1f}",
            f"- Empty transcripts: {transcript_stats['empty_transcripts']}",
            "",
            "![Transcript lengths](../figures/data_qa/transcript_lengths.png)",
        ])
    else:
        report_lines.append("No transcripts found in manifest.")

    report_lines.extend([
        "",
        "## 3. Class Balance",
        "",
        "| Emotion | Count |",
        "|---|---|",
    ])
    for emotion in EMOTION_LABELS:
        count = class_stats.get(f"count_{emotion}", 0)
        report_lines.append(f"| {emotion} | {count} |")

    report_lines.extend([
        "",
        "![Class balance](../figures/data_qa/class_balance.png)",
        "",
        "## 4. Corrupted Files",
        "",
        f"Found **{len(bad_files)}** problematic files.",
    ])

    if bad_files:
        report_lines.append("")
        report_lines.append("| File | Issue |")
        report_lines.append("|---|---|")
        for bf in bad_files[:20]:  # Show max 20
            report_lines.append(f"| `{Path(bf['file']).name}` | {bf['issue']} |")

    report_lines.extend([
        "",
        "## 5. Prosody Distributions (Pre-Training)",
        "",
        "![Prosody distributions](../figures/data_qa/prosody_distributions.png)",
        "",
        "## Quality Gate G1 Assessment",
        "",
        "- [ ] Core speaker has all 4 emotions with sufficient counts",
        "- [ ] No corrupted files (or issues resolved)",
        "- [ ] Class balance acceptable (no extreme imbalance)",
        "- [ ] Clip lengths reasonable (0.5s–15s)",
        "- [ ] Prosody distributions show expected emotion differentiation",
    ])

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    logger.info(f"QA report written to {report_path}")


def main():
    """CLI entry point."""
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Data Quality Assurance")
    parser.add_argument("--manifest", type=str, default="data/manifests/full_manifest.csv")
    parser.add_argument("--output-dir", type=str, default="figures/data_qa")
    parser.add_argument("--report", type=str, default="docs/data_qa_report.md")
    args = parser.parse_args()

    generate_qa_report(args.manifest, args.output_dir, args.report)


if __name__ == "__main__":
    main()
