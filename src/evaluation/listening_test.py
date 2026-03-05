"""Listening test stimulus pack generator (FR-5).

Creates a structured stimulus pack for P.808-inspired local listening tests.
Handles LUFS normalization, randomization, and response form generation.

Design (ADR-7):
- Training audio: peak normalized (preserve energy as signal)
- Eval stimuli: LUFS normalized to -23 LUFS (ensure listener fairness)

Test structure:
- 16 canary texts × 4 emotions × 4 systems (A0/A/B/C) = 256 stimuli max
- Practical subset: 4 texts × 4 emotions × 4 systems = 64 stimuli
- Estimated listener time: 12-18 minutes at 10-15s per stimulus

Usage:
    python -m src.evaluation.listening_test --config configs/eval.yaml
"""

import logging
import json
import random
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def create_stimulus_pack(
    manifest_path: str,
    output_dir: str = "outputs/listening_test",
    n_texts: int = 4,
    seed: int = 42,
) -> dict:
    """Create a listening test stimulus pack.

    Args:
        manifest_path: Path to eval_manifest.csv with LUFS file paths.
        output_dir: Output directory for stimulus pack.
        n_texts: Number of texts to include (subset for practical time).
        seed: Random seed for reproducibility.

    Returns:
        Dict with stimulus pack metadata.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_path)

    # Select subset of texts
    all_text_ids = sorted(df["text_id"].unique())
    rng = random.Random(seed)

    if len(all_text_ids) > n_texts:
        selected_texts = sorted(rng.sample(list(all_text_ids), n_texts))
    else:
        selected_texts = all_text_ids

    subset_df = df[df["text_id"].isin(selected_texts)].copy()

    # Randomize presentation order (blind the system identity)
    stimuli = []
    for idx, row in subset_df.iterrows():
        stimulus_id = f"stim_{len(stimuli):03d}"
        lufs_path = row.get("lufs_path", row.get("file_path", ""))

        stimuli.append({
            "stimulus_id": stimulus_id,
            "audio_file": lufs_path,
            "system": row["system"],        # Hidden from listener
            "emotion": row["emotion"],       # Shown to listener as "intended emotion"
            "text_id": row["text_id"],
            "text": row.get("text", ""),
        })

    # Shuffle for randomized presentation
    rng.shuffle(stimuli)

    # Assign presentation order
    for i, stim in enumerate(stimuli):
        stim["presentation_order"] = i + 1

    # Save stimulus manifest (researcher version — includes system labels)
    stim_df = pd.DataFrame(stimuli)
    stim_df.to_csv(output_dir / "stimulus_manifest_full.csv", index=False)

    # Save listener version (no system labels)
    listener_df = stim_df[["stimulus_id", "presentation_order", "audio_file",
                            "emotion", "text"]].copy()
    listener_df.to_csv(output_dir / "stimulus_manifest_listener.csv", index=False)

    # Generate response form template
    _generate_response_form(stimuli, output_dir)

    # Summary
    summary = {
        "total_stimuli": len(stimuli),
        "n_texts": len(selected_texts),
        "n_systems": len(subset_df["system"].unique()),
        "n_emotions": len(subset_df["emotion"].unique()),
        "estimated_time_min": len(stimuli) * 12 / 60,  # ~12s per stimulus
        "seed": seed,
    }

    with open(output_dir / "stimulus_pack_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Stimulus pack created: {len(stimuli)} stimuli, "
                f"~{summary['estimated_time_min']:.0f} min")
    return summary


def _generate_response_form(stimuli: list[dict], output_dir: Path) -> None:
    """Generate a Markdown response form for the listening test."""

    lines = [
        "# Emotive TTS Listening Test Response Form",
        "",
        "**Instructions:**",
        "- Listen to each audio sample once through headphones in a quiet environment.",
        "- For each sample, you are told the INTENDED emotion.",
        "- Rate the following on a 1-5 scale:",
        "  1. **Naturalness**: How natural does the speech sound? (1=very unnatural, 5=very natural)",
        "  2. **Emotion Match**: How well does the emotion in the audio match the intended emotion? "
        "(1=not at all, 5=perfectly)",
        "  3. **Overall Quality**: Overall impression (1=poor, 5=excellent)",
        "",
        "---",
        "",
    ]

    for stim in stimuli:
        lines.extend([
            f"## Stimulus {stim['presentation_order']}: `{stim['stimulus_id']}`",
            "",
            f"- **Intended emotion:** {stim['emotion']}",
            f"- **Text:** \"{stim['text']}\"",
            f"- **Audio:** `{Path(stim['audio_file']).name}`",
            "",
            "| Criterion | Rating (1-5) |",
            "|---|---|",
            "| Naturalness | ___ |",
            "| Emotion Match | ___ |",
            "| Overall Quality | ___ |",
            "",
            "Comments: ________________________________",
            "",
            "---",
            "",
        ])

    with open(output_dir / "response_form.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info(f"Response form generated: {output_dir / 'response_form.md'}")


def analyze_responses(
    responses_path: str,
    stimulus_manifest_path: str,
    output_dir: str = "tables",
) -> pd.DataFrame:
    """Analyze listening test responses (when available).

    Args:
        responses_path: Path to collected responses CSV.
        stimulus_manifest_path: Path to full stimulus manifest (with system labels).
        output_dir: Output directory for analysis tables.

    Returns:
        Analysis DataFrame.
    """
    # This will be filled in when responses are collected
    logger.info("Listening test analysis — waiting for response data")
    return pd.DataFrame()


def main():
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Generate listening test stimulus pack")
    parser.add_argument("--manifest", type=str,
                        default="data/processed/eval_stimuli/eval_manifest.csv")
    parser.add_argument("--output-dir", type=str, default="outputs/listening_test")
    parser.add_argument("--n-texts", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    create_stimulus_pack(args.manifest, args.output_dir, args.n_texts, args.seed)


if __name__ == "__main__":
    main()
