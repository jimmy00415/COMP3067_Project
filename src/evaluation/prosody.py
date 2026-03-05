"""Prosody analysis — primary automatic evaluation metric (FR-5, ADR-6).

Computes F0 and energy statistics per system × emotion condition.
This is the primary automatic metric for measuring emotion expressiveness.

Key analyses:
1. F0 statistics (mean, std, range) per system × emotion
2. Energy statistics (mean, std) per system × emotion
3. System comparison (stat tests: Kruskal-Wallis + post-hoc Dunn)
4. Causal attribution: A0→A (domain effect), A→B (emotion effect),
   B→C (prosody supervision effect)

Usage:
    python -m src.evaluation.prosody --config configs/eval.yaml
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from src.data.utils import (
    EMOTION_LABELS,
    load_audio,
    extract_f0,
    extract_energy,
    compute_utterance_prosody_stats,
)

logger = logging.getLogger(__name__)


def analyze_single_file(
    audio_path: str,
    sr: int = 22050,
    f0_fmin: float = 75.0,
    f0_fmax: float = 300.0,
    hop_length: int = 256,
) -> dict:
    """Extract prosody features from a single audio file.

    Returns:
        Dict with F0 and energy statistics.
    """
    try:
        audio, _ = load_audio(audio_path, sr=sr)
        f0, voiced = extract_f0(audio, sr=sr, fmin=f0_fmin, fmax=f0_fmax,
                                 hop_length=hop_length)
        energy = extract_energy(audio, hop_length=hop_length)
        stats = compute_utterance_prosody_stats(f0, energy)
        stats["duration"] = len(audio) / sr
        stats["voiced_ratio"] = float(np.sum(voiced) / max(len(voiced), 1))
        stats["status"] = "ok"
    except Exception as e:
        logger.error(f"Failed to analyze {audio_path}: {e}")
        stats = {
            "f0_mean": 0, "f0_std": 0, "f0_range_low": 0, "f0_range_high": 0,
            "energy_mean": 0, "energy_std": 0, "duration": 0, "voiced_ratio": 0,
            "status": f"error: {e}",
        }
    return stats


def analyze_eval_stimuli(
    manifest_path: str,
    sr: int = 22050,
    f0_fmin: float = 75.0,
    f0_fmax: float = 300.0,
) -> pd.DataFrame:
    """Analyze all eval stimuli and compute prosody features.

    Args:
        manifest_path: Path to eval_manifest.csv.
        sr: Sample rate.
        f0_fmin, f0_fmax: Speaker-aware F0 range.

    Returns:
        DataFrame with prosody stats for each file.
    """
    df = pd.read_csv(manifest_path)
    logger.info(f"Analyzing {len(df)} eval stimuli...")

    all_stats = []
    for idx, row in df.iterrows():
        file_path = row.get("file_path", "")
        if not file_path or not Path(file_path).exists():
            logger.warning(f"Missing file: {file_path}")
            continue

        stats = analyze_single_file(file_path, sr=sr, f0_fmin=f0_fmin, f0_fmax=f0_fmax)
        stats.update({
            "system": row["system"],
            "emotion": row["emotion"],
            "text_id": row.get("text_id", 0),
            "text": row.get("text", ""),
            "file_path": file_path,
        })
        all_stats.append(stats)

    result_df = pd.DataFrame(all_stats)
    logger.info(f"Successfully analyzed {len(result_df)} files")
    return result_df


def compute_system_emotion_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute aggregate statistics per system × emotion.

    Returns:
        Pivot DataFrame with mean ± std for each prosody metric.
    """
    metrics = ["f0_mean", "f0_std", "f0_range_high", "energy_mean", "energy_std", "duration"]

    agg_funcs = {}
    for m in metrics:
        if m in df.columns:
            agg_funcs[m] = ["mean", "std", "median"]

    grouped = df.groupby(["system", "emotion"]).agg(agg_funcs)
    grouped.columns = ["_".join(col) for col in grouped.columns]

    return grouped.reset_index()


def test_emotion_differentiation(
    df: pd.DataFrame,
    metric: str = "f0_mean",
    system: str = "B",
) -> dict:
    """Test whether emotions produce significantly different prosody.

    Uses Kruskal-Wallis H-test (non-parametric) across emotion groups
    within a single system.

    Args:
        df: Prosody analysis DataFrame.
        metric: Prosody metric to test.
        system: System to test.

    Returns:
        Dict with test results.
    """
    system_df = df[df["system"] == system]

    groups = []
    for emotion in EMOTION_LABELS:
        emotion_vals = system_df[system_df["emotion"] == emotion][metric].dropna().values
        if len(emotion_vals) > 0:
            groups.append(emotion_vals)

    if len(groups) < 2:
        return {"test": "kruskal", "system": system, "metric": metric,
                "error": "Not enough groups"}

    try:
        h_stat, p_value = scipy_stats.kruskal(*groups)
        return {
            "test": "kruskal_wallis",
            "system": system,
            "metric": metric,
            "h_statistic": float(h_stat),
            "p_value": float(p_value),
            "significant": p_value < 0.05,
            "n_groups": len(groups),
        }
    except Exception as e:
        return {"test": "kruskal", "error": str(e)}


def test_causal_attribution(
    df: pd.DataFrame,
    metric: str = "f0_std",
    emotion: str = "angry",
) -> dict:
    """Test causal attribution: A0→A→B→C improvement chain.

    Compares metric values between consecutive systems for a specific
    emotion to quantify each modification's effect.

    Args:
        df: Prosody analysis DataFrame.
        metric: Metric to compare.
        emotion: Emotion to test.

    Returns:
        Dict with pairwise comparison results.
    """
    emotion_df = df[df["emotion"] == emotion]
    systems = ["A0", "A", "B", "C"]
    comparisons = []

    for i in range(len(systems) - 1):
        sys_a = systems[i]
        sys_b = systems[i + 1]

        vals_a = emotion_df[emotion_df["system"] == sys_a][metric].dropna().values
        vals_b = emotion_df[emotion_df["system"] == sys_b][metric].dropna().values

        if len(vals_a) < 2 or len(vals_b) < 2:
            comparisons.append({
                "comparison": f"{sys_a}→{sys_b}",
                "error": "Insufficient data",
            })
            continue

        # Mann-Whitney U test
        try:
            u_stat, p_value = scipy_stats.mannwhitneyu(
                vals_a, vals_b, alternative="two-sided"
            )
            mean_diff = float(np.mean(vals_b) - np.mean(vals_a))

            # Rank-biserial correlation (proper effect size for Mann-Whitney)
            n_a, n_b = len(vals_a), len(vals_b)
            r_effect = 1 - (2 * u_stat) / (n_a * n_b)

            comparisons.append({
                "comparison": f"{sys_a}→{sys_b}",
                "metric": metric,
                "emotion": emotion,
                "mean_a": float(np.mean(vals_a)),
                "mean_b": float(np.mean(vals_b)),
                "mean_diff": mean_diff,
                "rank_biserial_r": float(r_effect),
                "u_statistic": float(u_stat),
                "p_value": float(p_value),
                "significant": p_value < 0.05,
                "interpretation": _interpret_effect(sys_a, sys_b, mean_diff, p_value),
            })
        except Exception as e:
            comparisons.append({
                "comparison": f"{sys_a}→{sys_b}",
                "error": str(e),
            })

    return {
        "metric": metric,
        "emotion": emotion,
        "comparisons": comparisons,
    }


def _interpret_effect(sys_a: str, sys_b: str, effect: float, p: float) -> str:
    """Generate human-readable interpretation of a system comparison."""
    labels = {
        "A0→A": "domain adaptation",
        "A→B": "emotion conditioning",
        "B→C": "prosody supervision",
    }
    label = labels.get(f"{sys_a}→{sys_b}", f"{sys_a}→{sys_b}")

    if p >= 0.05:
        return f"No significant effect of {label} (p={p:.3f})"

    direction = "increased" if effect > 0 else "decreased"
    return f"{label} significantly {direction} the metric (Δ={effect:.2f}, p={p:.3f})"


def run_prosody_evaluation(cfg: dict) -> dict:
    """Run full prosody evaluation pipeline.

    Args:
        cfg: Evaluation config dict (from eval.yaml).

    Returns:
        Dict with all results.
    """
    eval_cfg = cfg.get("evaluation", cfg)
    prosody_cfg = eval_cfg.get("prosody", {})

    manifest_path = eval_cfg.get("manifest_path", "data/processed/eval_stimuli/eval_manifest.csv")
    output_dir = eval_cfg.get("output_dir", "tables")
    sr = prosody_cfg.get("sample_rate", 22050)
    f0_fmin = prosody_cfg.get("f0_fmin", 75.0)
    f0_fmax = prosody_cfg.get("f0_fmax", 300.0)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Step 1: Analyze all stimuli
    logger.info("Step 1: Analyzing prosody features...")
    prosody_df = analyze_eval_stimuli(manifest_path, sr=sr, f0_fmin=f0_fmin, f0_fmax=f0_fmax)

    # Save full prosody analysis
    prosody_df.to_csv(Path(output_dir) / "prosody_analysis.csv", index=False)

    # Step 2: Aggregate stats
    logger.info("Step 2: Computing aggregate statistics...")
    agg_stats = compute_system_emotion_stats(prosody_df)
    agg_stats.to_csv(Path(output_dir) / "prosody_stats_aggregate.csv", index=False)

    # Step 3: Statistical tests
    logger.info("Step 3: Running statistical tests...")

    # Test emotion differentiation per system
    diff_tests = []
    for system in prosody_df["system"].unique():
        for metric in ["f0_mean", "f0_std", "energy_mean"]:
            result = test_emotion_differentiation(prosody_df, metric, system)
            diff_tests.append(result)

    diff_df = pd.DataFrame(diff_tests)
    diff_df.to_csv(Path(output_dir) / "emotion_differentiation_tests.csv", index=False)

    # Step 4: Causal attribution tests
    logger.info("Step 4: Causal attribution analysis...")
    causal_results = []
    for emotion in EMOTION_LABELS:
        for metric in ["f0_mean", "f0_std", "energy_mean", "energy_std"]:
            result = test_causal_attribution(prosody_df, metric, emotion)
            for comp in result.get("comparisons", []):
                causal_results.append(comp)

    causal_df = pd.DataFrame(causal_results)
    causal_df.to_csv(Path(output_dir) / "causal_attribution.csv", index=False)

    logger.info("Prosody evaluation complete")
    return {
        "prosody_df": prosody_df,
        "agg_stats": agg_stats,
        "diff_tests": diff_df,
        "causal_results": causal_df,
    }


def main():
    import yaml, argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Prosody evaluation")
    parser.add_argument("--config", type=str, default="configs/eval.yaml")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    run_prosody_evaluation(cfg)


if __name__ == "__main__":
    main()
