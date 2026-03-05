"""Visualization module for prosody analysis and evaluation results.

Generates publication-quality plots for the dissertation.
All plots saved at 300 DPI to figures/ directory.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

# Consistent style
SYSTEM_COLORS = {"A0": "#95a5a6", "A": "#3498db", "B": "#e74c3c", "C": "#2ecc71"}
EMOTION_COLORS = {"neutral": "#95a5a6", "angry": "#e74c3c", "amused": "#f39c12", "disgust": "#9b59b6"}
SYSTEM_ORDER = ["A0", "A", "B", "C"]
EMOTION_ORDER = ["neutral", "angry", "amused", "disgust"]


def set_plot_style():
    """Set consistent plot style for all figures."""
    sns.set_theme(style="whitegrid", font_scale=1.2)
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
    })


def plot_f0_by_system_emotion(
    df: pd.DataFrame,
    output_path: str = "figures/f0_by_system_emotion.png",
    metric: str = "f0_mean",
    title: str = "F0 Mean by System × Emotion",
) -> None:
    """Box plot of F0 metric grouped by system and emotion.

    This is the primary result figure showing emotion differentiation
    across the A0→A→B→C progression.
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=(12, 6))

    # Filter and order
    plot_df = df[df["system"].isin(SYSTEM_ORDER)].copy()
    plot_df["system"] = pd.Categorical(plot_df["system"], categories=SYSTEM_ORDER, ordered=True)
    plot_df["emotion"] = pd.Categorical(plot_df["emotion"], categories=EMOTION_ORDER, ordered=True)

    sns.boxplot(
        data=plot_df, x="system", y=metric, hue="emotion",
        palette=EMOTION_COLORS, ax=ax,
    )

    ax.set_title(title)
    ax.set_xlabel("System")
    ax.set_ylabel(f"{metric.replace('_', ' ').title()} (Hz)")
    ax.legend(title="Emotion", loc="upper right")

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


def plot_prosody_comparison_grid(
    df: pd.DataFrame,
    output_path: str = "figures/prosody_comparison_grid.png",
) -> None:
    """Grid of prosody metrics: F0 mean, F0 std, energy mean, energy std."""
    set_plot_style()
    metrics = ["f0_mean", "f0_std", "energy_mean", "energy_std"]
    titles = ["F0 Mean (Hz)", "F0 Std (Hz)", "Energy Mean", "Energy Std"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    plot_df = df[df["system"].isin(SYSTEM_ORDER)].copy()
    plot_df["system"] = pd.Categorical(plot_df["system"], categories=SYSTEM_ORDER, ordered=True)

    for ax, metric, title in zip(axes, metrics, titles):
        if metric not in plot_df.columns:
            ax.set_visible(False)
            continue

        sns.boxplot(
            data=plot_df, x="system", y=metric, hue="emotion",
            palette=EMOTION_COLORS, ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("System")
        ax.legend(title="Emotion", fontsize=8)

    fig.suptitle("Prosody Metrics: System × Emotion Comparison", fontsize=16, y=1.02)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


def plot_causal_chain(
    causal_df: pd.DataFrame,
    emotion: str = "angry",
    metric: str = "f0_std",
    output_path: str = "figures/causal_chain.png",
) -> None:
    """Visualize the A0→A→B→C causal improvement chain.

    Shows mean metric value and significance markers for each step.
    """
    set_plot_style()

    chain_df = causal_df[
        (causal_df["emotion"] == emotion) &
        (causal_df["metric"] == metric)
    ].copy()

    if chain_df.empty:
        logger.warning(f"No causal data for {emotion}/{metric}")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Extract values for each system
    systems = SYSTEM_ORDER
    means = []
    for sys in systems:
        row = chain_df[chain_df["comparison"].str.startswith(sys)]
        if not row.empty:
            means.append(row.iloc[0].get("mean_a", 0))
        else:
            # Last system — get from mean_b of previous comparison
            row = chain_df[chain_df["comparison"].str.endswith(sys)]
            if not row.empty:
                means.append(row.iloc[0].get("mean_b", 0))
            else:
                means.append(0)

    bars = ax.bar(systems, means, color=[SYSTEM_COLORS[s] for s in systems],
                  edgecolor="black", linewidth=1.2)

    # Add significance arrows between consecutive systems
    for i, (_, row) in enumerate(chain_df.iterrows()):
        if "significant" in row and row.get("significant"):
            # Draw significance marker
            x_pos = i + 0.5
            y_pos = max(means) * 1.05
            ax.annotate("*", xy=(x_pos, y_pos), fontsize=20, ha="center",
                        fontweight="bold", color="red")

    ax.set_title(f"Causal Chain: {metric.replace('_', ' ').title()} for {emotion.title()}")
    ax.set_xlabel("System")
    ax.set_ylabel(metric.replace("_", " ").title())

    # Add value labels on bars
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(means) * 0.01,
                f"{val:.1f}", ha="center", va="bottom", fontweight="bold")

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


def plot_ser_confusion(
    ser_df: pd.DataFrame,
    output_path: str = "figures/ser_confusion.png",
) -> None:
    """Plot SER probe confusion matrix per system."""
    set_plot_style()

    systems = [s for s in SYSTEM_ORDER if s in ser_df["system"].unique()]

    fig, axes = plt.subplots(1, len(systems), figsize=(5 * len(systems), 5))
    if len(systems) == 1:
        axes = [axes]

    for ax, system in zip(axes, systems):
        sys_df = ser_df[ser_df["system"] == system]
        if "predicted_mapped" not in sys_df.columns:
            continue

        # Build confusion matrix
        emotions = [e for e in EMOTION_ORDER if e in sys_df["emotion"].unique()]
        cm = pd.crosstab(
            sys_df["emotion"],
            sys_df["predicted_mapped"],
            normalize="index",
        )

        sns.heatmap(cm, annot=True, fmt=".2f", cmap="YlOrRd",
                     ax=ax, vmin=0, vmax=1)
        ax.set_title(f"System {system}")
        ax.set_xlabel("SER Predicted")
        ax.set_ylabel("Intended Emotion")

    fig.suptitle("SER Probe Confusion Matrices (auxiliary metric)", fontsize=14)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


def plot_training_curves(
    log_path: str,
    output_path: str = "figures/training_curves.png",
) -> None:
    """Plot training and validation loss curves from MLflow/CSV logs."""
    set_plot_style()

    try:
        df = pd.read_csv(log_path)
    except Exception as e:
        logger.warning(f"Cannot read training log: {e}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if "train_loss" in df.columns:
        axes[0].plot(df["epoch"], df["train_loss"], label="Train", linewidth=2)
    if "val_loss" in df.columns:
        axes[0].plot(df["epoch"], df["val_loss"], label="Val", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Curves")
    axes[0].legend()

    if "lr" in df.columns:
        axes[1].plot(df["epoch"], df["lr"], color="green", linewidth=2)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Learning Rate")
        axes[1].set_title("Learning Rate Schedule")

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


def generate_all_plots(cfg: dict) -> None:
    """Generate all evaluation plots.

    Args:
        cfg: Evaluation config dict.
    """
    eval_cfg = cfg.get("evaluation", cfg)
    tables_dir = eval_cfg.get("output_dir", "tables")
    figures_dir = eval_cfg.get("figures_dir", "figures")

    Path(figures_dir).mkdir(parents=True, exist_ok=True)

    # Prosody plots
    prosody_path = Path(tables_dir) / "prosody_analysis.csv"
    if prosody_path.exists():
        prosody_df = pd.read_csv(prosody_path)

        plot_f0_by_system_emotion(prosody_df, f"{figures_dir}/f0_by_system_emotion.png")
        plot_prosody_comparison_grid(prosody_df, f"{figures_dir}/prosody_comparison_grid.png")

        # Causal chain plots
        causal_path = Path(tables_dir) / "causal_attribution.csv"
        if causal_path.exists():
            causal_df = pd.read_csv(causal_path)
            for emotion in EMOTION_ORDER:
                plot_causal_chain(
                    causal_df, emotion=emotion, metric="f0_std",
                    output_path=f"{figures_dir}/causal_chain_{emotion}.png",
                )

    # SER plots
    ser_path = Path(tables_dir) / "ser_probe_results.csv"
    if ser_path.exists():
        ser_df = pd.read_csv(ser_path)
        plot_ser_confusion(ser_df, f"{figures_dir}/ser_confusion.png")

    logger.info(f"All plots generated in {figures_dir}/")


def main():
    import yaml, argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Generate evaluation plots")
    parser.add_argument("--config", type=str, default="configs/eval.yaml")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    generate_all_plots(cfg)


if __name__ == "__main__":
    main()
