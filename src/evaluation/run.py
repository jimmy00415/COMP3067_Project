"""One-command evaluation orchestrator (S6.9).

Runs all evaluation stages in sequence:
1. Prosody analysis (PRIMARY metric)
2. SER proxy (AUXILIARY — label-mismatch caveat)
3. Plot generation

Usage:
    python -m src.evaluation.run --config configs/eval.yaml
"""

import logging
import yaml
import argparse
from pathlib import Path

logger = logging.getLogger(__name__)


def run_full_evaluation(cfg: dict) -> dict:
    """Run the complete evaluation pipeline.

    Args:
        cfg: Evaluation config dict (from eval.yaml).

    Returns:
        Dict containing results from each stage.
    """
    results = {}

    # ---- Stage 1: Prosody analysis (PRIMARY) ----
    logger.info("=" * 60)
    logger.info("Stage 1: Prosody analysis (PRIMARY automatic metric)")
    logger.info("=" * 60)
    try:
        from src.evaluation.prosody import run_prosody_evaluation
        results["prosody"] = run_prosody_evaluation(cfg)
        logger.info("Prosody analysis complete")
    except Exception as e:
        logger.error(f"Prosody analysis failed: {e}")
        results["prosody"] = {"error": str(e)}

    # ---- Stage 2: SER proxy (AUXILIARY ONLY) ----
    logger.info("=" * 60)
    logger.info("Stage 2: SER proxy (AUXILIARY — label mismatch caveat)")
    logger.info("=" * 60)
    try:
        from src.evaluation.ser_probe import run_ser_evaluation
        results["ser"] = run_ser_evaluation(cfg)
        logger.info("SER proxy evaluation complete")
    except Exception as e:
        logger.error(f"SER evaluation failed: {e}")
        results["ser"] = {"error": str(e)}

    # ---- Stage 3: Plot generation ----
    logger.info("=" * 60)
    logger.info("Stage 3: Generating evaluation plots")
    logger.info("=" * 60)
    try:
        from src.evaluation.plots import generate_all_plots
        generate_all_plots(cfg)
        logger.info("Plot generation complete")
    except Exception as e:
        logger.error(f"Plot generation failed: {e}")
        results["plots"] = {"error": str(e)}

    # ---- Summary ----
    logger.info("=" * 60)
    logger.info("Evaluation pipeline complete")
    logger.info("=" * 60)
    eval_cfg = cfg.get("evaluation", cfg)
    output_dir = eval_cfg.get("output_dir", "tables")
    figures_dir = eval_cfg.get("figures_dir", "figures")
    logger.info(f"Tables saved to:  {output_dir}/")
    logger.info(f"Figures saved to: {figures_dir}/")

    return results


def main():
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Run full evaluation pipeline")
    parser.add_argument("--config", type=str, default="configs/eval.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    run_full_evaluation(cfg)


if __name__ == "__main__":
    main()
