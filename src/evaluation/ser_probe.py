"""SER (Speech Emotion Recognition) probe — auxiliary metric only (ADR-6).

Uses a pre-trained SpeechBrain SER model (trained on IEMOCAP) to classify
the emotion of generated audio. This is an AUXILIARY metric, NOT primary.

Critical limitation: Label mismatch between IEMOCAP (neutral/angry/happy/sad)
and our emotion set (neutral/angry/amused/disgust). Only neutral and angry
map cleanly. "Amused" may map to "happy" loosely; "disgust" has no IEMOCAP
equivalent — it is excluded from accuracy calculations.

The metric reported is `ser_proxy_agreement` — the percentage of correctly
classified samples for mappable emotions only.

Usage:
    python -m src.evaluation.ser_probe --config configs/eval.yaml
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# IEMOCAP ↔ Our emotions label mapping
# Only includes emotions with reasonable mappings
LABEL_MAPPING = {
    "neutral": "neu",    # Clean map
    "angry": "ang",      # Clean map
    "amused": "hap",     # Loose map (amused ≈ happy)
    "disgust": None,     # No IEMOCAP equivalent — EXCLUDED from accuracy
}

REVERSE_MAPPING = {v: k for k, v in LABEL_MAPPING.items() if v is not None}


class SERProbe:
    """SpeechBrain-based Speech Emotion Recognition probe.

    This is a frozen classifier — we do NOT fine-tune it.
    We use it only to probe whether the generated audio contains
    recognizable emotional cues.
    """

    def __init__(
        self,
        model_source: str = "speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
        save_dir: str = "pretrained_models/ser",
        use_cuda: bool = False,
    ):
        """Initialize SER probe.

        Args:
            model_source: SpeechBrain model hub identifier.
            save_dir: Directory to cache the downloaded model.
            use_cuda: Whether to use GPU.
        """
        self.model_source = model_source
        self.save_dir = save_dir
        self.use_cuda = use_cuda
        self._classifier = None

    def load(self):
        """Load the SpeechBrain classifier."""
        if self._classifier is not None:
            return

        try:
            from speechbrain.inference.classifiers import EncoderClassifier

            self._classifier = EncoderClassifier.from_hparams(
                source=self.model_source,
                savedir=self.save_dir,
                run_opts={"device": "cuda"} if self.use_cuda else {},
            )
            logger.info(f"SER probe loaded from {self.model_source}")
        except ImportError:
            logger.warning("SpeechBrain not available — SER probe disabled")
            raise

    def classify(self, audio_path: str) -> dict:
        """Classify emotion of a single audio file.

        Args:
            audio_path: Path to WAV file.

        Returns:
            Dict with predicted emotion and confidence scores.
        """
        self.load()

        try:
            out_prob, score, index, text_lab = self._classifier.classify_file(audio_path)

            # Get all class probabilities
            probs = out_prob.squeeze().cpu().numpy()
            predicted_label = text_lab[0] if text_lab else "unknown"

            return {
                "predicted_iemocap": predicted_label,
                "predicted_mapped": REVERSE_MAPPING.get(predicted_label, "unmapped"),
                "confidence": float(score.item()) if hasattr(score, "item") else float(score),
                "all_probs": {lab: float(p) for lab, p in zip(
                    self._classifier.hparams.label_encoder.decode_ndim(
                        list(range(len(probs)))
                    ) if hasattr(self._classifier.hparams, "label_encoder") else [],
                    probs,
                )},
                "status": "ok",
            }
        except Exception as e:
            logger.error(f"SER classification failed for {audio_path}: {e}")
            return {
                "predicted_iemocap": "error",
                "predicted_mapped": "error",
                "confidence": 0.0,
                "status": f"error: {e}",
            }

    def classify_batch(self, audio_paths: list[str]) -> list[dict]:
        """Classify a batch of audio files.

        Args:
            audio_paths: List of WAV file paths.

        Returns:
            List of classification result dicts.
        """
        self.load()
        results = []
        for path in audio_paths:
            result = self.classify(path)
            result["file_path"] = path
            results.append(result)
        return results


def compute_ser_proxy_agreement(
    df: pd.DataFrame,
    exclude_unmapped: bool = True,
) -> dict:
    """Compute ser_proxy_agreement metric.

    This is the percentage of generated samples where the SER probe's
    predicted emotion matches the intended emotion, considering only
    emotions with valid IEMOCAP mappings.

    Args:
        df: DataFrame with columns: emotion (intended), predicted_mapped (SER output).
        exclude_unmapped: If True, exclude emotions without IEMOCAP mapping (disgust).

    Returns:
        Dict with agreement metrics.
    """
    eval_df = df.copy()

    if exclude_unmapped:
        # Remove emotions with no IEMOCAP mapping
        unmapped_emotions = [e for e, m in LABEL_MAPPING.items() if m is None]
        eval_df = eval_df[~eval_df["emotion"].isin(unmapped_emotions)]
        logger.info(f"Excluded {len(df) - len(eval_df)} samples with unmapped emotions "
                     f"({unmapped_emotions})")

    if len(eval_df) == 0:
        return {"ser_proxy_agreement": 0.0, "n_samples": 0}

    # Compute agreement
    eval_df["correct"] = eval_df["emotion"] == eval_df["predicted_mapped"]
    overall_agreement = eval_df["correct"].mean()

    # Per-emotion agreement
    per_emotion = {}
    for emotion in eval_df["emotion"].unique():
        mask = eval_df["emotion"] == emotion
        if mask.sum() > 0:
            per_emotion[emotion] = float(eval_df.loc[mask, "correct"].mean())

    # Per-system agreement
    per_system = {}
    if "system" in eval_df.columns:
        for system in eval_df["system"].unique():
            mask = eval_df["system"] == system
            if mask.sum() > 0:
                per_system[system] = float(eval_df.loc[mask, "correct"].mean())

    return {
        "ser_proxy_agreement": float(overall_agreement),
        "n_samples": len(eval_df),
        "n_excluded": len(df) - len(eval_df),
        "per_emotion": per_emotion,
        "per_system": per_system,
    }


def run_ser_evaluation(cfg: dict) -> dict:
    """Run full SER probe evaluation.

    Args:
        cfg: Evaluation config dict.

    Returns:
        Dict with SER results.
    """
    eval_cfg = cfg.get("evaluation", cfg)
    ser_cfg = eval_cfg.get("ser_probe", {})

    manifest_path = eval_cfg.get("manifest_path", "data/processed/eval_stimuli/eval_manifest.csv")
    output_dir = eval_cfg.get("output_dir", "tables")
    use_cuda = ser_cfg.get("use_cuda", False)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load manifest
    df = pd.read_csv(manifest_path)

    # Initialize probe
    probe = SERProbe(
        model_source=ser_cfg.get("model", "speechbrain/emotion-recognition-wav2vec2-IEMOCAP"),
        use_cuda=use_cuda,
    )

    # Classify all files
    logger.info(f"Running SER probe on {len(df)} files...")
    all_results = []
    for _, row in df.iterrows():
        file_path = row.get("file_path", "")
        if not file_path or not Path(file_path).exists():
            continue

        result = probe.classify(file_path)
        result.update({
            "system": row["system"],
            "emotion": row["emotion"],
            "text_id": row.get("text_id", 0),
            "file_path": file_path,
        })
        all_results.append(result)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(Path(output_dir) / "ser_probe_results.csv", index=False)

    # Compute agreement
    agreement = compute_ser_proxy_agreement(results_df)

    # Save summary
    import json
    with open(Path(output_dir) / "ser_proxy_agreement.json", "w") as f:
        json.dump(agreement, f, indent=2)

    logger.info(f"SER proxy agreement: {agreement['ser_proxy_agreement']:.2%} "
                f"(n={agreement['n_samples']}, excluded={agreement['n_excluded']})")
    logger.info("NOTE: SER is an AUXILIARY metric only. Label mismatch limits interpretability.")

    return agreement


def main():
    import yaml, argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="SER probe evaluation")
    parser.add_argument("--config", type=str, default="configs/eval.yaml")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    run_ser_evaluation(cfg)


if __name__ == "__main__":
    main()
