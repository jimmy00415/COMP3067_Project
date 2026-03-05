"""Training callbacks: MLflow logging, checkpointing, audio sampling.

Designed to work with both custom training loop and Coqui trainer.
"""

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class MLflowCallback:
    """MLflow experiment tracking callback.

    Logs metrics, parameters, artifacts, and audio samples during training.
    """

    def __init__(
        self,
        experiment_name: str = "emotive-tts",
        run_name: Optional[str] = None,
        tracking_uri: Optional[str] = None,
        log_audio_every: int = 10,
    ):
        self.experiment_name = experiment_name
        self.run_name = run_name or f"run_{int(time.time())}"
        self.tracking_uri = tracking_uri
        self.log_audio_every = log_audio_every
        self._mlflow = None
        self._active = False

    def setup(self, config: dict):
        """Initialize MLflow run."""
        try:
            import mlflow
            self._mlflow = mlflow

            if self.tracking_uri:
                mlflow.set_tracking_uri(self.tracking_uri)

            mlflow.set_experiment(self.experiment_name)
            mlflow.start_run(run_name=self.run_name)

            # Log config as params (flatten nested dict)
            flat_cfg = self._flatten_dict(config)
            # MLflow has a param value length limit
            for k, v in flat_cfg.items():
                try:
                    mlflow.log_param(k, str(v)[:250])
                except Exception:
                    pass

            self._active = True
            logger.info(f"MLflow run started: {self.run_name}")
        except ImportError:
            logger.warning("MLflow not installed — tracking disabled")
        except Exception as e:
            logger.warning(f"MLflow setup failed: {e}")

    def log_metrics(self, metrics: dict, step: int):
        """Log scalar metrics."""
        if not self._active:
            return
        try:
            self._mlflow.log_metrics(metrics, step=step)
        except Exception as e:
            logger.debug(f"MLflow log_metrics failed: {e}")

    def log_audio(self, audio: np.ndarray, sr: int, name: str, step: int):
        """Log audio sample as artifact."""
        if not self._active:
            return
        if step % self.log_audio_every != 0:
            return
        try:
            import soundfile as sf
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, audio, sr)
                self._mlflow.log_artifact(f.name, artifact_path=f"audio/step_{step}")
        except Exception as e:
            logger.debug(f"MLflow log_audio failed: {e}")

    def log_figure(self, fig, name: str, step: int):
        """Log matplotlib figure."""
        if not self._active:
            return
        try:
            self._mlflow.log_figure(fig, f"figures/{name}_step{step}.png")
        except Exception as e:
            logger.debug(f"MLflow log_figure failed: {e}")

    def end(self):
        """End MLflow run."""
        if self._active:
            try:
                self._mlflow.end_run()
                logger.info("MLflow run ended")
            except Exception:
                pass
            self._active = False

    @staticmethod
    def _flatten_dict(d: dict, prefix: str = "") -> dict:
        """Flatten nested dict for MLflow param logging."""
        items = {}
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                items.update(MLflowCallback._flatten_dict(v, key))
            else:
                items[key] = v
        return items


class CheckpointCallback:
    """Manage checkpoint saving and restoration."""

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        save_every: int = 10,
        keep_last: int = 3,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_every = save_every
        self.keep_last = keep_last
        self._saved_checkpoints = []

    def should_save(self, epoch: int) -> bool:
        """Check if checkpoint should be saved this epoch."""
        return epoch % self.save_every == 0

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        step: int,
        val_loss: float,
        is_best: bool = False,
        extra: Optional[dict] = None,
    ):
        """Save checkpoint."""
        state = {
            "epoch": epoch,
            "global_step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
        }
        if extra:
            state.update(extra)

        # Regular checkpoint
        path = self.checkpoint_dir / f"checkpoint_epoch{epoch:04d}.pth"
        torch.save(state, path)
        self._saved_checkpoints.append(path)
        logger.info(f"Checkpoint saved: {path}")

        # Best model
        if is_best:
            best_path = self.checkpoint_dir / "best.pth"
            torch.save(state, best_path)
            logger.info(f"Best model saved: {best_path}")

        # Cleanup old checkpoints
        while len(self._saved_checkpoints) > self.keep_last:
            old = self._saved_checkpoints.pop(0)
            if old.exists() and old.name != "best.pth":
                old.unlink()
                logger.debug(f"Removed old checkpoint: {old}")

    def load_best(self, model: torch.nn.Module) -> dict:
        """Load best checkpoint into model."""
        best_path = self.checkpoint_dir / "best.pth"
        if not best_path.exists():
            raise FileNotFoundError(f"No best checkpoint at {best_path}")

        state = torch.load(best_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state_dict"], strict=False)
        logger.info(f"Loaded best checkpoint from epoch {state.get('epoch', '?')}")
        return state


class EarlyStoppingCallback:
    """Early stopping based on validation loss."""

    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0

    def should_stop(self, val_loss: float) -> bool:
        """Check if training should stop.

        Args:
            val_loss: Current validation loss.

        Returns:
            True if should stop.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                logger.info(f"Early stopping triggered (patience={self.patience})")
                return True
            return False

    @property
    def is_best(self) -> bool:
        """Whether the last check was a new best."""
        return self.counter == 0


class AudioSamplingCallback:
    """Generate and save audio samples during training for monitoring."""

    def __init__(
        self,
        canary_texts: list[dict],
        emotions: list[str],
        output_dir: str = "outputs/training_samples",
        sample_every: int = 10,
    ):
        self.canary_texts = canary_texts[:4]  # Use first 4 canaries
        self.emotions = emotions
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sample_every = sample_every

    def should_sample(self, epoch: int) -> bool:
        return epoch % self.sample_every == 0

    @torch.no_grad()
    def generate_samples(
        self,
        model,
        epoch: int,
        device: torch.device,
    ) -> list[dict]:
        """Generate audio samples from canary texts.

        Uses the model's ``infer()`` method (or the wrapped VITS synthesizer)
        to produce waveforms for a small set of canary texts × emotions.

        Args:
            model: EmotionVITS model (in eval mode).
            epoch: Current epoch number.
            device: Device to generate on.

        Returns:
            List of sample metadata dicts.
        """
        model.eval()
        samples = []
        sr = 22050

        for text_info in self.canary_texts:
            for emotion in self.emotions:
                text = text_info["text"]
                text_id = text_info["id"]

                try:
                    from src.data.utils import EMOTION_MAP

                    # Tokenize text
                    try:
                        tokenizer = model.vits.tokenizer
                        token_ids = tokenizer.text_to_ids(text)
                    except (AttributeError, Exception):
                        token_ids = [ord(c) % 256 for c in text]

                    x = torch.LongTensor([token_ids]).to(device)
                    x_lengths = torch.LongTensor([len(token_ids)]).to(device)
                    emotion_id = EMOTION_MAP.get(emotion, 0)
                    emotion_ids = torch.LongTensor([emotion_id]).to(device)

                    wav = model.infer(
                        x=x,
                        x_lengths=x_lengths,
                        emotion_ids=emotion_ids if model.use_emotion else None,
                    )

                    # Save audio
                    wav_np = wav.squeeze().cpu().numpy()
                    out_path = self.output_dir / f"epoch{epoch:04d}_{text_id}_{emotion}.wav"
                    import soundfile as sf
                    sf.write(str(out_path), wav_np, sr)

                    sample_info = {
                        "epoch": epoch,
                        "text_id": text_id,
                        "text": text,
                        "emotion": emotion,
                        "path": str(out_path),
                        "status": "ok",
                    }
                    samples.append(sample_info)
                except Exception as e:
                    logger.warning(f"Sample generation failed for '{text}' [{emotion}]: {e}")
                    samples.append({
                        "epoch": epoch,
                        "text_id": text_id,
                        "text": text,
                        "emotion": emotion,
                        "status": f"error: {e}",
                    })

        return samples
