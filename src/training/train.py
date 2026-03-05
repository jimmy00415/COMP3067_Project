"""Training pipeline for Systems A, B, C (FR-2).

Supports:
- System A: Domain adaptation on EmoV-DB, no emotion labels
- System B: A + emotion embedding conditioning
- System C: B + utterance-level prosody auxiliary heads

Uses Hydra configs (train_a.yaml, train_b.yaml, train_c.yaml).
Logs to MLflow. Saves checkpoints. Runs on Colab T4 (ADR-8).

Usage:
    python -m src.training.train --config configs/train_b.yaml

Or from Colab:
    from src.training.train import train
    train(cfg)
"""

import os
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.data.utils import EMOTION_MAP, compute_utterance_prosody_stats
from src.models.emotion_vits import EmotionVITS, build_emotion_vits, load_pretrained_vits
from src.models.prosody_heads import build_prosody_heads

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Audio helpers for training loop
# ---------------------------------------------------------------------------

def _compute_mel_spectrogram(
    audio: torch.Tensor,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
    n_mels: int = 80,
    sample_rate: int = 22050,
    fmin: float = 0.0,
    fmax: float = 8000.0,
) -> torch.Tensor:
    """Compute log-mel spectrogram from waveform tensor.

    Args:
        audio: Waveform tensor of shape ``(batch, samples)``.

    Returns:
        Log-mel spectrogram of shape ``(batch, n_mels, frames)``.
    """
    # STFT
    window = torch.hann_window(win_length, device=audio.device)
    stft = torch.stft(
        audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
        window=window, return_complex=True,
    )
    magnitudes = stft.abs()  # (batch, n_fft//2+1, frames)

    # Mel filterbank
    import librosa
    mel_basis = librosa.filters.mel(
        sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax,
    )
    mel_basis = torch.from_numpy(mel_basis).float().to(audio.device)

    mel = torch.matmul(mel_basis, magnitudes)  # (batch, n_mels, frames)
    log_mel = torch.log(torch.clamp(mel, min=1e-5))
    return log_mel


def _kl_loss(
    z_p: torch.Tensor,
    logs_q: torch.Tensor,
    m_p: torch.Tensor,
    logs_p: torch.Tensor,
    z_mask: torch.Tensor,
) -> torch.Tensor:
    """KL divergence between posterior and prior (standard VITS KL loss).

    All tensors have shape ``(batch, hidden_dim, time)``.
    """
    kl = logs_p - logs_q - 0.5 + 0.5 * (
        (z_p - m_p) ** 2 * torch.exp(-2.0 * logs_p)
        + torch.exp(2.0 * (logs_q - logs_p))
    )
    kl = torch.sum(kl * z_mask)
    loss = kl / torch.sum(z_mask)
    return loss


class EmotiveTTSDataset(Dataset):
    """Dataset for EmotionVITS training.

    Loads processed audio files and their associated metadata from
    the manifest CSV created by the data preparation pipeline.
    """

    def __init__(
        self,
        manifest_path: str,
        max_samples: Optional[int] = None,
        sr: int = 22050,
        max_audio_len: int = 22050 * 10,  # 10 seconds max
    ):
        """Initialize dataset.

        Args:
            manifest_path: Path to CSV manifest (train.csv / val.csv).
            max_samples: Limit number of samples (for debugging).
            sr: Sample rate.
            max_audio_len: Maximum audio length in samples.
        """
        self.df = pd.read_csv(manifest_path)
        if max_samples and max_samples > 0:
            self.df = self.df.head(max_samples)
        self.sr = sr
        self.max_audio_len = max_audio_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load processed audio
        audio_path = row.get("processed_path", row.get("file_path", ""))
        try:
            import librosa
            audio, _ = librosa.load(audio_path, sr=self.sr, mono=True)
        except Exception as e:
            logger.error(f"Failed to load {audio_path}: {e}")
            audio = np.zeros(self.sr, dtype=np.float32)  # 1 second of silence

        # Truncate if too long
        if len(audio) > self.max_audio_len:
            audio = audio[:self.max_audio_len]

        # Text
        text = str(row.get("text", ""))

        # Emotion
        emotion = row.get("emotion", "neutral")
        emotion_id = EMOTION_MAP.get(emotion, 0)

        # Prosody targets (for System C)
        prosody_targets = {}
        if "f0_mean" in row:
            prosody_targets["f0_stats"] = np.array([
                row.get("f0_mean", 0),
                row.get("f0_std", 0),
                row.get("f0_range_low", 0),
                row.get("f0_range_high", 0),
            ], dtype=np.float32)
        if "energy_mean" in row:
            prosody_targets["energy_stats"] = np.array([
                row.get("energy_mean", 0),
                row.get("energy_std", 0),
            ], dtype=np.float32)

        return {
            "audio": torch.FloatTensor(audio),
            "text": text,
            "emotion_id": emotion_id,
            "prosody_targets": prosody_targets,
            "audio_path": audio_path,
        }


def collate_fn(batch):
    """Custom collate function for variable-length audio."""
    # Find max audio length in batch
    max_len = max(item["audio"].size(0) for item in batch)

    # Pad audio
    audio_padded = torch.zeros(len(batch), max_len)
    audio_lengths = torch.zeros(len(batch), dtype=torch.long)

    texts = []
    emotion_ids = torch.zeros(len(batch), dtype=torch.long)

    f0_stats_list = []
    energy_stats_list = []
    has_prosody = "f0_stats" in batch[0]["prosody_targets"]

    for i, item in enumerate(batch):
        audio = item["audio"]
        audio_padded[i, :audio.size(0)] = audio
        audio_lengths[i] = audio.size(0)
        texts.append(item["text"])
        emotion_ids[i] = item["emotion_id"]

        if has_prosody:
            f0_stats_list.append(torch.FloatTensor(item["prosody_targets"]["f0_stats"]))
            energy_stats_list.append(torch.FloatTensor(item["prosody_targets"]["energy_stats"]))

    result = {
        "audio": audio_padded,
        "audio_lengths": audio_lengths,
        "texts": texts,
        "emotion_ids": emotion_ids,
    }

    if has_prosody:
        result["f0_stats"] = torch.stack(f0_stats_list)
        result["energy_stats"] = torch.stack(energy_stats_list)

    return result


class Trainer:
    """Training loop for EmotionVITS systems.

    Handles:
    - Building model + optimizer
    - Training loop with gradient accumulation
    - Validation
    - Checkpoint saving
    - MLflow logging
    - Early stopping
    """

    def __init__(self, cfg: dict):
        """Initialize trainer from config dict.

        Args:
            cfg: Configuration dict (from train_a/b/c.yaml).
        """
        self.cfg = cfg
        self.system = cfg.get("system", "A")
        self.device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("use_cuda", True) else "cpu")

        # Training hyperparams
        train_cfg = cfg.get("training", {})
        self.max_epochs = train_cfg.get("max_epochs", 100)
        self.batch_size = train_cfg.get("batch_size", 16)
        self.lr = train_cfg.get("lr", 1e-4)
        self.grad_accum_steps = train_cfg.get("grad_accum_steps", 1)
        self.fp16 = train_cfg.get("fp16", True) and torch.cuda.is_available()
        self.max_samples = train_cfg.get("max_samples", None)
        self.save_every = train_cfg.get("save_every", 10)
        self.eval_every = train_cfg.get("eval_every", 5)

        # Early stopping
        es_cfg = cfg.get("early_stopping", {})
        self.patience = es_cfg.get("patience", 10)
        self.min_delta = es_cfg.get("min_delta", 0.001)

        # Paths
        self.checkpoint_dir = Path(cfg.get("checkpoint_dir", f"checkpoints/system_{self.system.lower()}"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Model
        self.model = None
        self.optimizer = None
        self.scaler = None
        self.scheduler = None

        # Tracking
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.global_step = 0

    def build_model(self):
        """Build EmotionVITS model based on system config."""
        model_cfg = self.cfg.get("model", {})
        system = self.system.upper()

        # Prosody heads for System C
        prosody_heads = None
        if system == "C":
            ph_cfg = self.cfg.get("prosody_heads", {})
            prosody_heads = build_prosody_heads(
                input_dim=model_cfg.get("embedding_dim", 192),
                hidden_dim=ph_cfg.get("hidden_dim", 128),
                f0_output_dim=ph_cfg.get("f0_output_dim", 4),
                energy_output_dim=ph_cfg.get("energy_output_dim", 2),
            )

        self.model = build_emotion_vits(
            system=system,
            checkpoint_path=self.cfg.get("init_from"),
            use_cuda=(self.device.type == "cuda"),
            num_emotions=model_cfg.get("num_emotions", 4),
            embedding_dim=model_cfg.get("embedding_dim", 192),
            prosody_heads=prosody_heads,
            prosody_loss_weight=self.cfg.get("prosody_heads", {}).get("loss_weight", 0.1),
        )

        self.model = self.model.to(self.device)

        # Optimizer (only trainable params)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.lr,
            betas=(0.8, 0.99),
            eps=1e-9,
            weight_decay=0.01,
        )

        # Mixed precision
        if self.fp16:
            self.scaler = torch.amp.GradScaler("cuda")
        else:
            self.scaler = None

        # LR scheduler
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer,
            gamma=self.cfg.get("training", {}).get("lr_decay", 0.999),
        )

        params = self.model.count_parameters()
        logger.info(f"Model built: {params}")

    def build_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        """Create train and validation dataloaders."""
        data_cfg = self.cfg.get("data", {})
        manifests_dir = data_cfg.get("manifests_dir", "data/manifests")

        train_dataset = EmotiveTTSDataset(
            manifest_path=os.path.join(manifests_dir, "train.csv"),
            max_samples=self.max_samples,
            sr=data_cfg.get("sample_rate", 22050),
        )

        val_dataset = EmotiveTTSDataset(
            manifest_path=os.path.join(manifests_dir, "val.csv"),
            max_samples=self.max_samples,
            sr=data_cfg.get("sample_rate", 22050),
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=2,
            pin_memory=(self.device.type == "cuda"),
            drop_last=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=2,
            pin_memory=(self.device.type == "cuda"),
        )

        logger.info(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")
        return train_loader, val_loader

    def train_step(self, batch: dict) -> dict:
        """Single training step with real VITS loss computation.

        Computes:
        - Mel reconstruction loss (L1)
        - KL divergence loss (posterior vs prior)
        - Prosody auxiliary loss (System C only, weighted by λ)

        Args:
            batch: Collated batch dict.

        Returns:
            Dict of loss values.
        """
        self.model.train()

        # Move to device
        audio = batch["audio"].to(self.device)
        audio_lengths = batch["audio_lengths"].to(self.device)
        emotion_ids = batch["emotion_ids"].to(self.device) if self.system.upper() != "A" else None

        # Prosody targets for System C
        prosody_targets = None
        if self.system.upper() == "C" and "f0_stats" in batch:
            prosody_targets = {
                "f0_stats": batch["f0_stats"].to(self.device),
                "energy_stats": batch["energy_stats"].to(self.device),
            }

        texts = batch["texts"]

        # --- Tokenize text using VITS tokenizer ---
        try:
            tokenizer = self.model.vits.tokenizer
            text_tokens = []
            for t in texts:
                ids = tokenizer.text_to_ids(t)
                text_tokens.append(torch.LongTensor(ids))
        except (AttributeError, Exception):
            # Fallback: use character-level tokenization
            text_tokens = []
            for t in texts:
                ids = [ord(c) % 256 for c in t]
                text_tokens.append(torch.LongTensor(ids))

        # Pad text tokens
        max_text_len = max(len(t) for t in text_tokens)
        x = torch.zeros(len(text_tokens), max_text_len, dtype=torch.long, device=self.device)
        x_lengths = torch.zeros(len(text_tokens), dtype=torch.long, device=self.device)
        for i, t in enumerate(text_tokens):
            x[i, :len(t)] = t.to(self.device)
            x_lengths[i] = len(t)

        # --- Compute mel spectrogram from audio ---
        mel = _compute_mel_spectrogram(audio).to(self.device)
        mel_lengths = (audio_lengths / 256).long().clamp(min=1)  # hop_length=256

        with torch.amp.autocast("cuda", enabled=self.fp16):
            # Full EmotionVITS forward pass
            outputs = self.model(
                x=x,
                x_lengths=x_lengths,
                y=mel,
                y_lengths=mel_lengths,
                emotion_ids=emotion_ids,
                prosody_targets=prosody_targets,
            )

            # --- Reconstruction loss: L1 on mel spectrogram ---
            o_hat = outputs["model_outputs"]  # Generated audio
            # Compute mel of generated audio
            o_hat_squeezed = o_hat.squeeze(1)  # (batch, samples)
            # Trim/pad to match target length
            min_len = min(o_hat_squeezed.shape[-1], audio.shape[-1])
            mel_hat = _compute_mel_spectrogram(o_hat_squeezed[:, :min_len])
            mel_target = _compute_mel_spectrogram(audio[:, :min_len])
            # L1 on mel
            mel_min_frames = min(mel_hat.shape[-1], mel_target.shape[-1])
            recon_loss = F.l1_loss(
                mel_hat[:, :, :mel_min_frames],
                mel_target[:, :, :mel_min_frames],
            )

            # --- KL loss ---
            kl_loss = torch.tensor(0.0, device=self.device)
            if all(k in outputs for k in ("z_p", "m_p", "logs_p", "m_q", "logs_q")):
                z_p = outputs["z_p"]
                m_p = outputs["m_p"]
                logs_p = outputs["logs_p"]
                logs_q = outputs["logs_q"]
                # Create mask from mel_lengths
                max_mel_len = z_p.shape[-1]
                z_mask = torch.arange(max_mel_len, device=self.device).unsqueeze(0) < mel_lengths.unsqueeze(1)
                z_mask = z_mask.unsqueeze(1).float()  # (batch, 1, time)
                kl_loss = _kl_loss(z_p, logs_q, m_p, logs_p, z_mask)

            # --- Prosody auxiliary loss (System C) ---
            prosody_loss = outputs.get("prosody_loss", torch.tensor(0.0, device=self.device))

            # --- Total loss ---
            kl_weight = self.cfg.get("training", {}).get("kl_weight", 1.0)
            loss = recon_loss + kl_weight * kl_loss + prosody_loss

        losses = {
            "total_loss": loss.item(),
            "recon_loss": recon_loss.item(),
            "kl_loss": kl_loss.item(),
        }
        if self.system.upper() == "C":
            losses["prosody_loss"] = prosody_loss.item()

        # Backward
        if self.scaler:
            self.scaler.scale(loss).backward()
            if (self.global_step + 1) % self.grad_accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
        else:
            loss.backward()
            if (self.global_step + 1) % self.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()

        self.global_step += 1
        return losses

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> dict:
        """Run validation loop.

        Computes the same losses as ``train_step`` but without backprop.

        Returns:
            Dict of averaged validation metrics.
        """
        self.model.eval()
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        total_prosody = 0.0
        n_batches = 0

        for batch in val_loader:
            audio = batch["audio"].to(self.device)
            audio_lengths = batch["audio_lengths"].to(self.device)
            emotion_ids = batch["emotion_ids"].to(self.device) if self.system.upper() != "A" else None

            prosody_targets = None
            if self.system.upper() == "C" and "f0_stats" in batch:
                prosody_targets = {
                    "f0_stats": batch["f0_stats"].to(self.device),
                    "energy_stats": batch["energy_stats"].to(self.device),
                }

            texts = batch["texts"]

            # Tokenize
            try:
                tokenizer = self.model.vits.tokenizer
                text_tokens = [torch.LongTensor(tokenizer.text_to_ids(t)) for t in texts]
            except (AttributeError, Exception):
                text_tokens = [torch.LongTensor([ord(c) % 256 for c in t]) for t in texts]

            max_text_len = max(len(t) for t in text_tokens)
            x = torch.zeros(len(text_tokens), max_text_len, dtype=torch.long, device=self.device)
            x_lengths = torch.zeros(len(text_tokens), dtype=torch.long, device=self.device)
            for i, t in enumerate(text_tokens):
                x[i, :len(t)] = t.to(self.device)
                x_lengths[i] = len(t)

            mel = _compute_mel_spectrogram(audio).to(self.device)
            mel_lengths = (audio_lengths / 256).long().clamp(min=1)

            with torch.amp.autocast("cuda", enabled=self.fp16):
                outputs = self.model(
                    x=x, x_lengths=x_lengths,
                    y=mel, y_lengths=mel_lengths,
                    emotion_ids=emotion_ids,
                    prosody_targets=prosody_targets,
                )

                # Reconstruction loss
                o_hat = outputs["model_outputs"].squeeze(1)
                min_len = min(o_hat.shape[-1], audio.shape[-1])
                mel_hat = _compute_mel_spectrogram(o_hat[:, :min_len])
                mel_target = _compute_mel_spectrogram(audio[:, :min_len])
                mel_min_frames = min(mel_hat.shape[-1], mel_target.shape[-1])
                recon_loss = F.l1_loss(
                    mel_hat[:, :, :mel_min_frames],
                    mel_target[:, :, :mel_min_frames],
                )

                # KL loss
                kl_loss = torch.tensor(0.0, device=self.device)
                if all(k in outputs for k in ("z_p", "m_p", "logs_p", "m_q", "logs_q")):
                    z_p = outputs["z_p"]
                    max_mel_len = z_p.shape[-1]
                    z_mask = (torch.arange(max_mel_len, device=self.device).unsqueeze(0) < mel_lengths.unsqueeze(1))
                    z_mask = z_mask.unsqueeze(1).float()
                    kl_loss = _kl_loss(z_p, outputs["logs_q"], outputs["m_p"], outputs["logs_p"], z_mask)

                prosody_loss = outputs.get("prosody_loss", torch.tensor(0.0, device=self.device))

                kl_weight = self.cfg.get("training", {}).get("kl_weight", 1.0)
                loss = recon_loss + kl_weight * kl_loss + prosody_loss

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            total_prosody += prosody_loss.item()
            n_batches += 1

        denom = max(n_batches, 1)
        metrics = {
            "val_loss": total_loss / denom,
            "val_recon_loss": total_recon / denom,
            "val_kl_loss": total_kl / denom,
        }
        if self.system.upper() == "C":
            metrics["val_prosody_loss"] = total_prosody / denom
        return metrics

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "vits_state_dict": self.model.vits.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "system": self.system,
            "config": self.cfg,
        }

        # Save emotion embedding separately for easy transfer
        if self.model.emotion_embedding is not None:
            state["emotion_embedding_state"] = self.model.emotion_embedding.state_dict()

        # Save prosody heads separately
        if self.model.prosody_heads is not None:
            state["prosody_heads_state"] = self.model.prosody_heads.state_dict()

        # Regular checkpoint
        path = self.checkpoint_dir / f"epoch_{epoch:04d}.pth"
        torch.save(state, path)
        logger.info(f"Saved checkpoint: {path}")

        # Best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best.pth"
            torch.save(state, best_path)
            logger.info(f"Saved best checkpoint: {best_path}")

    def train(self):
        """Full training loop."""
        logger.info(f"Starting training for System {self.system}")
        logger.info(f"Device: {self.device}, FP16: {self.fp16}")
        logger.info(f"Max epochs: {self.max_epochs}, Batch size: {self.batch_size}")
        logger.info(f"LR: {self.lr}, Patience: {self.patience}")

        # Build model and data
        self.build_model()
        train_loader, val_loader = self.build_dataloaders()

        # Optional: MLflow tracking
        mlflow_enabled = False
        try:
            import mlflow
            experiment_name = self.cfg.get("mlflow", {}).get("experiment", f"emotive-tts-system-{self.system}")
            mlflow.set_experiment(experiment_name)
            mlflow.start_run(run_name=f"system_{self.system}_{int(time.time())}")
            mlflow.log_params({
                "system": self.system,
                "lr": self.lr,
                "batch_size": self.batch_size,
                "max_epochs": self.max_epochs,
                "fp16": self.fp16,
            })
            mlflow_enabled = True
            logger.info("MLflow tracking enabled")
        except Exception as e:
            logger.warning(f"MLflow not available: {e}. Training without tracking.")

        # Training loop
        for epoch in range(1, self.max_epochs + 1):
            epoch_start = time.time()
            epoch_losses = []

            for batch_idx, batch in enumerate(train_loader):
                losses = self.train_step(batch)
                epoch_losses.append(losses["total_loss"])

                if batch_idx % 50 == 0:
                    parts = [f"Loss: {losses['total_loss']:.4f}"]
                    if "recon_loss" in losses:
                        parts.append(f"Recon: {losses['recon_loss']:.4f}")
                    if "kl_loss" in losses:
                        parts.append(f"KL: {losses['kl_loss']:.4f}")
                    if "prosody_loss" in losses:
                        parts.append(f"Pros: {losses['prosody_loss']:.4f}")
                    logger.info(
                        f"Epoch {epoch}/{self.max_epochs}, "
                        f"Batch {batch_idx}/{len(train_loader)}, "
                        + ", ".join(parts)
                    )

            # Epoch stats
            avg_train_loss = np.mean(epoch_losses) if epoch_losses else 0.0
            epoch_time = time.time() - epoch_start

            # Validation
            if epoch % self.eval_every == 0 or epoch == self.max_epochs:
                val_metrics = self.validate(val_loader)
                val_loss = val_metrics["val_loss"]

                logger.info(
                    f"Epoch {epoch}: train_loss={avg_train_loss:.4f}, "
                    f"val_loss={val_loss:.4f}, time={epoch_time:.1f}s"
                )

                # MLflow logging
                if mlflow_enabled:
                    log_dict = {
                        "train_loss": avg_train_loss,
                        "val_loss": val_loss,
                        "epoch_time": epoch_time,
                        "lr": self.optimizer.param_groups[0]["lr"],
                    }
                    # Add detailed val metrics
                    for k, v in val_metrics.items():
                        if k != "val_loss":
                            log_dict[k] = v
                    mlflow.log_metrics(log_dict, step=epoch)

                # Early stopping check
                if val_loss < self.best_val_loss - self.min_delta:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self.save_checkpoint(epoch, is_best=True)
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.patience:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break

            # Regular checkpoint
            if epoch % self.save_every == 0:
                self.save_checkpoint(epoch, is_best=False)

            # LR decay
            self.scheduler.step()

        # Save final checkpoint
        self.save_checkpoint(epoch, is_best=False)

        if mlflow_enabled:
            mlflow.end_run()

        logger.info(f"Training complete. Best val loss: {self.best_val_loss:.4f}")
        return {"best_val_loss": self.best_val_loss, "epochs_trained": epoch}


def train_with_coqui(cfg: dict) -> None:
    """Train using Coqui's native Trainer (recommended approach).

    This leverages Coqui TTS's built-in training infrastructure,
    which handles spectrogram computation, discriminator training,
    and all the VITS-specific details properly.

    The key modifications are applied via config overrides and
    model surgery (adding emotion embedding + prosody heads).

    Args:
        cfg: Training configuration dict.
    """
    from TTS.tts.configs.vits_config import VitsConfig
    from TTS.tts.models.vits import Vits
    from TTS.trainer import Trainer as CoquiTrainer, TrainerArgs

    system = cfg.get("system", "A")
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})

    # Build Coqui VITS config
    vits_config = VitsConfig(
        run_name=f"emotive_tts_system_{system}",
        output_path=str(cfg.get("checkpoint_dir", f"checkpoints/system_{system.lower()}")),
        batch_size=train_cfg.get("batch_size", 16),
        eval_batch_size=train_cfg.get("batch_size", 16),
        num_loader_workers=2,
        epochs=train_cfg.get("max_epochs", 100),
        lr_gen=train_cfg.get("lr", 1e-4),
        lr_disc=train_cfg.get("lr", 1e-4),
        mixed_precision=train_cfg.get("fp16", True),
    )

    # Data paths
    vits_config.datasets = [{
        "name": "emovdb",
        "path": data_cfg.get("processed_dir", "data/processed/train"),
        "meta_file_train": os.path.join(data_cfg.get("manifests_dir", "data/manifests"), "train.csv"),
        "meta_file_val": os.path.join(data_cfg.get("manifests_dir", "data/manifests"), "val.csv"),
    }]

    # Initialize model
    model = Vits.init_from_config(vits_config)

    # Load pretrained weights
    init_from = cfg.get("init_from")
    if init_from == "pretrained":
        logger.info("Initializing from pretrained LJSpeech VITS")
        # Download and load pretrained weights
        from TTS.api import TTS
        pretrained = TTS(model_name="tts_models/en/ljspeech/vits", gpu=False)
        pretrained_state = pretrained.synthesizer.tts_model.state_dict()
        model.load_state_dict(pretrained_state, strict=False)
    elif init_from and Path(init_from).exists():
        logger.info(f"Initializing from checkpoint: {init_from}")
        state = torch.load(init_from, map_location="cpu", weights_only=False)
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"], strict=False)
        elif "vits_state_dict" in state:
            model.load_state_dict(state["vits_state_dict"], strict=False)

    # Apply freeze strategy
    freeze_cfg = cfg.get("freeze", {})
    modules_to_freeze = freeze_cfg.get("modules", [])
    modules_to_unfreeze = freeze_cfg.get("unfreeze", [])

    for name, param in model.named_parameters():
        should_freeze = any(m in name for m in modules_to_freeze)
        should_unfreeze = any(m in name for m in modules_to_unfreeze)
        param.requires_grad = not should_freeze or should_unfreeze

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    # Launch Coqui trainer
    trainer = CoquiTrainer(
        TrainerArgs(),
        vits_config,
        output_path=vits_config.output_path,
        model=model,
    )
    trainer.fit()


def train(cfg: dict) -> dict:
    """Main entry point for training.

    Args:
        cfg: Configuration dict.

    Returns:
        Training results dict.
    """
    use_coqui_trainer = cfg.get("use_coqui_trainer", False)

    if use_coqui_trainer:
        train_with_coqui(cfg)
        return {"method": "coqui_trainer"}
    else:
        trainer = Trainer(cfg)
        return trainer.train()


def main():
    """CLI entry point."""
    import yaml
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Train EmotionVITS")
    parser.add_argument("--config", type=str, required=True, help="Training config YAML")
    parser.add_argument("--use-coqui-trainer", action="store_true",
                        help="Use Coqui's native Trainer instead of custom loop")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if args.use_coqui_trainer:
        cfg["use_coqui_trainer"] = True

    train(cfg)


if __name__ == "__main__":
    main()
