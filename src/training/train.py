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

import gc
import os
import logging
import shutil
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

# Cache mel filterbanks to avoid recomputing on every call
_MEL_BASIS_CACHE: dict[tuple, torch.Tensor] = {}


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

    # Mel filterbank �?cached per (device, sr, n_fft, n_mels, fmin, fmax)
    cache_key = (str(audio.device), sample_rate, n_fft, n_mels, fmin, fmax)
    if cache_key not in _MEL_BASIS_CACHE:
        import librosa
        mel_np = librosa.filters.mel(
            sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax,
        )
        _MEL_BASIS_CACHE[cache_key] = torch.from_numpy(mel_np).float().to(audio.device)
    mel_basis = _MEL_BASIS_CACHE[cache_key]

    mel = torch.matmul(mel_basis, magnitudes)  # (batch, n_mels, frames)
    log_mel = torch.log(torch.clamp(mel, min=1e-5))
    return log_mel


def _compute_linear_spectrogram(
    audio: torch.Tensor,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
) -> torch.Tensor:
    """Compute linear spectrogram (magnitude) from waveform.

    Returns shape ``(batch, n_fft//2+1, frames)`` �?513 bins for n_fft=1024.
    This is what the VITS posterior encoder expects as input.
    """
    window = torch.hann_window(win_length, device=audio.device)
    stft = torch.stft(
        audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
        window=window, return_complex=True,
    )
    magnitudes = stft.abs()  # (batch, n_fft//2+1, frames)
    return magnitudes


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

        # Filter out rows whose audio files don't exist on disk
        path_col = "processed_path" if "processed_path" in self.df.columns else "file_path"
        if path_col in self.df.columns:
            exists_mask = self.df[path_col].apply(lambda p: os.path.isfile(str(p)))
            n_missing = (~exists_mask).sum()
            if n_missing > 0 and n_missing < len(self.df):
                logger.warning(f"Dropping {n_missing}/{len(self.df)} rows with missing audio files")
                self.df = self.df[exists_mask].reset_index(drop=True)
            elif n_missing == len(self.df):
                sample = self.df[path_col].iloc[0]
                raise FileNotFoundError(
                    f"ALL {n_missing} audio files are missing (e.g. {sample}). "
                    f"The manifest exists but processed audio does not. "
                    f"Re-run the data preparation step (Section 1b) to regenerate audio files."
                )

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
        self.lr = train_cfg.get("optimizer", {}).get("lr", 1e-4)
        self.grad_accum_steps = train_cfg.get("grad_accum_steps", 1)
        self.fp16 = train_cfg.get("fp16", True) and torch.cuda.is_available()
        self.max_samples = train_cfg.get("max_samples", None)
        self.save_every = train_cfg.get("save_every", 10)
        self.eval_every = train_cfg.get("eval_every", 5)

        # Early stopping
        es_cfg = train_cfg.get("early_stopping", {})
        self.patience = es_cfg.get("patience", 10)
        self.min_delta = es_cfg.get("min_delta", 0.001)

        # Paths
        ckpt_dir = train_cfg.get("checkpoint", {}).get("save_dir", f"checkpoints/system_{self.system.lower()}")
        self.checkpoint_dir = Path(ckpt_dir)
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
        prosody_cfg = model_cfg.get("prosody", {})
        emotion_cfg = model_cfg.get("emotion", {})
        prosody_heads = None
        if system == "C":
            f0_cfg = prosody_cfg.get("f0_head", {})
            energy_cfg = prosody_cfg.get("energy_head", {})
            prosody_heads = build_prosody_heads(
                input_dim=emotion_cfg.get("embedding_dim", 192),
                hidden_dim=f0_cfg.get("hidden_dim", 128),
                f0_output_dim=f0_cfg.get("output_dim", 4),
                energy_output_dim=energy_cfg.get("output_dim", 2),
            )

        # Read init_from from training config (where YAMLs define it)
        train_cfg = self.cfg.get("training", {})
        init_from = train_cfg.get("init_from", self.cfg.get("init_from"))

        self.model = build_emotion_vits(
            system=system,
            checkpoint_path=init_from,
            use_cuda=(self.device.type == "cuda"),
            num_emotions=emotion_cfg.get("num_emotions", 4),
            embedding_dim=emotion_cfg.get("embedding_dim", 192),
            prosody_heads=prosody_heads,
            prosody_loss_weight=prosody_cfg.get("loss_weight", 0.1),
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

        # num_workers=0 for Colab stability (multiprocessing can deadlock)
        num_workers = self.cfg.get("training", {}).get("num_workers", 0)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=(self.device.type == "cuda"),
            drop_last=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
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

        # --- Compute spectrograms from audio ---
        # The VITS posterior encoder expects a LINEAR spectrogram (513 bins),
        # NOT a mel spectrogram (80 bins).
        linear_spec = _compute_linear_spectrogram(audio).to(self.device)
        spec_lengths = (audio_lengths / 256).long().clamp(min=1)  # hop_length=256

        with torch.amp.autocast("cuda", enabled=self.fp16):
            # Full EmotionVITS forward pass
            outputs = self.model(
                x=x,
                x_lengths=x_lengths,
                y=linear_spec,
                y_lengths=spec_lengths,
                emotion_ids=emotion_ids,
                prosody_targets=prosody_targets,
            )

            # --- Reconstruction loss (monitoring only) ---
            # Decoder & posterior encoder are frozen for all systems,
            # so mel_loss is constant and provides no useful gradients.
            o_hat = outputs["model_outputs"]
            with torch.no_grad():
                o_hat_squeezed = o_hat.squeeze(1)
                min_len = min(o_hat_squeezed.shape[-1], audio.shape[-1])
                mel_hat = _compute_mel_spectrogram(o_hat_squeezed[:, :min_len])
                mel_target = _compute_mel_spectrogram(audio[:, :min_len])
                mel_loss = F.l1_loss(mel_hat, mel_target)

            # --- KL divergence loss ---
            z_p = outputs["z_p"]
            m_p = outputs["m_p"]
            logs_p = outputs["logs_p"]
            m_q = outputs["m_q"]
            logs_q = outputs["logs_q"]

            # Build mask for KL loss
            y_mask = torch.ones(z_p.shape[0], 1, z_p.shape[2], device=z_p.device)
            kl_loss = _kl_loss(z_p, logs_q, m_p, logs_p, y_mask)

            # --- Duration predictor loss ---
            duration_loss = outputs.get("duration_loss", torch.tensor(0.0, device=self.device))

            # --- Prosody loss (System C) ---
            prosody_loss = outputs.get("prosody_loss", torch.tensor(0.0, device=self.device))

            # --- Total loss (mel excluded: frozen decoder) ---
            total_loss = kl_loss + duration_loss + prosody_loss

        losses = {
            "loss_total": total_loss.item(),
            "loss_mel": mel_loss.item(),
            "loss_kl": kl_loss.item(),
            "loss_dur": duration_loss.item() if isinstance(duration_loss, torch.Tensor) else duration_loss,
            "loss_prosody": prosody_loss.item() if isinstance(prosody_loss, torch.Tensor) else prosody_loss,
        }

        # Backward
        if self.scaler:
            self.scaler.scale(total_loss / self.grad_accum_steps).backward()
        else:
            (total_loss / self.grad_accum_steps).backward()

        return losses

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> dict:
        """Run validation loop.

        Args:
            val_loader: Validation DataLoader.

        Returns:
            Dict of average validation losses.
        """
        self.model.eval()
        total_losses = {}
        n_batches = 0

        for batch in val_loader:
            losses = self._val_step(batch)
            for k, v in losses.items():
                total_losses[k] = total_losses.get(k, 0.0) + v
            n_batches += 1

        avg_losses = {k: v / max(n_batches, 1) for k, v in total_losses.items()}
        self.model.train()
        return avg_losses

    @torch.no_grad()
    def _val_step(self, batch: dict) -> dict:
        """Single validation step (no gradient)."""
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

        linear_spec = _compute_linear_spectrogram(audio).to(self.device)
        spec_lengths = (audio_lengths / 256).long().clamp(min=1)

        with torch.amp.autocast("cuda", enabled=self.fp16):
            outputs = self.model(
                x=x, x_lengths=x_lengths, y=linear_spec, y_lengths=spec_lengths,
                emotion_ids=emotion_ids, prosody_targets=prosody_targets,
            )
            o_hat = outputs["model_outputs"]
            o_hat_squeezed = o_hat.squeeze(1)
            min_len = min(o_hat_squeezed.shape[-1], audio.shape[-1])
            mel_hat = _compute_mel_spectrogram(o_hat_squeezed[:, :min_len])
            mel_target = _compute_mel_spectrogram(audio[:, :min_len])
            mel_loss = F.l1_loss(mel_hat, mel_target)

            z_p = outputs["z_p"]
            y_mask = torch.ones(z_p.shape[0], 1, z_p.shape[2], device=z_p.device)
            kl_loss = _kl_loss(z_p, outputs["logs_q"], outputs["m_p"], outputs["logs_p"], y_mask)

            duration_loss = outputs.get("duration_loss", torch.tensor(0.0, device=self.device))
            prosody_loss = outputs.get("prosody_loss", torch.tensor(0.0, device=self.device))
            total_loss = kl_loss + duration_loss + prosody_loss

        return {
            "loss_total": total_loss.item(),
            "loss_mel": mel_loss.item(),
            "loss_kl": kl_loss.item(),
            "loss_dur": duration_loss.item() if isinstance(duration_loss, torch.Tensor) else duration_loss,
            "loss_prosody": prosody_loss.item() if isinstance(prosody_loss, torch.Tensor) else prosody_loss,
        }

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Full training loop with validation, checkpointing, and early stopping.

        Args:
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
        """
        from src.training.callbacks import (
            CheckpointCallback,
            EarlyStoppingCallback,
            MLflowCallback,
        )

        ckpt_cb = CheckpointCallback(
            checkpoint_dir=str(self.checkpoint_dir),
            save_every=self.save_every,
            keep_last=3,
        )
        es_cb = EarlyStoppingCallback(
            patience=self.patience,
            min_delta=self.min_delta,
        )
        mlflow_cb = MLflowCallback(
            experiment_name="emotive-tts",
            run_name=f"system_{self.system.lower()}",
        )
        mlflow_cb.setup(self.cfg)

        logger.info(f"Starting training: System {self.system}, {self.max_epochs} epochs, "
                     f"batch_size={self.batch_size}, lr={self.lr}")

        for epoch in range(1, self.max_epochs + 1):
            self.model.train()
            epoch_losses = {}
            n_batches = 0

            for batch_idx, batch in enumerate(train_loader):
                losses = self.train_step(batch)
                for k, v in losses.items():
                    epoch_losses[k] = epoch_losses.get(k, 0.0) + v
                n_batches += 1
                self.global_step += 1

                # Gradient step (with accumulation)
                if (batch_idx + 1) % self.grad_accum_steps == 0:
                    if self.cfg.get("training", {}).get("gradient_clip_val"):
                        clip_val = self.cfg["training"]["gradient_clip_val"]
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in self.model.parameters() if p.requires_grad],
                            clip_val,
                        )
                    if self.scaler:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad()

            # Average epoch losses
            avg_train = {k: v / max(n_batches, 1) for k, v in epoch_losses.items()}

            # Validation (every epoch for reliable early stopping)
            avg_val = self.validate(val_loader)

            # LR schedule
            if self.scheduler:
                self.scheduler.step()

            # Logging
            log_metrics = {f"train/{k}": v for k, v in avg_train.items()}
            log_metrics.update({f"val/{k}": v for k, v in avg_val.items()})
            log_metrics["lr"] = self.optimizer.param_groups[0]["lr"]
            mlflow_cb.log_metrics(log_metrics, step=epoch)

            val_loss = avg_val.get("loss_total", avg_train.get("loss_total", 0))
            logger.info(
                f"Epoch {epoch}/{self.max_epochs} -- "
                f"train_loss={avg_train.get('loss_total', 0):.4f}, "
                f"val_loss={val_loss:.4f}, "
                f"kl={avg_train.get('loss_kl', 0):.4f}, "
                f"dur={avg_train.get('loss_dur', 0):.4f}, "
                f"mel={avg_train.get('loss_mel', 0):.4f}, "
                f"lr={self.optimizer.param_groups[0]['lr']:.2e}"
            )

            # Checkpointing
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss

            if ckpt_cb.should_save(epoch) or is_best:
                ckpt_cb.save(
                    model=self.model, optimizer=self.optimizer,
                    epoch=epoch, step=self.global_step,
                    val_loss=val_loss, is_best=is_best,
                )

            # Early stopping
            if es_cb.should_stop(val_loss):
                logger.info(f"Early stopping at epoch {epoch}")
                break

        # Save final checkpoint
        ckpt_cb.save(
            model=self.model, optimizer=self.optimizer,
            epoch=epoch, step=self.global_step,
            val_loss=val_loss, is_best=False,
        )
        mlflow_cb.end()
        logger.info("Training complete")


def train(cfg: dict):
    """High-level entry point for training.

    Args:
        cfg: Full config dict (from train_a/b/c.yaml).
    """
    # Seed
    seed = cfg.get("training", {}).get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    trainer = Trainer(cfg)
    trainer.build_model()
    train_loader, val_loader = trainer.build_dataloaders()
    trainer.train(train_loader, val_loader)


def main():
    """CLI entry point."""
    import argparse
    import yaml

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Train EmotionVITS systems")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to training config YAML (e.g., configs/train_b.yaml)")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    train(cfg)


if __name__ == "__main__":
    main()
