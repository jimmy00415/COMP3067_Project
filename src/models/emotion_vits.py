"""EmotionVITS model — Systems B and C (FR-2, ADR-2, ADR-4).

Extends the Coqui VITS architecture with:
- System B: nn.Embedding(4, 192) for emotion conditioning (additive injection
  at text encoder output)
- System C: B + utterance-level prosody auxiliary heads (F0 stats + energy stats)

The causal chain is A0 → A → B → C. Each system initialises from the
previous one's best checkpoint, so attribution of each improvement is clean.
"""

import logging
from pathlib import Path
from typing import Optional

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)


class EmotionEmbedding(nn.Module):
    """Learnable emotion embedding for VITS conditioning.

    Architecture choice (ADR-2): nn.Embedding(num_emotions, embedding_dim).
    Injection method: additive at text encoder output.
    """

    def __init__(
        self,
        num_emotions: int = 4,
        embedding_dim: int = 192,
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_emotions, embedding_dim)
        # Initialize with small values to minimize initial impact
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)
        self.embedding_dim = embedding_dim
        self.num_emotions = num_emotions

    def forward(self, emotion_ids: torch.Tensor) -> torch.Tensor:
        """Look up emotion embeddings.

        Args:
            emotion_ids: Integer tensor of shape (batch,).

        Returns:
            Embedding tensor of shape (batch, embedding_dim).
        """
        return self.embedding(emotion_ids)


class EmotionVITS(nn.Module):
    """EmotionVITS wrapper around Coqui VITS.

    This module wraps a pretrained/fine-tuned Coqui VITS model and injects
    emotion conditioning. For System C, it also adds prosody heads.

    Design notes:
    - We don't subclass Vits directly due to Coqui's complex init.
    - Instead, we wrap the model and intercept forward passes.
    - The emotion embedding is added to the text encoder output
      (latent z before duration predictor + flow).
    """

    def __init__(
        self,
        vits_model: nn.Module,
        use_emotion: bool = True,
        num_emotions: int = 4,
        embedding_dim: int = 192,
        injection_method: str = "add",
        use_prosody_heads: bool = False,
        prosody_heads: Optional[nn.Module] = None,
        prosody_loss_weight: float = 0.1,
    ):
        """Initialize EmotionVITS.

        Args:
            vits_model: A Coqui VITS model instance.
            use_emotion: Whether to use emotion conditioning (False for System A).
            num_emotions: Number of emotion categories.
            embedding_dim: Emotion embedding dimension (must match VITS hidden dim).
            injection_method: How to inject emotion. "add" = additive, "concat" = concatenation.
            use_prosody_heads: Whether to attach prosody prediction heads (System C).
            prosody_heads: Pre-built ProsodyHeads module.
            prosody_loss_weight: Lambda weighting for prosody auxiliary loss.
        """
        super().__init__()
        self.vits = vits_model
        self.use_emotion = use_emotion
        self.injection_method = injection_method
        self.use_prosody_heads = use_prosody_heads
        self.prosody_loss_weight = prosody_loss_weight

        if use_emotion:
            self.emotion_embedding = EmotionEmbedding(num_emotions, embedding_dim)
        else:
            self.emotion_embedding = None

        if use_prosody_heads and prosody_heads is not None:
            self.prosody_heads = prosody_heads
        else:
            self.prosody_heads = None

    def inject_emotion(
        self,
        hidden: torch.Tensor,
        emotion_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Inject emotion embedding into hidden states.

        Args:
            hidden: Text encoder output, shape (batch, hidden_dim, seq_len).
            emotion_ids: Emotion label indices, shape (batch,).

        Returns:
            Modified hidden states with same shape.
        """
        if not self.use_emotion or self.emotion_embedding is None:
            return hidden

        # Get emotion embedding: (batch, embedding_dim)
        emb = self.emotion_embedding(emotion_ids)

        if self.injection_method == "add":
            # Broadcast addition: (batch, hidden_dim) -> (batch, hidden_dim, 1)
            emb = emb.unsqueeze(-1)
            hidden = hidden + emb
        elif self.injection_method == "concat":
            # Concatenate along time dimension (not recommended, changes shape)
            emb = emb.unsqueeze(-1)  # (batch, hidden_dim, 1)
            hidden = torch.cat([hidden, emb], dim=-1)
        else:
            raise ValueError(f"Unknown injection method: {self.injection_method}")

        return hidden

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        y: torch.Tensor,
        y_lengths: torch.Tensor,
        emotion_ids: Optional[torch.Tensor] = None,
        prosody_targets: Optional[dict[str, torch.Tensor]] = None,
    ) -> dict:
        """Forward pass for training.

        This is a modified forward that injects emotion embeddings
        into the VITS pipeline. The exact injection point depends
        on the Coqui VITS implementation.

        Args:
            x: Text token IDs, shape (batch, text_len).
            x_lengths: Text lengths, shape (batch,).
            y: Mel/linear spec target, shape (batch, n_freq, mel_len).
            y_lengths: Spec lengths, shape (batch,).
            emotion_ids: Emotion labels, shape (batch,). None for System A.
            prosody_targets: Dict of prosody target tensors for System C.

        Returns:
            Dict with all losses and outputs.
        """
        # --- Step 1: Run text encoder ---
        # Access the VITS text encoder
        if hasattr(self.vits, "text_encoder"):
            text_encoder = self.vits.text_encoder
        elif hasattr(self.vits, "enc_p"):
            text_encoder = self.vits.enc_p
        else:
            raise AttributeError("Cannot find text encoder in VITS model")

        # Get text encoder output
        x_encoded, m_p, logs_p, x_mask = text_encoder(x, x_lengths)

        # --- Step 2: Inject emotion ---
        if emotion_ids is not None:
            x_encoded = self.inject_emotion(x_encoded, emotion_ids)

        # --- Step 3: Run rest of VITS pipeline ---
        # The standard VITS forward computes:
        # - Posterior encoder on y → z, m_q, logs_q
        # - Flow: z → z_p (for KL loss)
        # - Duration predictor on x_encoded
        # - Decoder on z → audio

        # Posterior encoder
        if hasattr(self.vits, "posterior_encoder"):
            z, m_q, logs_q, y_mask = self.vits.posterior_encoder(y, y_lengths)
        elif hasattr(self.vits, "enc_q"):
            z, m_q, logs_q, y_mask = self.vits.enc_q(y, y_lengths)
        else:
            raise AttributeError("Cannot find posterior encoder in VITS model")

        # Flow
        if hasattr(self.vits, "flow"):
            z_p = self.vits.flow(z, y_mask)
        else:
            z_p = z

        # --- Monotonic Alignment Search (standard VITS training) ---
        # Align text-length prior (m_p, logs_p) to spec-length posterior
        # so all KL-loss tensors share the same time dimension.
        with torch.no_grad():
            s_p_sq_r = torch.exp(-2 * logs_p)  # 1/sigma_p^2, (B, H, T_text)
            neg_cent1 = torch.sum(
                -0.5 * math.log(2 * math.pi) - logs_p, dim=1
            ).unsqueeze(-1)                                        # (B, T_text, 1)
            neg_cent2 = torch.matmul(
                -0.5 * s_p_sq_r.transpose(1, 2), z_p ** 2
            )                                                      # (B, T_text, T_spec)
            neg_cent3 = torch.matmul(
                (s_p_sq_r * m_p).transpose(1, 2), z_p
            )                                                      # (B, T_text, T_spec)
            neg_cent4 = torch.sum(
                -0.5 * (m_p ** 2) * s_p_sq_r, dim=1
            ).unsqueeze(-1)                                        # (B, T_text, 1)
            neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4

            attn_mask = x_mask.transpose(1, 2) * y_mask   # (B, T_text, T_spec)
            attn = self._maximum_path(neg_cent, attn_mask) # (B, T_text, T_spec)

        # --- Duration predictor loss from MAS alignment ---
        dp = getattr(self.vits, "dp", None) or getattr(self.vits, "duration_predictor", None)
        duration_loss = torch.tensor(0.0, device=x.device)
        if dp is not None:
            w = attn.sum(-1).unsqueeze(1)  # (B, 1, T_text) per-phone frames
            try:
                # Coqui StochasticDurationPredictor.forward(x, x_mask, dr, g, ...)
                # dr = ground-truth durations, reverse=False (training mode)
                dur_nll = dp(x_encoded, x_mask, dr=w, g=None, reverse=False)
                duration_loss = dur_nll / x_mask.sum()
            except (TypeError, RuntimeError, AssertionError):
                try:
                    logw_hat = dp(x_encoded, x_mask, g=None, reverse=True)
                    log_w_gt = torch.log(w.clamp(min=1e-6))
                    duration_loss = F.l1_loss(logw_hat * x_mask, log_w_gt * x_mask)
                except (TypeError, RuntimeError):
                    pass  # DP interface incompatible, skip duration loss

        # Expand prior params from text-length to spec-length
        # (B, H, T_text) @ (B, T_text, T_spec) -> (B, H, T_spec)
        m_p = torch.bmm(m_p, attn)
        logs_p = torch.bmm(logs_p, attn)

        # Decoder
        if hasattr(self.vits, "dec"):
            o_hat = self.vits.dec(z * y_mask)
        elif hasattr(self.vits, "waveform_decoder"):
            o_hat = self.vits.waveform_decoder(z * y_mask)
        else:
            raise AttributeError("Cannot find decoder in VITS model")

        outputs = {
            "model_outputs": o_hat,
            "z": z,
            "z_p": z_p,
            "m_p": m_p,
            "logs_p": logs_p,
            "m_q": m_q,
            "logs_q": logs_q,
            "x_encoded": x_encoded,
            "duration_loss": duration_loss,
        }

        # --- Step 4: Prosody heads (System C) ---
        if self.use_prosody_heads and self.prosody_heads is not None:
            # Pool encoder output across time for utterance-level prediction
            # Use mean pooling over valid (non-masked) positions
            x_mask_float = x_mask.float()  # (batch, 1, seq_len)
            pooled = (x_encoded * x_mask_float).sum(dim=-1) / x_mask_float.sum(dim=-1).clamp(min=1)
            # pooled: (batch, hidden_dim)

            prosody_preds = self.prosody_heads(pooled)
            outputs["prosody_preds"] = prosody_preds

            if prosody_targets is not None:
                prosody_loss = self.prosody_heads.compute_loss(
                    prosody_preds, prosody_targets
                )
                outputs["prosody_loss"] = prosody_loss * self.prosody_loss_weight

        return outputs

    def infer(
        self,
        x: torch.Tensor,
        x_lengths: Optional[torch.Tensor] = None,
        emotion_ids: Optional[torch.Tensor] = None,
        noise_scale: float = 0.667,
        length_scale: float = 1.0,
        noise_scale_w: float = 0.8,
    ) -> torch.Tensor:
        """Inference: generate audio from text.

        Args:
            x: Text token IDs, shape (1, text_len) or (batch, text_len).
            x_lengths: Text lengths. If None, computed from x.
            emotion_ids: Emotion label indices. None for neutral/System A.
            noise_scale: Noise scale for sampling.
            length_scale: Duration scale.
            noise_scale_w: Noise scale for duration predictor.

        Returns:
            Generated waveform tensor.
        """
        if x_lengths is None:
            x_lengths = torch.tensor([x.shape[-1]], dtype=torch.long, device=x.device)

        # Get text encoder output
        if hasattr(self.vits, "text_encoder"):
            x_encoded, m_p, logs_p, x_mask = self.vits.text_encoder(x, x_lengths)
        elif hasattr(self.vits, "enc_p"):
            x_encoded, m_p, logs_p, x_mask = self.vits.enc_p(x, x_lengths)
        else:
            raise AttributeError("Cannot find text encoder")

        # Inject emotion
        if emotion_ids is not None:
            x_encoded = self.inject_emotion(x_encoded, emotion_ids)

        # Duration predictor
        if hasattr(self.vits, "duration_predictor"):
            dp = self.vits.duration_predictor
        elif hasattr(self.vits, "dp"):
            dp = self.vits.dp
        else:
            raise AttributeError("Cannot find duration predictor")

        # Standard VITS inference pipeline
        # Coqui VITS may use a stochastic duration predictor that accepts
        # noise_scale_w via g, or a deterministic one.  We try the
        # stochastic variant first, then fall back.
        try:
            logw = dp(x_encoded, x_mask, g=None, reverse=True, noise_scale=noise_scale_w)
        except TypeError:
            try:
                logw = dp(x_encoded, x_mask, g=None)
            except TypeError:
                logw = dp(x_encoded, x_mask)
        w = torch.exp(logw) * x_mask * length_scale

        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = torch.unsqueeze(
            self._sequence_mask(y_lengths, None), 1
        ).to(x_mask.dtype)

        # Attention alignment (monotonic)
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = self._generate_path(w_ceil, attn_mask)

        # attn: (B, 1, T_text, T_spec) -> expand prior to spec-length
        attn_sq = attn.squeeze(1)  # (B, T_text, T_spec)
        m_p_expanded = torch.bmm(m_p, attn_sq)       # (B, H, T_spec)
        logs_p_expanded = torch.bmm(logs_p, attn_sq)  # (B, H, T_spec)

        # Sample from prior
        z_p = m_p_expanded + torch.randn_like(m_p_expanded) * torch.exp(logs_p_expanded) * noise_scale

        # Inverse flow
        if hasattr(self.vits, "flow"):
            z = self.vits.flow(z_p, y_mask, reverse=True)
        else:
            z = z_p

        # Decode
        if hasattr(self.vits, "dec"):
            o = self.vits.dec((z * y_mask)[:, :, :])
        elif hasattr(self.vits, "waveform_decoder"):
            o = self.vits.waveform_decoder((z * y_mask)[:, :, :])
        else:
            raise AttributeError("Cannot find decoder")

        return o

    @staticmethod
    def _sequence_mask(length, max_length=None):
        if max_length is None:
            max_length = length.max()
        x = torch.arange(max_length, dtype=length.dtype, device=length.device)
        return x.unsqueeze(0) < length.unsqueeze(1)

    @staticmethod
    def _maximum_path(value, mask):
        """Monotonic Alignment Search (MAS) via Viterbi DP.

        Finds the maximum-sum monotonic path through the log-likelihood
        matrix — standard VITS training alignment.

        Args:
            value: (B, T_text, T_spec) log-likelihood scores.
            mask:  (B, T_text, T_spec) valid-position mask (1=valid).

        Returns:
            path: (B, T_text, T_spec) hard 0/1 alignment.
        """
        device = value.device
        dtype = value.dtype
        value_np = value.detach().cpu().numpy().astype(np.float64)
        mask_np = mask.detach().cpu().numpy().astype(np.bool_)
        B, T_x, T_y = value_np.shape
        path = np.zeros_like(value_np, dtype=np.float32)

        for b in range(B):
            t_x = int(mask_np[b, :, 0].sum())
            t_y = int(mask_np[b, 0, :].sum())
            if t_x < 1 or t_y < 1:
                continue

            v = np.full((t_x, t_y), -1e9, dtype=np.float64)
            v[0, 0] = value_np[b, 0, 0]

            # Forward DP
            for j in range(1, t_y):
                for i in range(min(t_x, j + 1)):
                    prev = v[i, j - 1]
                    if i > 0:
                        prev = max(prev, v[i - 1, j - 1])
                    v[i, j] = value_np[b, i, j] + prev

            # Backtrace
            i = t_x - 1
            for j in range(t_y - 1, -1, -1):
                path[b, i, j] = 1.0
                if j > 0 and i > 0 and v[i - 1, j - 1] > v[i, j - 1]:
                    i -= 1

        return torch.from_numpy(path).to(device=device, dtype=dtype)

    @staticmethod
    def _generate_path(duration, mask):
        """Generate alignment path from durations."""
        b, _, t_y, t_x = mask.shape
        cum_dur = torch.cumsum(duration, -1)
        cum_dur_flat = cum_dur.view(b * t_x)

        path = torch.zeros(b, t_x, t_y, dtype=mask.dtype, device=mask.device)

        for i in range(b):
            for j in range(t_x):
                start = int(cum_dur[i, 0, j].item()) - int(duration[i, 0, j].item())
                end = int(cum_dur[i, 0, j].item())
                if start < t_y and end > 0:
                    path[i, j, max(0, start):min(t_y, end)] = 1.0

        return path.unsqueeze(1)  # (b, 1, t_x, t_y)

    def freeze_for_system_a(self):
        """Freeze parameters for System A training (ADR-3).

        Freeze: text_encoder, posterior_encoder, decoder
        Unfreeze: duration_predictor, flow
        """
        # Freeze everything first
        for param in self.vits.parameters():
            param.requires_grad = False

        # Unfreeze duration predictor
        dp = getattr(self.vits, "duration_predictor", None) or getattr(self.vits, "dp", None)
        if dp:
            for param in dp.parameters():
                param.requires_grad = True

        # Unfreeze flow
        flow = getattr(self.vits, "flow", None)
        if flow:
            for param in flow.parameters():
                param.requires_grad = True

        logger.info("Froze for System A: text_encoder, posterior_encoder, decoder frozen; "
                     "duration_predictor, flow unfrozen")

    def freeze_for_system_b(self):
        """Freeze parameters for System B training (ADR-3).

        Same as System A freeze, PLUS emotion embedding is unfrozen.
        """
        self.freeze_for_system_a()

        # Unfreeze emotion embedding
        if self.emotion_embedding is not None:
            for param in self.emotion_embedding.parameters():
                param.requires_grad = True
            logger.info("Emotion embedding unfrozen for System B")

    def freeze_for_system_c(self):
        """Freeze parameters for System C training (ADR-3).

        Same as System B freeze, PLUS prosody heads are unfrozen.
        """
        self.freeze_for_system_b()

        # Unfreeze prosody heads
        if self.prosody_heads is not None:
            for param in self.prosody_heads.parameters():
                param.requires_grad = True
            logger.info("Prosody heads unfrozen for System C")

    def count_parameters(self) -> dict:
        """Count trainable vs frozen parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
            "trainable_pct": 100.0 * trainable / max(total, 1),
        }


def load_pretrained_vits(
    model_name: str = "tts_models/en/ljspeech/vits",
    use_cuda: bool = False,
) -> nn.Module:
    """Load a pretrained Coqui VITS model.

    Args:
        model_name: Coqui model identifier.
        use_cuda: Whether to load on GPU.

    Returns:
        The raw VITS nn.Module.
    """
    from TTS.api import TTS

    tts = TTS(model_name=model_name, gpu=use_cuda)
    vits_model = tts.synthesizer.tts_model
    return vits_model


def build_emotion_vits(
    system: str = "B",
    checkpoint_path: Optional[str] = None,
    use_cuda: bool = False,
    num_emotions: int = 4,
    embedding_dim: int = 192,
    prosody_heads: Optional[nn.Module] = None,
    prosody_loss_weight: float = 0.1,
) -> EmotionVITS:
    """Factory function to build EmotionVITS for a given system.

    Args:
        system: "A", "B", or "C".
        checkpoint_path: Path to checkpoint to init from.
            - System A: pretrained LJSpeech (or None to download)
            - System B: System A checkpoint path
            - System C: System B checkpoint path
        use_cuda: GPU flag.
        num_emotions: Number of emotion classes.
        embedding_dim: Emotion embedding dimension.
        prosody_heads: ProsodyHeads module (required for System C).
        prosody_loss_weight: Lambda for prosody loss (System C only).

    Returns:
        EmotionVITS instance with appropriate config.
    """
    system = system.upper()

    # Load base VITS model
    if checkpoint_path and Path(checkpoint_path).exists():
        # Load from a saved EmotionVITS checkpoint
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "vits_state_dict" in state:
            vits_model = load_pretrained_vits(use_cuda=use_cuda)
            vits_model.load_state_dict(state["vits_state_dict"], strict=False)
        else:
            vits_model = load_pretrained_vits(use_cuda=use_cuda)
    else:
        vits_model = load_pretrained_vits(use_cuda=use_cuda)

    # Build EmotionVITS
    model = EmotionVITS(
        vits_model=vits_model,
        use_emotion=(system in ("B", "C")),
        num_emotions=num_emotions,
        embedding_dim=embedding_dim,
        injection_method="add",
        use_prosody_heads=(system == "C"),
        prosody_heads=prosody_heads if system == "C" else None,
        prosody_loss_weight=prosody_loss_weight,
    )

    # Load full EmotionVITS state if available
    if checkpoint_path and Path(checkpoint_path).exists():
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"], strict=False)
            logger.info(f"Loaded EmotionVITS state from {checkpoint_path}")

    # Apply freeze strategy
    if system == "A":
        model.freeze_for_system_a()
    elif system == "B":
        model.freeze_for_system_b()
    elif system == "C":
        model.freeze_for_system_c()

    if use_cuda:
        model = model.cuda()

    params = model.count_parameters()
    logger.info(f"System {system}: {params['trainable']:,} trainable / {params['total']:,} total "
                f"({params['trainable_pct']:.1f}%)")

    return model
