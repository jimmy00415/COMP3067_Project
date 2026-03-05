"""System A0/A baseline wrapper (FR-2, ADR-1).

System A0: Raw pretrained VITS from LJSpeech (reference baseline).
            Never fine-tuned. Exists to isolate domain adaptation effect.

System A:  Same architecture, fine-tuned on EmoV-DB WITHOUT emotion labels.
            Proves domain adaptation effect vs A0.

Both use the standard Coqui TTS Synthesizer for inference.
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class BaselineSynthesizer:
    """Wrapper around Coqui TTS for System A0/A inference.

    For training System A, we use the standard Coqui training pipeline
    with selective parameter freezing (freeze text_encoder, posterior_encoder,
    decoder; unfreeze duration_predictor, flow). See configs/train_a.yaml.

    This class handles inference only.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        config_path: Optional[str] = None,
        use_pretrained: bool = True,
        pretrained_name: str = "tts_models/en/ljspeech/vits",
        use_cuda: bool = False,
    ):
        """Initialize synthesizer.

        Args:
            model_path: Path to fine-tuned checkpoint (.pth). If None + use_pretrained,
                       loads the standard pretrained model (A0 mode).
            config_path: Path to model config JSON. Required if model_path set.
            use_pretrained: If True and model_path is None, download pretrained.
            pretrained_name: Coqui model name for pretrained download.
            use_cuda: Whether to use GPU.
        """
        self.model_path = model_path
        self.config_path = config_path
        self.use_pretrained = use_pretrained
        self.pretrained_name = pretrained_name
        self.use_cuda = use_cuda
        self._synthesizer = None

    def load(self):
        """Lazily load the Coqui Synthesizer."""
        if self._synthesizer is not None:
            return

        from TTS.utils.synthesizer import Synthesizer

        if self.model_path and Path(self.model_path).exists():
            logger.info(f"Loading fine-tuned model from {self.model_path}")
            self._synthesizer = Synthesizer(
                tts_checkpoint=self.model_path,
                tts_config_path=self.config_path,
                use_cuda=self.use_cuda,
            )
        elif self.use_pretrained:
            logger.info(f"Loading pretrained model: {self.pretrained_name}")
            from TTS.api import TTS
            tts = TTS(model_name=self.pretrained_name, gpu=self.use_cuda)
            self._synthesizer = tts.synthesizer
        else:
            raise ValueError(
                "Must provide model_path or set use_pretrained=True"
            )

    @property
    def synthesizer(self):
        """Access the underlying Coqui Synthesizer (lazy init)."""
        if self._synthesizer is None:
            self.load()
        return self._synthesizer

    def synthesize(self, text: str) -> tuple:
        """Generate speech from text.

        Args:
            text: Input text string.

        Returns:
            Tuple of (audio_array, sample_rate).
        """
        self.load()
        wav = self._synthesizer.tts(text)
        sr = self._synthesizer.output_sample_rate
        return wav, sr

    def synthesize_to_file(self, text: str, output_path: str) -> str:
        """Generate speech and save to WAV file.

        Args:
            text: Input text.
            output_path: Output WAV file path.

        Returns:
            Path to saved file.
        """
        self.load()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        wav = self._synthesizer.tts(text)
        self._synthesizer.save_wav(wav, output_path)
        return output_path

    def get_config(self) -> dict:
        """Return model configuration."""
        self.load()
        return self._synthesizer.tts_config if hasattr(self._synthesizer, "tts_config") else {}


def create_system_a0() -> BaselineSynthesizer:
    """Create System A0 (pretrained LJSpeech, no fine-tuning)."""
    return BaselineSynthesizer(use_pretrained=True, pretrained_name="tts_models/en/ljspeech/vits")


def create_system_a(checkpoint_path: str, config_path: str, use_cuda: bool = False) -> BaselineSynthesizer:
    """Create System A (fine-tuned on EmoV-DB, no emotion labels).

    Args:
        checkpoint_path: Path to System A checkpoint.
        config_path: Path to model config.
        use_cuda: Whether to use GPU.
    """
    return BaselineSynthesizer(
        model_path=checkpoint_path,
        config_path=config_path,
        use_pretrained=False,
        use_cuda=use_cuda,
    )
