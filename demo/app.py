"""Gradio demo app for Emotive TTS (FR-4).

Interactive UI: text input + emotion dropdown → synthesized audio.
Supports Systems A0, A, B, C with side-by-side comparison.

Usage:
    python demo/app.py
    # Opens http://localhost:7860
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Lazy imports for faster startup
_MODELS = {}


def _load_model(system: str, checkpoint_path: Optional[str] = None, use_cuda: bool = False):
    """Lazily load and cache a model."""
    if system in _MODELS:
        return _MODELS[system]

    if system == "A0":
        from src.models.baseline import create_system_a0
        model = create_system_a0()
        model.use_cuda = use_cuda
        model.load()
        _MODELS[system] = model

    elif system == "A":
        from src.models.baseline import create_system_a
        if checkpoint_path and Path(checkpoint_path).exists():
            model = create_system_a(checkpoint_path, "", use_cuda=use_cuda)
            model.load()
            _MODELS[system] = model
        else:
            logger.warning(f"System A checkpoint not found: {checkpoint_path}")
            return None

    elif system in ("B", "C"):
        from src.models.emotion_vits import build_emotion_vits
        from src.models.prosody_heads import build_prosody_heads

        prosody_heads = build_prosody_heads() if system == "C" else None
        model = build_emotion_vits(
            system=system,
            checkpoint_path=checkpoint_path,
            use_cuda=use_cuda,
            prosody_heads=prosody_heads,
        )
        model.eval()
        _MODELS[system] = model

    return _MODELS.get(system)


def synthesize(
    text: str,
    emotion: str,
    system: str = "B",
    checkpoint_path: Optional[str] = None,
    intensity: float = 1.0,
) -> tuple[int, np.ndarray]:
    """Synthesize speech with emotion.

    Args:
        text: Input text.
        emotion: Target emotion (neutral/angry/amused/disgust).
        system: System variant (A0/A/B/C).
        checkpoint_path: Path to model checkpoint.
        intensity: Emotion intensity scalar (0.0–2.0). Values >1 amplify
            the emotion embedding; <1 dampen it.  Only affects B/C.

    Returns:
        Tuple of (sample_rate, audio_array).
    """
    import torch
    from src.data.utils import EMOTION_MAP

    sr = 22050

    if system == "A0":
        model = _load_model("A0")
        if model is None:
            return sr, np.zeros(sr)
        wav, sr = model.synthesize(text)
        return sr, np.array(wav)

    elif system == "A":
        model = _load_model("A", checkpoint_path)
        if model is None:
            return sr, np.zeros(sr)
        wav, sr = model.synthesize(text)
        return sr, np.array(wav)

    elif system in ("B", "C"):
        model = _load_model(system, checkpoint_path)
        if model is None:
            return sr, np.zeros(sr)

        emotion_id = EMOTION_MAP.get(emotion, 0)
        device = next(model.parameters()).device

        # Tokenize
        if hasattr(model.vits, 'tokenizer'):
            tokens = model.vits.tokenizer.text_to_ids(text)
        else:
            tokens = [ord(c) for c in text.lower()]

        x = torch.LongTensor([tokens]).to(device)
        eid = torch.LongTensor([emotion_id]).to(device)

        # Apply intensity scaling to emotion embedding temporarily
        if intensity != 1.0 and model.emotion_embedding is not None:
            original_weight = model.emotion_embedding.embedding.weight.data.clone()
            model.emotion_embedding.embedding.weight.data *= intensity

        with torch.no_grad():
            wav_tensor = model.infer(x, emotion_ids=eid)

        # Restore original weights
        if intensity != 1.0 and model.emotion_embedding is not None:
            model.emotion_embedding.embedding.weight.data = original_weight

        wav = wav_tensor.squeeze().cpu().numpy()
        return sr, wav

    return sr, np.zeros(sr)


def build_demo(
    checkpoint_dir: str = "checkpoints",
    default_system: str = "B",
) -> "gr.Blocks":
    """Build Gradio demo interface.

    Args:
        checkpoint_dir: Directory containing system checkpoints.
        default_system: Default system to select.

    Returns:
        Gradio Blocks app.
    """
    import gradio as gr

    emotions = ["neutral", "angry", "amused", "disgust"]
    systems = ["A0", "A", "B", "C"]

    example_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "I cannot believe what just happened!",
        "Please pass the salt and pepper.",
        "That was the funniest thing I have ever seen.",
    ]

    def generate(text, emotion, system, intensity):
        ckpt = f"{checkpoint_dir}/system_{system.lower()}/best.pth"
        sr, audio = synthesize(text, emotion, system, ckpt, intensity=intensity)
        return (sr, audio)

    def compare_systems(text, emotion, intensity):
        """Generate audio from all systems for comparison."""
        results = []
        for sys in systems:
            ckpt = f"{checkpoint_dir}/system_{sys.lower()}/best.pth"
            try:
                sr, audio = synthesize(text, emotion, sys, ckpt, intensity=intensity)
                results.append((sr, audio))
            except Exception as e:
                logger.error(f"System {sys} failed: {e}")
                results.append((22050, np.zeros(22050)))
        return tuple(results)

    with gr.Blocks(title="Emotive TTS Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🎭 Emotive TTS Demo
            ### Exploring emotion in neural text-to-speech via VITS fine-tuning

            This demo lets you compare four system variants:
            - **A0**: Pretrained LJSpeech VITS (reference baseline)
            - **A**: Domain-adapted on EmoV-DB (no emotion labels)
            - **B**: A + emotion embedding conditioning
            - **C**: B + prosody auxiliary supervision
            """
        )

        with gr.Tab("Single System"):
            with gr.Row():
                with gr.Column():
                    text_input = gr.Textbox(
                        label="Input Text",
                        value=example_texts[0],
                        lines=2,
                    )
                    emotion_input = gr.Dropdown(
                        choices=emotions,
                        value="neutral",
                        label="Target Emotion",
                    )
                    system_input = gr.Dropdown(
                        choices=systems,
                        value=default_system,
                        label="System",
                    )
                    intensity_input = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                        label="Emotion Intensity (stretch goal S8.4)",
                        info="Scale emotion embedding. 0=neutral, 1=normal, 2=exaggerated. Only affects B/C.",
                    )
                    generate_btn = gr.Button("Generate", variant="primary")

                with gr.Column():
                    audio_output = gr.Audio(label="Generated Speech", type="numpy")

            generate_btn.click(
                fn=generate,
                inputs=[text_input, emotion_input, system_input, intensity_input],
                outputs=audio_output,
            )

        with gr.Tab("System Comparison"):
            with gr.Row():
                text_compare = gr.Textbox(
                    label="Input Text",
                    value=example_texts[0],
                    lines=2,
                )
                emotion_compare = gr.Dropdown(
                    choices=emotions,
                    value="angry",
                    label="Target Emotion",
                )
                intensity_compare = gr.Slider(
                    minimum=0.0, maximum=2.0, value=1.0, step=0.1,
                    label="Emotion Intensity",
                )
            compare_btn = gr.Button("Compare All Systems", variant="primary")

            with gr.Row():
                audio_a0 = gr.Audio(label="System A0 (pretrained)")
                audio_a = gr.Audio(label="System A (domain-adapted)")
                audio_b = gr.Audio(label="System B (+ emotion)")
                audio_c = gr.Audio(label="System C (+ prosody)")

            compare_btn.click(
                fn=compare_systems,
                inputs=[text_compare, emotion_compare, intensity_compare],
                outputs=[audio_a0, audio_a, audio_b, audio_c],
            )

        gr.Markdown(
            """
            ---
            *COMP3065 Final Year Project — Emotion-Conditioned VITS*

            **Note:** Systems A0 and A generate the same output regardless of
            emotion selection (they have no emotion conditioning). This is by design
            — it demonstrates the causal attribution of emotion embedding (A→B) and
            prosody supervision (B→C).
            """
        )

    return demo


def main():
    """Launch the Gradio demo."""
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Launch Emotive TTS demo")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--system", type=str, default="B")
    parser.add_argument("--share", action="store_true", help="Create public link")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    demo = build_demo(args.checkpoint_dir, args.system)
    demo.launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
