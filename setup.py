"""Setup script for Emotionally Controlled TTS project."""
from setuptools import setup, find_packages

setup(
    name="emotive-tts",
    version="0.1.0",
    description="Compute-Efficient Emotionally Controlled Text-to-Speech",
    packages=find_packages(),
    python_requires=">=3.10,<3.13",
    install_requires=[
        "TTS==0.22.0",
        "torch>=2.0",
        "torchaudio>=2.0",
        "hydra-core>=1.3",
        "omegaconf>=2.3",
        "mlflow>=2.10",
        "librosa>=0.10",
        "soundfile>=0.12",
        "numpy>=1.24,<2.0",
        "pandas>=2.0",
        "scipy>=1.11",
        "matplotlib>=3.7",
        "seaborn>=0.13",
    ],
    extras_require={
        "dev": ["pytest>=8.0", "ruff>=0.4", "pre-commit"],
        "demo": ["gradio>=4.0"],
        "eval": ["speechbrain>=1.0"],
    },
)
