"""M0.1 Inference — autoregressive text generation."""

from src.inference.generate import generate
from src.inference.sampling import sample

__all__ = ["generate", "sample"]
