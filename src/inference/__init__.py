"""M0.1 Inference — autoregressive text generation."""

from src.inference.generate import generate
from src.inference.sampling import sample

__all__ = ["generate", "sample"]
from src.inference.generate import GenerationMetrics, generate
from src.inference.prompt_cache import PromptCacheStats, PromptPrefixCache

__all__ = [
    "GenerationMetrics",
    "PromptCacheStats",
    "PromptPrefixCache",
    "generate",
]
