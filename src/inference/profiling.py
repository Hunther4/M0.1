from dataclasses import asdict

import torch

from src.inference.generate import generate
from src.inference.generate import GenerationMetrics
from src.inference.prompt_cache import PromptPrefixCache
from src.tokenizer.bpe import Tokenizer
from src.model.lm import TransformerLM


def profile_inference_detailed(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_gen_len: int = 10,
    prompt_cache: PromptPrefixCache | None = None,
) -> dict[str, float | int]:
    metrics = GenerationMetrics()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        start_mem = torch.cuda.max_memory_allocated()
    else:
        start_mem = 0

    generate(
        model,
        tokenizer,
        prompt,
        max_gen_len=max_gen_len,
        prompt_cache=prompt_cache,
        metrics=metrics,
    )

    if torch.cuda.is_available():
        end_mem = torch.cuda.max_memory_allocated()
    else:
        end_mem = 0

    result: dict[str, float | int] = asdict(metrics)
    result["total_seconds"] = metrics.total_seconds
    result["tokens_per_second"] = metrics.tokens_per_second
    result["peak_memory_bytes"] = int(end_mem - start_mem)
    return result


def profile_inference(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_gen_len: int = 10,
) -> tuple[float, float]:
    """Backward-compatible compact profiling result."""
    result = profile_inference_detailed(model, tokenizer, prompt, max_gen_len)
    return float(result["tokens_per_second"]), float(result["peak_memory_bytes"])
