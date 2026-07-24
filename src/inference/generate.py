"""Autoregressive text generation using a trained TransformerLM.

Processes prompt tokens sequentially to fill KV caches, then generates
new tokens one at a time with configurable sampling parameters.
"""

from dataclasses import dataclass
import time

import torch
from torch import Tensor

from src.inference.prompt_cache import PromptPrefixCache
from src.inference.sampling import sample
from src.model.lm import TransformerLM
from src.tokenizer.bpe import Tokenizer
from src.transformer.config import M01Config
from src.transformer.kv_cache import build_attention_cache


@dataclass
class GenerationMetrics:
    """Timing and cache telemetry populated by :func:`generate`."""

    prompt_tokens: int = 0
    generated_tokens: int = 0
    reused_prompt_tokens: int = 0
    prefill_seconds: float = 0.0
    decode_seconds: float = 0.0

    @property
    def total_seconds(self) -> float:
        return self.prefill_seconds + self.decode_seconds

    @property
    def tokens_per_second(self) -> float:
        return self.generated_tokens / self.total_seconds if self.total_seconds else 0.0


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def generate(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_gen_len: int = 100,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    device: torch.device | None = None,
    prompt_cache: PromptPrefixCache | None = None,
    metrics: GenerationMetrics | None = None,
) -> str:
    """Generate text from a prompt using autoregressive decoding.

    Fills KV caches by processing each prompt token sequentially, then
    generates ``max_gen_len`` new tokens with the specified sampling strategy.

    Args:
        model: Trained TransformerLM in eval mode (caller must set).
        tokenizer: BPE tokenizer for encode/decode.
        prompt: Input text to condition generation on.
        max_gen_len: Maximum number of new tokens to generate.
        temperature: Sampling temperature (0 = greedy).
        top_k: Top-k filtering threshold.
        top_p: Nucleus sampling threshold.
        device: Device for tensors. Auto-detected from model device if None.
        prompt_cache: Optional bounded cache for reuse across prompts.
        metrics: Optional object populated with timing and cache telemetry.

    Returns:
        Generated text (prompt + continuation).

    Raises:
        ValueError: If prompt is empty, max_gen_len < 1, or context
            length would be exceeded.
    """
    if not prompt.strip():
        raise ValueError("Prompt must be non-empty")
    if max_gen_len < 1:
        raise ValueError(f"max_gen_len must be >= 1, got {max_gen_len}")

    if device is None:
        device = next(model.parameters()).device

    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        raise ValueError("Prompt produced no tokens")
    invalid_ids = [token_id for token_id in prompt_ids if not 0 <= token_id < model.config.vocab_size]
    if invalid_ids:
        raise ValueError(
            "Tokenizer is incompatible with the model vocabulary: "
            f"token id {invalid_ids[0]} is outside [0, {model.config.vocab_size})"
        )

    # Validate that prompt + generation fits within the model's context window
    context_length = model.config.context_length
    if len(prompt_ids) > context_length:
        raise ValueError(
            f"Prompt length ({len(prompt_ids)} tokens) exceeds model context "
            f"length ({context_length}). Shorten the prompt."
        )
    if len(prompt_ids) + max_gen_len > context_length:
        raise ValueError(
            f"Prompt ({len(prompt_ids)} tokens) + max_gen_len ({max_gen_len}) "
            f"exceeds context_length ({context_length})"
        )

    generated: list[int] = list(prompt_ids)

    # Build KV caches: one per transformer layer
    model_dtype = next(model.parameters()).dtype
    kv_caches = [
        build_attention_cache(model.config, device, model_dtype)
        for _ in range(model.config.n_layers)
    ]

    eos_id = tokenizer.special_tokens.get("<|endoftext|>", 256)
    cacheable_prompt = prompt_ids[:-1]
    reused_tokens = 0
    if prompt_cache is not None:
        reused_tokens = prompt_cache.restore_longest_prefix(
            model, cacheable_prompt, kv_caches
        )

    if metrics is not None:
        metrics.prompt_tokens = len(prompt_ids)
        metrics.generated_tokens = 0
        metrics.reused_prompt_tokens = reused_tokens
        metrics.prefill_seconds = 0.0
        metrics.decode_seconds = 0.0

    with torch.inference_mode():
        # 1. Parallel prefill phase
        # Only process the suffix not restored by the prompt-prefix cache.
        _synchronize(device)
        prefill_start = time.perf_counter()
        if reused_tokens < len(cacheable_prompt):
            prefill_x = torch.tensor(
                [cacheable_prompt[reused_tokens:]], dtype=torch.long, device=device
            )
            model(prefill_x, kv_caches)
        _synchronize(device)
        prefill_seconds = time.perf_counter() - prefill_start

        # Store immutable prefill state before decoding mutates the working caches.
        if prompt_cache is not None:
            prompt_cache.store(model, cacheable_prompt, kv_caches)

        # 2. Autoregressive decoding phase
        current_id = prompt_ids[-1]
        x = torch.zeros((1, 1), dtype=torch.long, device=device)

        _synchronize(device)
        decode_start = time.perf_counter()
        generated_tokens = 0
        for _ in range(max_gen_len):
            x[0, 0] = current_id
            logits: Tensor = model(x, kv_caches)  # (1, 1, V)

            next_logits = logits[0, -1, :]  # (vocab_size,)
            next_id = sample(next_logits, temperature, top_k, top_p)
            generated_tokens += 1
            if next_id == eos_id:
                break

            generated.append(next_id)
            current_id = next_id
        _synchronize(device)
        decode_seconds = time.perf_counter() - decode_start

    if metrics is not None:
        metrics.generated_tokens = generated_tokens
        metrics.prefill_seconds = prefill_seconds
        metrics.decode_seconds = decode_seconds

    return tokenizer.decode(generated)
