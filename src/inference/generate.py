"""Autoregressive text generation using a trained TransformerLM.

Processes prompt tokens sequentially to fill KV caches, then generates
new tokens one at a time with configurable sampling parameters.
"""

import torch
from torch import Tensor

from src.inference.sampling import sample
from src.model.lm import TransformerLM
from src.tokenizer.bpe import Tokenizer
from src.transformer.config import M01Config
from src.transformer.kv_cache import KVCache


def generate(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_gen_len: int = 100,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    device: torch.device | None = None,
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

    Returns:
        Generated text (prompt + continuation).

    Raises:
        ValueError: If prompt is empty or max_gen_len < 1.
    """
    if not prompt.strip():
        raise ValueError("Prompt must be non-empty")
    if max_gen_len < 1:
        raise ValueError(f"max_gen_len must be >= 1, got {max_gen_len}")

    if device is None:
        device = next(model.parameters()).device

    prompt_ids = tokenizer.encode(prompt)
    generated: list[int] = list(prompt_ids)

    # Build KV caches: one per transformer layer
    # KVCache shape: (1, max_seq_len, n_heads, d_head)
    kv_caches = [
        KVCache(model.config.context_length, model.config.n_heads, model.config.d_head, device)
        for _ in range(model.config.n_layers)
    ]

    eos_id = tokenizer.special_tokens.get("<|endoftext|>", 256)

    with torch.no_grad():
        # Single continuous autoregressive loop:
        #   prompt tokens → fill cache without sampling
        #   generated tokens → sample next token each step
        total_steps = len(prompt_ids) + max_gen_len

        for step in range(total_steps):
            # Current token: from prompt if still in prompt region,
            # otherwise the last generated token
            if step < len(prompt_ids):
                current_id = prompt_ids[step]
            else:
                current_id = generated[-1]

            x = torch.tensor([[current_id]], device=device)
            logits: Tensor = model(x, kv_caches)  # (1, 1, V)

            # Still processing prompt — just fill cache, don't sample
            if step < len(prompt_ids) - 1:
                continue

            # Last prompt token: sample the FIRST new token
            # Generated tokens: sample the NEXT token
            next_logits = logits[0, -1, :]  # (vocab_size,)
            next_id = sample(next_logits, temperature, top_k, top_p)

            if next_id == eos_id:
                break

            generated.append(next_id)

    return tokenizer.decode(generated)
