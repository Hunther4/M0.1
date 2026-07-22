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
from src.transformer.kv_cache import build_attention_cache


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

    with torch.inference_mode():
        # 1. Parallel prefill phase
        # Process all prompt tokens except the last one in one forward pass to populate the caches.
        if len(prompt_ids) > 1:
            prefill_x = torch.tensor([prompt_ids[:-1]], dtype=torch.long, device=device)
            model(prefill_x, kv_caches)
            
        # 2. Autoregressive decoding phase
        current_id = prompt_ids[-1]
        x = torch.zeros((1, 1), dtype=torch.long, device=device)
        
        for _ in range(max_gen_len):
            x[0, 0] = current_id
            logits: Tensor = model(x, kv_caches)  # (1, 1, V)
            
            next_logits = logits[0, -1, :]  # (vocab_size,)
            next_id = sample(next_logits, temperature, top_k, top_p)
            
            if next_id == eos_id:
                break
                
            generated.append(next_id)
            current_id = next_id

    return tokenizer.decode(generated)
