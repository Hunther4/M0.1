"""Sampling strategies for autoregressive text generation.

Provides temperature scaling, top-k filtering, and top-p (nucleus) sampling
for selecting the next token from model logits.
"""

import torch
from torch import Tensor


def sample(
    logits: Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> int:
    """Sample a token index from logits with optional temperature and filtering.

    Args:
        logits: Raw logits of shape (vocab_size,).
        temperature: Scaling factor. Values < 1 sharpen, > 1 flatten.
            0 means argmax (greedy).
        top_k: Keep only top-k highest logits before sampling.
        top_p: Keep smallest set of tokens whose cumulative probability
            exceeds top_p (nucleus sampling). Applied after top-k if both set.

    Returns:
        Sampled token ID as a Python int.

    Raises:
        ValueError: If temperature < 0.
    """
    if temperature < 0:
        raise ValueError(f"Temperature must be >= 0, got {temperature}")

    if temperature == 0:
        return int(logits.argmax(dim=-1).item())

    scaled = logits / temperature

    # Top-k: zero out everything below top k
    if top_k is not None and top_k > 0:
        k = min(top_k, scaled.size(-1))
        threshold = torch.topk(scaled, k).values[..., -1, None]
        scaled = scaled.where(scaled >= threshold, float("-inf"))

    probs = torch.softmax(scaled, dim=-1)

    # Top-p: keep smallest set of tokens with cumulative prob > top_p
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        mask = cumulative > top_p
        mask[..., 1:] = mask[..., :-1].clone()  # keep at least one token
        mask[..., 0] = False
        sorted_probs[mask] = 0.0
        probs = sorted_probs.scatter_(dim=-1, index=sorted_indices, src=sorted_probs)

    # Multinomial sampling
    return int(torch.multinomial(probs, 1).item())
