"""Qualitative coherence and needle-in-a-haystack benchmarks."""

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module


def _model_device(model: Module) -> torch.device:
    try:
        parameter = next(model.parameters())
        if isinstance(parameter, Tensor):
            return parameter.device
    except (StopIteration, TypeError, AttributeError):
        pass
    return torch.device("cpu")


def coherence_test(
    model: Module,
    prompt: str,
    tokenizer,
    interval: int = 128,
    max_length: int = 512,
) -> dict:
    """Report token perplexity by interval while preserving the full prefix."""
    if interval < 1:
        raise ValueError("interval must be positive")
    if max_length < 2:
        raise ValueError("max_length must be at least 2")

    model.eval()
    encoded = tokenizer.encode(prompt)[:max_length]
    input_ids = torch.tensor(encoded, device=_model_device(model)).unsqueeze(0)
    interval_perplexities: list[float] = []

    # A single causal forward gives every token its real preceding context.
    # Reporting intervals only aggregate losses; they do not detach history.
    with torch.no_grad():
        if input_ids.size(1) > 1:
            logits = model(input_ids)
            prediction_count = min(logits.size(1) - 1, input_ids.size(1) - 1)
            if prediction_count > 0:
                token_losses = F.cross_entropy(
                    logits[:, :prediction_count].reshape(-1, logits.size(-1)),
                    input_ids[:, 1 : prediction_count + 1].reshape(-1),
                    reduction="none",
                )
                for start in range(0, prediction_count, interval):
                    interval_loss = token_losses[start : start + interval].mean()
                    interval_perplexities.append(torch.exp(interval_loss).item())

    average = (
        sum(interval_perplexities) / len(interval_perplexities)
        if interval_perplexities
        else float("inf")
    )
    return {
        "interval_perplexities": interval_perplexities,
        "average_coherence": average,
        "interval": interval,
        "evaluated_tokens": max(0, min(len(encoded) - 1, input_ids.size(1) - 1)),
    }


def niah_test(
    model: Module,
    haystack_text: str,
    needle: str,
    tokenizer,
    context_length: int = 512,
    depth: float = 0.5,
    query: str = "What was the hidden fact? Answer exactly: ",
) -> dict:
    """Measure teacher-forced retrieval of a fact inserted at a chosen depth.

    The needle is inserted once in the filler. A retrieval query is appended
    after the complete context and the model is scored on reproducing the
    needle as its answer, so local continuation of the inserted phrase cannot
    satisfy the benchmark.
    """
    if not 0.1 <= depth <= 0.9:
        raise ValueError("depth must be in the interval [0.1, 0.9]")
    if context_length < 1:
        raise ValueError("context_length must be positive")

    model.eval()
    device = _model_device(model)
    filler_tokens = tokenizer.encode(haystack_text)
    needle_tokens = tokenizer.encode(needle)
    query_tokens = tokenizer.encode(query)
    if not needle_tokens:
        raise ValueError("needle must encode to at least one token")

    available = context_length - 2 * len(needle_tokens) - len(query_tokens)
    if available < 1:
        query_tokens = tokenizer.encode("?")
        available = context_length - 2 * len(needle_tokens) - len(query_tokens)
    if available < 1:
        return {
            "needle": needle,
            "avg_probability": 0.0,
            "accuracy": 0.0,
            "context_length": context_length,
            "error": "context_too_short",
        }

    filler = filler_tokens[:available]
    insert_at = min(len(filler), max(0, round(len(filler) * depth)))
    context_ids = filler[:insert_at] + needle_tokens + filler[insert_at:]
    answer_start = len(context_ids) + len(query_tokens)
    full_ids = context_ids + query_tokens + needle_tokens
    input_ids = torch.tensor([full_ids], device=device)

    with torch.no_grad():
        probabilities = torch.softmax(model(input_ids), dim=-1)
        needle_probabilities: list[float] = []
        correct = 0
        for offset, token in enumerate(needle_tokens):
            prediction_position = answer_start + offset - 1
            if 0 <= prediction_position < probabilities.size(1):
                distribution = probabilities[0, prediction_position]
                needle_probabilities.append(distribution[token].item())
                correct += int(distribution.argmax().item() == token)

    average_probability = (
        sum(needle_probabilities) / len(needle_probabilities)
        if needle_probabilities
        else 0.0
    )
    accuracy = correct / len(needle_probabilities) if needle_probabilities else 0.0
    return {
        "needle": needle,
        "avg_probability": average_probability,
        "accuracy": accuracy,
        "context_length": context_length,
        "needle_start_token": insert_at,
        "answer_start_token": answer_start,
        "depth": depth,
        "actual_depth": insert_at / max(1, len(filler)),
        "haystack_unique_token_ratio": len(set(filler)) / max(1, len(filler)),
        "haystack_total_tokens": len(full_ids),
    }
