"""MoE runtime metrics computation, logging, and router collapse detection.

Provides pure functions for computing metrics from MoE layers after a forward
pass, a duck-typed logging protocol, and collapse detection logic.
"""

import math
from typing import Protocol

import torch
import torch.nn as nn


def compute_moe_metrics(model: nn.Module) -> dict:
    """Compute MoE routing metrics from a model's MoE layers.

    Iterates the model's transformer blocks and collects metrics from any
    MoELayer instances that have stored gate_probs (i.e., have been forwarded
    in training mode).

    Args:
        model: A TransformerLM (or compatible nn.Module) with .blocks and .config.

    Returns:
        dict with per-layer and global metrics keys, or {} if no MoE layers
        or none have been forwarded in training mode yet.
    """
    # Quick check: if model config says dense, skip
    if getattr(model.config, "num_experts", 1) <= 1:
        return {}

    metrics: dict = {}
    total_entropy = 0.0
    n_layers_moe = 0
    total_routed_tokens = 0

    for i, block in enumerate(model.blocks):
        ff = block.ff
        if not hasattr(ff, "gate_probs"):
            continue  # Dense FeedForward layer, skip

        n_layers_moe += 1
        n_experts = ff.num_experts
        layer_key = f"layer_{i}"

        # aux_loss
        metrics[f"{layer_key}/aux_loss"] = ff.get_aux_loss().item()

        # expert_usage_histogram: token count per expert
        hist = torch.bincount(ff.topk_indices.flatten(), minlength=n_experts)
        metrics[f"{layer_key}/histogram"] = hist.tolist()

        # expert_load_std and coefficient of variation
        hist_float = hist.float()
        std = float(torch.std(hist_float))
        mean = float(hist_float.mean())
        metrics[f"{layer_key}/load_std"] = std
        metrics[f"{layer_key}/load_cv"] = std / mean if mean > 0 else 0.0

        # router entropy: per-token entropy normalized by log(num_experts)
        probs = ff.gate_probs
        per_token_entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
        mean_entropy = float(per_token_entropy.mean().detach())
        normalized = mean_entropy / math.log(n_experts) if n_experts > 1 else 0.0
        metrics[f"{layer_key}/entropy"] = min(normalized, 1.0)

        total_entropy += normalized
        total_routed_tokens += int(hist.sum().item())

    # If no MoE layers with stored data were found, return empty
    if n_layers_moe == 0:
        return {}

    # Global aggregates
    metrics["global/mean_entropy"] = total_entropy / n_layers_moe
    metrics["global/n_layers_moe"] = n_layers_moe
    metrics["global/total_routed_tokens"] = total_routed_tokens

    # Router collapse: any expert with 0 tokens across all layers?
    all_dead: list[int] = []
    for key, val in metrics.items():
        if key.endswith("/histogram"):
            dead = [i for i, count in enumerate(val) if count == 0]
            all_dead.extend(dead)
    metrics["global/router_collapse"] = len(all_dead) > 0

    return metrics


class MetricsLogger(Protocol):
    """Protocol for MoE metric loggers.

    Any object with a ``log(metrics, step)`` method satisfies this protocol
    (duck typing). New backends (wandb, tensorboard) just need to match
    the signature.
    """

    def log(self, metrics: dict, step: int) -> None: ...


class ConsoleLogger:
    """Console-based metrics logger that prints formatted metric values."""

    def log(self, metrics: dict, step: int) -> None:
        """Print formatted metrics to stdout.

        Args:
            metrics: dict from compute_moe_metrics()
            step: Current training step number.
        """
        if not metrics:
            print(f"[MoE Metrics] Step {step}: (no MoE metrics)")
            return

        print(f"\n{'=' * 50}")
        print(f"  MoE Metrics — Step {step}")
        print(f"{'=' * 50}")

        # Print per-layer metrics
        for key in sorted(metrics.keys()):
            if key.startswith("global/"):
                continue
            val = metrics[key]
            if isinstance(val, float):
                print(f"  {key:<30s} {val:.6f}")
            else:
                print(f"  {key:<30s} {val}")

        # Print global aggregates
        print(f"  {'─' * 40}")
        for key in sorted(metrics.keys()):
            if not key.startswith("global/"):
                continue
            val = metrics[key]
            if isinstance(val, float):
                print(f"  {key:<30s} {val:.6f}")
            else:
                print(f"  {key:<30s} {val}")

        print(f"{'=' * 50}\n")


def detect_router_collapse(
    histogram: torch.Tensor | list[int],
    counter: int,
    threshold: int,
) -> tuple[bool, int]:
    """Detect router collapse based on consecutive zero-expert steps.

    Tracks a streak counter for consecutive steps where at least one
    expert receives zero tokens. When the streak reaches the threshold,
    signals collapse. The counter resets to 0 whenever all experts
    receive at least one token.

    Args:
        histogram: 1D tensor or list of token counts per expert.
        counter: Current streak count of consecutive steps with a dead expert.
        threshold: Max allowed consecutive dead-expert steps before stopping.

    Returns:
        Tuple of (should_stop, updated_counter).
    """
    if isinstance(histogram, torch.Tensor):
        has_dead = bool((histogram == 0).any().item())
    else:
        has_dead = any(count == 0 for count in histogram)

    if has_dead:
        counter += 1
    else:
        counter = 0

    return counter >= threshold, counter
