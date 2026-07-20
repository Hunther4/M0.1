"""MoE runtime metrics: 20+ metrics across 4 categories.

All metrics are computed from data already stored by MoELayer.forward().
No extra forward passes needed. Each category is a pure function.
"""

import math
from typing import Any, Protocol

import torch
import torch.nn as nn


# ── Distribution Metrics ──────────────────────────────────────────────────────

def expert_usage_histogram(moe: nn.Module) -> list[int]:
    """Token count assigned to each routed expert this step."""
    hist = torch.bincount(moe.topk_indices.flatten(), minlength=moe.num_experts)
    return hist.tolist()


def tokens_per_expert(moe: nn.Module) -> list[int]:
    """Same as histogram — tokens per expert this step."""
    return expert_usage_histogram(moe)


def expert_utilization_pct(moe: nn.Module, capacity: int | None = None) -> list[float]:
    """Utilization % of each expert relative to capacity (if set) or max observed."""
    hist = torch.tensor(expert_usage_histogram(moe), dtype=torch.float)
    if capacity is not None and capacity > 0:
        cap = float(capacity)
    else:
        cap = float(hist.max().clamp(min=1).item())
    return (hist / cap).tolist()


def gini_coefficient(moe: nn.Module) -> float:
    """Gini coefficient for expert token distribution. 0 = perfect balance, 1 = monopoly."""
    hist = torch.tensor(expert_usage_histogram(moe), dtype=torch.float)
    if hist.sum() <= 0:
        return 0.0
    sorted_h = torch.sort(hist)[0]
    n = len(sorted_h)
    cumsum = torch.cumsum(sorted_h, dim=0)
    # Gini = (2 * sum(i * x_i)) / (n * sum(x_i)) - (n+1)/n
    gini = (2.0 * torch.arange(1, n + 1, dtype=torch.float) @ sorted_h) / (n * hist.sum()) - (n + 1.0) / n
    return max(0.0, min(1.0, gini.item()))


def imbalance_ratio(moe: nn.Module) -> float:
    """max_usage / min_usage among experts. 1.0 = perfectly balanced."""
    hist = torch.tensor(expert_usage_histogram(moe), dtype=torch.float)
    min_val = hist[hist > 0].min().item() if (hist > 0).any() else 1.0
    max_val = hist.max().item()
    return max_val / max(min_val, 1.0)


# ── Router Metrics ────────────────────────────────────────────────────────────

def router_entropy(moe: nn.Module, normalize: bool = True) -> float:
    """Mean per-token normalized entropy: 0 = deterministic, 1 = uniform."""
    probs = moe.gate_probs
    n = moe.num_experts
    per_token = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
    mean = float(per_token.mean().item())
    if normalize and n > 1:
        return min(mean / math.log(n), 1.0)
    return mean


def router_confidence(moe: nn.Module) -> float:
    """Mean max probability assigned by router. High = confident routing."""
    return float(moe.gate_probs.max(dim=-1).values.mean().item())


def gate_logits_std(moe: nn.Module) -> float:
    """Std dev of raw gate logits. Growing std = router getting confident."""
    return float(moe.gate_logits.std().item())


def gate_logits_mean(moe: nn.Module) -> float:
    """Mean of raw gate logits."""
    return float(moe.gate_logits.mean().item())


def top1_frequency(moe: nn.Module, num_experts: int | None = None) -> list[float]:
    """Fraction of steps each expert is the #1 choice."""
    n = num_experts if num_experts is not None else moe.num_experts
    top1 = moe.topk_indices[:, 0]  # (N,) — top-1 expert per token
    freq = torch.bincount(top1, minlength=n).float() / max(top1.size(0), 1)
    return freq.tolist()


def top2_frequency(moe: nn.Module, num_experts: int | None = None) -> list[float]:
    """Fraction of steps each expert appears in top-2."""
    n = num_experts if num_experts is not None else moe.num_experts
    if moe.topk_indices.size(-1) < 2:
        return [0.0] * n
    top2 = moe.topk_indices[:, :2].flatten()
    freq = torch.bincount(top2, minlength=n).float() / max(moe.topk_indices.size(0), 1)
    return freq.tolist()


def top4_overlap(moe: nn.Module) -> float:
    """Not meaningful with tk < 4. Returns 0.0 until Stage 4+."""
    return 0.0  # placeholder for Stage 4


# ── Health Metrics ────────────────────────────────────────────────────────────

def dead_expert_info(moe: nn.Module, num_experts: int | None = None) -> dict:
    """Which experts received 0 tokens this step."""
    n = num_experts if num_experts is not None else moe.num_experts
    hist = torch.bincount(moe.topk_indices.flatten(), minlength=n)
    return {
        "dead_count": int((hist == 0).sum().item()),
        "dead_ids": [int(i) for i in range(n) if hist[i].item() == 0],
    }


def expert_saturation(moe: nn.Module, capacity: int | None = None) -> list[float]:
    """How close each expert is to capacity. 1.0 = full."""
    hist = torch.tensor(expert_usage_histogram(moe), dtype=torch.float)
    if capacity is not None and capacity > 0:
        cap = float(capacity)
    else:
        # Estimate capacity as tokens/experts * top_k
        total = float(hist.sum().item())
        cap = max(total / max(moe.num_experts, 1), 1.0)
    return (hist / cap).tolist()


def aux_loss_ema(current: float, prev_ema: float, decay: float = 0.99) -> float:
    """Exponential moving average of aux loss."""
    return decay * prev_ema + (1.0 - decay) * current


# ── Quality Metrics (lightweight, no extra forward passes) ────────────────────

def expert_kl_divergence(moe: nn.Module, num_experts: int | None = None) -> float:
    """Mean KL between gate probability distributions (pairwise). O(N^2)."""
    n = num_experts if num_experts is not None else moe.num_experts
    probs = moe.gate_probs  # (N, n)
    # Compute mean distribution per expert (weighted by assignment)
    # Simplified: average gate prob distribution = uniform target
    avg_dist = probs.mean(dim=0)  # (n,)
    uniform = torch.full_like(avg_dist, 1.0 / n)
    kl = (avg_dist * (torch.log(avg_dist + 1e-8) - torch.log(uniform + 1e-8))).sum()
    return float(kl.item())


def shared_expert_usage(moe: nn.Module, num_shared: int | None = None) -> dict:
    """Shared experts are always active for all tokens. Reports count."""
    return {
        "num_shared": num_shared if num_shared is not None else moe.num_shared_experts,
        "active_always": True,
    }


def detect_router_collapse(
    histogram: torch.Tensor | list[int],
    counter: int,
    threshold: int,
) -> tuple[bool, int]:
    """Detect router collapse based on consecutive zero-expert steps."""
    if isinstance(histogram, torch.Tensor):
        has_dead = bool((histogram == 0).any().item())
    else:
        has_dead = any(count == 0 for count in histogram)
    if has_dead:
        counter += 1
    else:
        counter = 0
    return counter >= threshold, counter


# ── Orchestrator ──────────────────────────────────────────────────────────────

def compute_moe_metrics(model: nn.Module) -> dict:
    """Compute ALL MoE metrics from model's MoE layers.

    Collects routing metrics per layer and global aggregates.
    Returns empty dict if no MoE layers with routing data exist.
    """
    # Quick check: no MoE layers or dense-only model
    if not hasattr(model, "blocks") or getattr(getattr(model, "config", None), "num_experts", 1) <= 1:
        return {}

    metrics: dict[str, Any] = {}
    layer_entropies: list[float] = []
    layer_gini: list[float] = []
    layer_dead_counts: list[int] = []
    n_layers_moe = 0

    for i, block in enumerate(model.blocks):
        ff = block.ff
        if not hasattr(ff, "gate_probs"):
            continue
        if not hasattr(ff, "num_experts") or ff.num_experts <= 1:
            continue

        n_layers_moe += 1
        n_exp = ff.num_experts
        prefix = f"layer_{i}"

        # ── Distribution ──
        hist = expert_usage_histogram(ff)
        metrics[f"{prefix}/histogram"] = hist
        metrics[f"{prefix}/gini"] = gini_coefficient(ff)
        metrics[f"{prefix}/imbalance_ratio"] = imbalance_ratio(ff)
        metrics[f"{prefix}/util_pct"] = expert_utilization_pct(ff)

        # ── Router ──
        metrics[f"{prefix}/entropy"] = router_entropy(ff)
        metrics[f"{prefix}/confidence"] = router_confidence(ff)
        metrics[f"{prefix}/gate_logits_std"] = gate_logits_std(ff)
        metrics[f"{prefix}/gate_logits_mean"] = gate_logits_mean(ff)
        metrics[f"{prefix}/top1_freq"] = top1_frequency(ff, n_exp)
        if ff.moe_top_k >= 2:
            metrics[f"{prefix}/top2_freq"] = top2_frequency(ff, n_exp)
        metrics[f"{prefix}/topk"] = ff.moe_top_k

        # ── Health ──
        dead = dead_expert_info(ff, n_exp)
        metrics[f"{prefix}/dead_count"] = dead["dead_count"]
        metrics[f"{prefix}/dead_ids"] = dead["dead_ids"]
        metrics[f"{prefix}/saturation"] = expert_saturation(ff)

        # ── Quality ──
        metrics[f"{prefix}/kl_vs_uniform"] = expert_kl_divergence(ff, n_exp)
        metrics[f"{prefix}/aux_loss"] = ff.get_aux_loss().item()

        # Collect for global aggregates
        layer_entropies.append(metrics[f"{prefix}/entropy"])
        layer_gini.append(metrics[f"{prefix}/gini"])
        layer_dead_counts.append(dead["dead_count"])

    if n_layers_moe == 0:
        return {}

    # ── Global Aggregates ──
    metrics["global/mean_entropy"] = sum(layer_entropies) / n_layers_moe
    metrics["global/mean_gini"] = sum(layer_gini) / n_layers_moe
    metrics["global/total_dead"] = sum(layer_dead_counts)
    metrics["global/n_layers_moe"] = n_layers_moe
    metrics["global/num_experts"] = model.config.num_experts
    metrics["global/moe_top_k"] = model.config.moe_top_k
    metrics["global/num_shared"] = model.config.num_shared_experts
    metrics["global/router_collapse"] = sum(layer_dead_counts) >= model.config.num_experts * n_layers_moe

    return metrics


# ── Protocols ──────────────────────────────────────────────────────────────────

class MetricsLogger(Protocol):
    """Protocol for MoE metric loggers (duck typing).

    Any object with a ``log(metrics, step)`` method satisfies this protocol.
    """

    def log(self, metrics: dict, step: int) -> None: ...


# ── Console Logger ────────────────────────────────────────────────────────────

class ConsoleLogger:
    """Formatted console output of MoE metrics."""

    def log(self, metrics: dict, step: int) -> None:
        if not metrics:
            print(f"\n  MoE Metrics — Step {step}: (no MoE metrics)", flush=True)
            return
        n_moe = metrics.get("global/n_layers_moe", 0)
        if n_moe == 0:
            print(f"\n  MoE Metrics — Step {step}: (no moe layers)", flush=True)
            return

        print(f"\n  MoE Metrics — Step {step}")
        print(f"  Entropy: {metrics.get('global/mean_entropy', 0):.3f}  "
              f"Gini: {metrics.get('global/mean_gini', 0):.3f}  "
              f"Dead: {metrics.get('global/total_dead', 0)}  "
              f"Collapse: {'⚠' if metrics.get('global/router_collapse', False) else '✓'}")
        print(end="", flush=True)
