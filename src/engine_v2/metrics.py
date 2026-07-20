"""Dynamic Metric Registry & Advanced MoE Dashboard Metrics."""

import math
from typing import Dict, Any, List
import torch


class MetricRegistry:
    """Dynamic Metric Registry storing and computing research metrics."""

    def __init__(self) -> None:
        self._metrics: Dict[str, Any] = {}

    def update(self, key: str, value: Any) -> None:
        """Update a metric value."""
        self._metrics[key] = value

    def get_all(self) -> Dict[str, Any]:
        """Return a copy of all current metrics."""
        return dict(self._metrics)

    @staticmethod
    def compute_gini_index(expert_counts: torch.Tensor) -> float:
        """Calculate Gini coefficient of expert usage (0.0 = perfect equality, 1.0 = total inequality)."""
        if expert_counts.sum() == 0:
            return 0.0
        sorted_counts, _ = torch.sort(expert_counts.float())
        n = len(sorted_counts)
        index = torch.arange(1, n + 1, device=expert_counts.device).float()
        return ((2 * index - n - 1) * sorted_counts).sum().item() / (n * sorted_counts.sum().item())

    @staticmethod
    def compute_kl_divergence(gate_probs: torch.Tensor) -> float:
        """Calculate KL divergence between router distribution and uniform distribution."""
        if gate_probs.numel() == 0:
            return 0.0
        avg_probs = gate_probs.mean(dim=0)
        num_experts = len(avg_probs)
        uniform = torch.full_like(avg_probs, 1.0 / num_experts)
        kl = torch.sum(avg_probs * torch.log((avg_probs + 1e-8) / uniform)).item()
        return max(0.0, kl)
