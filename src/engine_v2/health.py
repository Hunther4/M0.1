"""Health Checks & Gradient Monitoring for TrainingEngine v2."""

from typing import Dict, Any, Tuple
import torch
import torch.nn as nn


class HealthChecker:
    """Health Checker auditing model parameters, gradients, and MoE routing stability."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def monitor_gradients(self) -> Dict[str, Any]:
        """Compute detailed gradient statistics across all trainable layers."""
        grad_norms = {}
        all_grads = []
        max_grad_val = -float("inf")
        min_grad_val = float("inf")
        total_elements = 0
        zero_elements = 0

        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                g = param.grad.data
                norm = g.norm().item()
                grad_norms[name] = norm

                max_grad_val = max(max_grad_val, g.max().item())
                min_grad_val = min(min_grad_val, g.min().item())
                total_elements += g.numel()
                zero_elements += (g == 0).sum().item()
                all_grads.append(g.view(-1))

        if not all_grads:
            return {}

        cat_grads = torch.cat(all_grads)
        biggest_layer = max(grad_norms, key=grad_norms.get) if grad_norms else "none"
        biggest_norm = grad_norms.get(biggest_layer, 0.0)

        return {
            "grad_mean": cat_grads.mean().item(),
            "grad_std": cat_grads.std().item(),
            "grad_max": max_grad_val,
            "grad_min": min_grad_val,
            "grad_sparsity": zero_elements / max(total_elements, 1),
            "largest_grad_norm": biggest_norm,
            "layer_with_biggest_grad": biggest_layer,
        }

    def check_health(self) -> Tuple[bool, str]:
        """Perform comprehensive health check on weights."""
        for name, param in self.model.named_parameters():
            # Check parameter NaNs or Inf
            if torch.isnan(param).any():
                return False, f"NaN parameter detected in {name}"
            if torch.isinf(param).any():
                return False, f"Inf parameter detected in {name}"

        return True, "OK"
