"""Health Checks & Gradient Monitoring for TrainingEngine v2."""

from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn


class HealthChecker:
    """Health Checker auditing model parameters, gradients, MoE routing stability, and loss values."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def monitor_gradients(self) -> Dict[str, Any]:
        """Compute detailed gradient statistics across all trainable layers.

        Collect all gradient statistics on the accelerator, then transfer one
        compact summary tensor to the CPU to avoid per-layer synchronization.
        """
        grad_names = []
        grad_norms = []
        all_grads = []
        max_grad_val = None
        min_grad_val = None
        total_elements = 0
        zero_elements = None

        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                g = param.grad.data
                grad_names.append(name)
                grad_norms.append(g.norm())

                layer_max = g.max()
                layer_min = g.min()
                layer_zeros = (g == 0).sum()
                max_grad_val = (
                    layer_max
                    if max_grad_val is None
                    else torch.maximum(max_grad_val, layer_max)
                )
                min_grad_val = (
                    layer_min
                    if min_grad_val is None
                    else torch.minimum(min_grad_val, layer_min)
                )
                total_elements += g.numel()
                zero_elements = (
                    layer_zeros if zero_elements is None else zero_elements + layer_zeros
                )
                all_grads.append(g.view(-1))

        if not all_grads:
            return {}

        cat_grads = torch.cat(all_grads)
        layer_norms = torch.stack(grad_norms)
        biggest_index = torch.argmax(layer_norms)
        biggest_norm = layer_norms[biggest_index]

        summary = torch.stack(
            (
                cat_grads.mean().to(dtype=torch.float64),
                cat_grads.std().to(dtype=torch.float64),
                max_grad_val.to(dtype=torch.float64),
                min_grad_val.to(dtype=torch.float64),
                (zero_elements.to(dtype=torch.float64) / max(total_elements, 1)),
                biggest_norm.to(dtype=torch.float64),
                biggest_index.to(dtype=torch.float64),
            )
        )
        (
            grad_mean,
            grad_std,
            grad_max,
            grad_min,
            grad_sparsity,
            largest_grad_norm,
            biggest_index,
        ) = summary.detach().to(device="cpu", dtype=torch.float64).tolist()

        return {
            "grad_mean": grad_mean,
            "grad_std": grad_std,
            "grad_max": grad_max,
            "grad_min": grad_min,
            "grad_sparsity": grad_sparsity,
            "largest_grad_norm": largest_grad_norm,
            "layer_with_biggest_grad": grad_names[int(biggest_index)],
        }

    def check_loss(self, loss_value: Optional[float]) -> Tuple[bool, str]:
        """Check whether the loss value is valid (not NaN, Inf, or None).

        This is the FIRST line of defence: a NaN loss means the forward pass
        produced an invalid signal and the optimizer step MUST be skipped
        to prevent weight corruption.

        Args:
            loss_value: The scalar loss from the forward pass (``step_loss``).

        Returns:
            ``(True, "OK")`` if the loss is valid and finite.
            ``(False, reason)`` if the loss is NaN, Inf, or None.
        """
        if loss_value is None:
            return False, "Loss is None"
        if not isinstance(loss_value, (int, float)):
            # Torch scalar tensor — convert for the check
            if isinstance(loss_value, torch.Tensor):
                if loss_value.numel() != 1:
                    return False, f"Loss is not a scalar; shape={list(loss_value.shape)}"
                loss_value = loss_value.item()
            else:
                return False, f"Loss has unexpected type: {type(loss_value).__name__}"
        # NaN check FIRST (NaN != NaN per IEEE 754) — must come before the range
        # check because comparisons with NaN are always False.
        if loss_value != loss_value:
            return False, "Loss is NaN"
        if abs(loss_value) == float("inf"):
            return False, f"Loss is Inf: {loss_value}"
        if not (loss_value <= 1e10 and loss_value >= -1e10):
            return False, f"Loss is out of plausible range: {loss_value}"
        return True, "OK"

    def check_health(self) -> Tuple[bool, str]:
        """Perform comprehensive health check on weights."""
        for name, param in self.model.named_parameters():
            # Check parameter NaNs or Inf
            if torch.isnan(param).any():
                return False, f"NaN parameter detected in {name}"
            if torch.isinf(param).any():
                return False, f"Inf parameter detected in {name}"

        return True, "OK"
