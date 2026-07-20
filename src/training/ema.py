"""Exponential Moving Average (EMA) for model parameters.

Maintains shadow weights with Polyak averaging:
    shadow = decay * shadow + (1 - decay) * model_param

EMA weights typically produce better validation metrics than
the final training weights.
"""

from typing import Any
import torch
import torch.nn as nn


class ModelEMA:
    """EMA wrapper that maintains and applies shadow model parameters.

    Usage:
        ema = ModelEMA(model, decay=0.999)
        for step in range(steps):
            loss.backward()
            optimizer.step()
            ema.update()  # after optimizer step

        # Use EMA weights for validation/checkpoint
        ema.apply_shadow()
        val_loss = validate(model)
        ema.restore()
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.model = model
        self.shadow: dict[str, torch.Tensor] = {}
        self.backup: dict[str, torch.Tensor] = {}
        self._in_place: bool = False

        # Register model parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()

    def update(self) -> None:
        """Update shadow weights with Polyak averaging.

        Call after optimizer.step().
        Skips if shadow is in_place (can't update while applied).
        """
        if self._in_place:
            return
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.shadow and param.requires_grad:
                    self.shadow[name].lerp_(param.data, 1.0 - self.decay)

    def apply_shadow(self) -> None:
        """Swap model weights to EMA shadow. Call before validation."""
        if self._in_place:
            return
        self.backup.clear()
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.shadow and param.requires_grad:
                    self.backup[name] = param.data.clone().detach()
                    param.data.copy_(self.shadow[name])
        self._in_place = True

    def restore(self) -> None:
        """Restore original model weights. Call after validation."""
        if not self._in_place:
            return
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.backup and param.requires_grad:
                    param.data.copy_(self.backup[name])
        self.backup.clear()
        self._in_place = False

    def state_dict(self) -> dict:
        """Return EMA state for checkpointing."""
        return {
            "decay": self.decay,
            "shadow": {k: v.clone() for k, v in self.shadow.items()},
        }

    def load_state_dict(self, state: dict) -> None:
        """Load EMA state from checkpoint."""
        self.decay = state.get("decay", self.decay)
        for k, v in state.get("shadow", {}).items():
            if k in self.shadow:
                self.shadow[k].copy_(v)
