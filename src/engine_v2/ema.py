"""Exponential Moving Average (EMA) for TrainingEngine v2."""

from typing import Dict, Any
import torch
import torch.nn as nn


class EMA:
    """Exponential Moving Average (EMA) of model parameters for validation and inference."""

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        self.model = model
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}
        self.register()

    def register(self) -> None:
        """Register initial parameter shadows matching current parameter devices."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()

    def update(self) -> None:
        """Update shadow parameters with current model weights using in-place lerp."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if name not in self.shadow or self.shadow[name].device != param.device:
                    self.shadow[name] = param.data.clone().detach()
                else:
                    # In-place EMA: shadow = shadow + (1-decay) * (param - shadow)
                    self.shadow[name].lerp_(param.data, 1.0 - self.decay)

    def apply_shadow(self) -> None:
        """Replace model weights with shadow parameters (saving backup)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone().detach()
                if name in self.shadow:
                    param.data.copy_(self.shadow[name].to(param.device))

    def restore(self) -> None:
        """Restore original model weights from backup."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup.clear()

    def state_dict(self) -> Dict[str, Any]:
        """Return state dict of EMA shadow parameters."""
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state dict into EMA shadow parameters."""
        if "decay" in state_dict:
            self.decay = state_dict["decay"]
        if "shadow" in state_dict:
            self.shadow = {k: v.clone().detach() for k, v in state_dict["shadow"].items()}
