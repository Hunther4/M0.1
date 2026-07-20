"""AMPContext: Unified Mixed Precision Wrapper for TrainingEngine v2."""

from typing import Dict, Any, Optional
import torch


class AMPContext:
    """Generic AMP Context supporting PyTorch 2.x torch.amp.autocast and GradScaler."""

    def __init__(self, device: torch.device, enabled: bool = True) -> None:
        self.device = device
        self.enabled = enabled and (device.type == "cuda")
        self.device_type = "cuda" if device.type == "cuda" else "cpu"
        self._is_unscaled = False

        if self.enabled:
            self.scaler: Optional[torch.amp.GradScaler] = torch.amp.GradScaler(self.device_type)
        else:
            self.scaler = None

    def autocast(self) -> torch.amp.autocast:
        """Return autocast context manager."""
        return torch.amp.autocast(device_type=self.device_type, enabled=self.enabled)

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss tensor if scaler active."""
        if self.enabled and self.scaler is not None:
            return self.scaler.scale(loss)
        return loss

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        """Unscale optimizer gradients safely."""
        if self.enabled and self.scaler is not None and not self._is_unscaled:
            try:
                self.scaler.unscale_(optimizer)
                self._is_unscaled = True
            except RuntimeError:
                pass

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        """Step optimizer via scaler."""
        if self.enabled and self.scaler is not None:
            self.scaler.step(optimizer)
        else:
            optimizer.step()

    def update(self) -> None:
        """Update scaler scale factor."""
        if self.enabled and self.scaler is not None:
            self.scaler.update()
            self._is_unscaled = False

    def state_dict(self) -> Dict[str, Any]:
        """Return state dict of GradScaler."""
        if self.enabled and self.scaler is not None:
            return self.scaler.state_dict()
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state dict into GradScaler."""
        if self.enabled and self.scaler is not None and state_dict:
            self.scaler.load_state_dict(state_dict)
