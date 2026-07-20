"""AMPContext: Unified precision context (FP16/BF16/FP32) for PyTorch ROCm/CUDA."""

from typing import Any
import torch


class AMPContext:
    """Decoupled precision and gradient scaling context wrapper."""

    def __init__(self, device: torch.device, enabled: bool = True, dtype: str = "float16") -> None:
        self.device = device
        self.enabled = enabled and (device.type == "cuda")

        # Map dtype string to torch dtype
        if dtype == "bfloat16":
            self.amp_dtype = torch.bfloat16
        else:
            self.amp_dtype = torch.float16

        # Generic torch.amp.GradScaler
        self.scaler = torch.amp.GradScaler(device.type, enabled=self.enabled)

    def autocast(self):
        """Return autocast context manager."""
        return torch.amp.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.enabled,
        )

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for backward pass."""
        return self.scaler.scale(loss)

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        """Unscale gradients for clipping."""
        self.scaler.unscale_(optimizer)

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        """Step optimizer with scaled gradients."""
        self.scaler.step(optimizer)

    def update(self) -> None:
        """Update scaler scale factor."""
        self.scaler.update()

    def get_scale(self) -> float:
        """Return current scaler scale value."""
        return self.scaler.get_scale()
