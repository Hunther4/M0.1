"""RMS Normalization Layer.

RMSNorm normalizes inputs by root mean square along the last dimension,
then scales by a learnable gamma parameter. No shift/bias parameter.

Formula: y = (x / sqrt(mean(x^2) + eps)) * gamma

Reference: https://arxiv.org/abs/1910.07467
"""

import torch
import torch.nn as nn
from torch import Tensor


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Args:
        d_model: Input feature dimension
        eps: Small constant for numerical stability (default: 1e-5)
    """

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))

    def forward(self, x: Tensor) -> Tensor:
        """Apply RMSNorm.

        Args:
            x: Input tensor of shape (..., d_model)

        Returns:
            Normalized tensor of the same shape
        """
        # Compute RMS along last dimension: sqrt(mean(x^2) + eps)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # Normalize and scale
        return (x / rms) * self.gamma
