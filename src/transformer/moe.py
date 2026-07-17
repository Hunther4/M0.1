"""Mixture of Experts (MoE) Layer - Placeholder.

Placeholder implementation for MoE that routes to FeedForward experts.
Currently uses first expert only. Will be expanded in Fase 2.
"""

import torch
import torch.nn as nn
from .config import M01Config
from .feedforward import FeedForward


class MoELayer(nn.Module):
    """Mixture of Experts layer - placeholder implementation.
    
    Routes input to FeedForward experts. Currently uses first expert only.
    Gate network will be implemented in Fase 2 for top-k routing.
    """
    
    def __init__(self, config: M01Config) -> None:
        """Initialize MoE layer.
        
        Args:
            config: M01Config with num_experts, d_model, d_ff
        """
        super().__init__()
        
        self.config = config
        self.num_experts = config.num_experts
        
        # Create expert networks
        self.experts = nn.ModuleList([
            FeedForward(config) for _ in range(config.num_experts)
        ])
        
        # Gate network for routing (will be used in Fase 2)
        # For now, it's a simple linear layer
        self.gate = nn.Linear(config.d_model, config.num_experts, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass - currently routes to first expert only.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        # Placeholder: just use first expert
        # In Fase 2, this will implement top-k routing
        return self.experts[0](x)