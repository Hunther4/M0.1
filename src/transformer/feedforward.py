"""Feedforward Network with SwiGLU Activation.

Dense SwiGLU feedforward network, MoE-ready via num_experts parameter.
SwiGLU: down(SiLU(gate(x)) ⊙ up(x))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import M01Config


class FeedForward(nn.Module):
    """Feedforward network with SwiGLU activation.
    
    When num_experts=1 (default), this is a dense SwiGLU layer.
    When num_experts>1, this can be used as a single expert in MoE.
    """
    
    def __init__(self, config: M01Config) -> None:
        """Initialize feedforward network.
        
        Args:
            config: M01Config with d_model, d_ff
        """
        super().__init__()
        
        # SwiGLU projections
        self.gate_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=False)
        
        # Store config for potential MoE usage
        self.config = config
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with SwiGLU activation.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        # SwiGLU: down(SiLU(gate(x)) ⊙ up(x))
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        
        # SiLU activation: x * sigmoid(x)
        activated = F.silu(gate)
        
        # Element-wise multiplication
        hidden = activated * up
        
        # Down projection
        output = self.down_proj(hidden)
        
        return output