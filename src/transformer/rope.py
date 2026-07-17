"""Rotary Positional Embedding (RoPE).

Implements explicit sin/cos rotation for positional encoding.
Reference: https://arxiv.org/abs/2104.09864

Math:
    θᵢ = 1 / rope_theta^(2i/d_head) for i ∈ [0, d_head/2)
    Position m rotation:
        rotated_even = x_even·cos(mθ) - x_odd·sin(mθ)
        rotated_odd  = x_even·sin(mθ) + x_odd·cos(mθ)
"""

import math
import torch
import torch.nn as nn
from .config import M01Config


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Positional Embedding using explicit sin/cos rotation.
    
    This applies rotation to each dimension pair (even, odd) of the input,
    encoding positional information directly into the representation.
    """
    
    def __init__(self, config: M01Config) -> None:
        """Initialize RoPE with precomputed frequencies.
        
        Args:
            config: M01Config with d_head and rope_theta
        """
        super().__init__()
        
        # Compute inverse frequencies: θᵢ = 1 / rope_theta^(2i/d_head)
        # for i ∈ [0, d_head/2)
        d_head = config.d_head
        freqs = 1.0 / (config.rope_theta ** (torch.arange(0, d_head, 2).float() / d_head))
        
        # Store as buffer (not a parameter, but part of module state)
        # Shape: (d_head//2,)
        self.register_buffer("freqs", freqs)
    
    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """Apply rotary positional embedding to input tensor.
        
        Args:
            x: Input tensor of shape (batch, seq_len, n_heads, d_head)
            offset: Starting position offset (for cached generation)
            
        Returns:
            Rotated tensor of same shape
        """
        batch, seq_len, n_heads, d_head = x.shape
        
        # Create position indices: [offset, offset+1, ..., offset+seq_len-1]
        positions = torch.arange(offset, offset + seq_len, device=x.device).float()
        
        # Compute angles: positions * freqs
        # positions: (seq_len,) → (seq_len, 1)
        # freqs: (d_head//2,) → (1, d_head//2)
        # angles: (seq_len, d_head//2)
        angles = positions.unsqueeze(1) * self.freqs.unsqueeze(0)
        
        # Compute sin and cos
        # Both: (seq_len, d_head//2)
        sin = torch.sin(angles)
        cos = torch.cos(angles)
        
        # Reshape for broadcasting: (1, seq_len, 1, d_head//2)
        sin = sin.unsqueeze(0).unsqueeze(2)
        cos = cos.unsqueeze(0).unsqueeze(2)
        
        # Split x into even and odd indices along last dimension
        x_even = x[..., 0::2]  # (batch, seq_len, n_heads, d_head//2)
        x_odd = x[..., 1::2]   # (batch, seq_len, n_heads, d_head//2)
        
        # Apply rotation:
        # rotated_even = x_even * cos - x_odd * sin
        # rotated_odd  = x_even * sin + x_odd * cos
        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos
        
        # Interleave back: (batch, seq_len, n_heads, d_head)
        rotated = torch.stack([rotated_even, rotated_odd], dim=-1)
        # Flatten last two dimensions to get (..., d_head)
        rotated = rotated.reshape(batch, seq_len, n_heads, d_head)
        
        return rotated