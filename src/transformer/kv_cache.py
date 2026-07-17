"""Key-Value Cache for Autoregressive Generation.

Separate KVCache class for storing past key/value tensors during autoregressive generation.
Pre-allocates buffers to avoid repeated torch.cat operations.
"""

import torch
from typing import Tuple


class KVCache:
    """Key-Value Cache for storing past attention states.
    
    Pre-allocates buffers of size (1, max_seq_len, n_heads, d_head) for K and V.
    During generation, new K/V are appended at the current position.
    """
    
    def __init__(self, max_seq_len: int, n_heads: int, d_head: int) -> None:
        """Initialize KV cache with pre-allocated buffers.
        
        Args:
            max_seq_len: Maximum sequence length to cache
            n_heads: Number of attention heads
            d_head: Dimension per head
        """
        self.max_seq_len = max_seq_len
        self.n_heads = n_heads
        self.d_head = d_head
        
        # Pre-allocate K and V buffers with zeros
        # Shape: (1, max_seq_len, n_heads, d_head)
        self.k = torch.zeros(1, max_seq_len, n_heads, d_head)
        self.v = torch.zeros(1, max_seq_len, n_heads, d_head)
        
        # Track current position
        self._seq_len = 0
    
    def append(self, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Append new K/V tensors to cache and return full cached tensors.
        
        Args:
            k: New key tensor of shape (batch, 1, n_heads, d_head)
            v: New value tensor of shape (batch, 1, n_heads, d_head)
            
        Returns:
            Tuple of (full_k, full_v) tensors up to current length
        """
        batch_size = k.shape[0]
        assert batch_size == 1, "Batch size must be 1 for KV cache"
        assert k.shape[1] == 1, "Sequence length must be 1 for single append"
        
        # Store at current position
        self.k[:, self._seq_len:self._seq_len + 1] = k
        self.v[:, self._seq_len:self._seq_len + 1] = v
        
        # Increment position
        self._seq_len += 1
        
        # Return sliced tensors up to current length
        return self.k[:, :self._seq_len].clone(), self.v[:, :self._seq_len].clone()
    
    def reset(self) -> None:
        """Reset cache to empty state."""
        self.k.zero_()
        self.v.zero_()
        self._seq_len = 0
    
    @property
    def seq_len(self) -> int:
        """Current cached sequence length."""
        return self._seq_len