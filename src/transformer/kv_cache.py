"""Key-Value Cache for Autoregressive Generation.

Stores past key/value tensors during autoregressive generation.
Pre-allocates buffers to avoid repeated torch.cat operations.
"""

import torch
from typing import Tuple


class KVCache:
    """Key-Value Cache for storing past attention states.
    
    Pre-allocates buffers of size (batch_size, max_seq_len, n_heads, d_head) for K and V.
    Supports dynamic batch resizing if needed.
    """
    
    def __init__(self, max_seq_len: int, n_heads: int, d_head: int, device: torch.device | None = None) -> None:
        """Initialize KV cache with pre-allocated buffers.
        
        Args:
            max_seq_len: Maximum sequence length to cache
            n_heads: Number of attention heads
            d_head: Dimension per head
            device: Device to allocate tensors on (default: CPU)
        """
        self.max_seq_len = max_seq_len
        self.n_heads = n_heads
        self.d_head = d_head
        self.device = device or torch.device("cpu")
        
        # Pre-allocate K and V buffers with zeros (default batch_size = 1, expands dynamically)
        self.k = torch.zeros(1, max_seq_len, n_heads, d_head, device=self.device)
        self.v = torch.zeros(1, max_seq_len, n_heads, d_head, device=self.device)
        
        # Track current position
        self._seq_len = 0
    
    def append(self, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Append new K/V tensors to cache and return full cached tensors.
        
        Args:
            k: New key tensor of shape (batch, seq_len, n_heads, d_head)
            v: New value tensor of shape (batch, seq_len, n_heads, d_head)
            
        Returns:
            Tuple of (full_k, full_v) tensors up to current length
        """
        batch_size = k.shape[0]
        new_len = k.shape[1]
        
        # Dynamically adjust batch size if it doesn't match the cache allocation
        if self.k.shape[0] != batch_size:
            self.k = torch.zeros(batch_size, self.max_seq_len, self.n_heads, self.d_head, device=self.device, dtype=k.dtype)
            self.v = torch.zeros(batch_size, self.max_seq_len, self.n_heads, self.d_head, device=self.device, dtype=v.dtype)
            self._seq_len = 0
            
        if self._seq_len + new_len > self.max_seq_len:
            raise ValueError(
                f"KV cache capacity exceeded: cached {self._seq_len + new_len} tokens, "
                f"max capacity is {self.max_seq_len}"
            )
        
        # Store at current position
        self.k[:, self._seq_len:self._seq_len + new_len] = k
        self.v[:, self._seq_len:self._seq_len + new_len] = v
        
        # Increment position
        self._seq_len += new_len
        
        # Return sliced tensors up to current length as views (no clone)
        return self.k[:, :self._seq_len], self.v[:, :self._seq_len]
    
    def reset(self) -> None:
        """Reset cache to empty state."""
        self.k = torch.zeros_like(self.k)
        self.v = torch.zeros_like(self.v)
        self._seq_len = 0
    
    @property
    def seq_len(self) -> int:
        """Current cached sequence length."""
        return self._seq_len