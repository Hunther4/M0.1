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
    
    def __init__(self, max_seq_len: int, n_heads: int, d_head: int, device: torch.device | None = None, dtype: torch.dtype | None = None) -> None:
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
        self.dtype = dtype or torch.float32
        
        # Pre-allocate K and V buffers with zeros (default batch_size = 1, expands dynamically)
        self.k = torch.zeros(1, max_seq_len, n_heads, d_head, device=self.device, dtype=self.dtype)
        self.v = torch.zeros(1, max_seq_len, n_heads, d_head, device=self.device, dtype=self.dtype)
        
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

        if self._seq_len and batch_size != self.k.shape[0]:
            raise ValueError(
                "KV cache batch size cannot change after tokens are cached; reset the cache first"
            )
        if self._seq_len and (k.device != self.k.device or k.dtype != self.k.dtype or v.device != self.v.device or v.dtype != self.v.dtype):
            raise ValueError("KV cache device and dtype cannot change after tokens are cached")

        if self._seq_len == 0 and (k.device != self.k.device or k.dtype != self.k.dtype):
            self.device = k.device
            self.dtype = k.dtype
        
        # Dynamically adjust batch size if it doesn't match the cache allocation
        if self.k.shape[0] != batch_size or self.k.device != k.device or self.k.dtype != k.dtype:
            self.k = torch.zeros(batch_size, self.max_seq_len, self.n_heads, self.d_head, device=k.device, dtype=k.dtype)
            self.v = torch.zeros(batch_size, self.max_seq_len, self.n_heads, self.d_head, device=v.device, dtype=v.dtype)
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


class MLAKVCache:
    """Cache the compressed MLA latent and positional RoPE key per layer."""

    def __init__(
        self,
        max_seq_len: int,
        n_heads: int,
        latent_dim: int,
        rope_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.max_seq_len = max_seq_len
        self.n_heads = n_heads
        self.latent_dim = latent_dim
        self.rope_dim = rope_dim
        self.device = device or torch.device("cpu")
        self.dtype = dtype or torch.float32
        self.latent = torch.zeros(
            1, max_seq_len, latent_dim, device=self.device, dtype=self.dtype
        )
        self.rope = torch.zeros(
            1, max_seq_len, n_heads, rope_dim, device=self.device, dtype=self.dtype
        )
        self._seq_len = 0

    def append(
        self, latent: torch.Tensor, rope: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if latent.ndim != 3 or latent.shape[-1] != self.latent_dim:
            raise ValueError("MLA latent has an incompatible shape")
        if rope.ndim != 4 or rope.shape[2:] != (self.n_heads, self.rope_dim):
            raise ValueError("MLA RoPE key has an incompatible shape")
        if latent.shape[:2] != rope.shape[:2]:
            raise ValueError("MLA latent and RoPE key must share batch and sequence shapes")

        batch_size, new_len = latent.shape[:2]
        if self._seq_len and batch_size != self.latent.shape[0]:
            raise ValueError("MLA cache batch size cannot change after tokens are cached")
        if self._seq_len and (
            latent.device != self.latent.device
            or latent.dtype != self.latent.dtype
            or rope.device != self.rope.device
            or rope.dtype != self.rope.dtype
        ):
            raise ValueError("MLA cache device and dtype cannot change after tokens are cached")
        if self._seq_len + new_len > self.max_seq_len:
            raise ValueError(
                f"MLA cache capacity exceeded: cached {self._seq_len + new_len} tokens, "
                f"max capacity is {self.max_seq_len}"
            )

        if self._seq_len == 0 and (
            self.latent.shape[0] != batch_size
            or self.latent.device != latent.device
            or self.latent.dtype != latent.dtype
        ):
            self.device, self.dtype = latent.device, latent.dtype
            self.latent = torch.zeros(
                batch_size,
                self.max_seq_len,
                self.latent_dim,
                device=latent.device,
                dtype=latent.dtype,
            )
            self.rope = torch.zeros(
                batch_size,
                self.max_seq_len,
                self.n_heads,
                self.rope_dim,
                device=rope.device,
                dtype=rope.dtype,
            )

        end = self._seq_len + new_len
        self.latent[:, self._seq_len:end] = latent
        self.rope[:, self._seq_len:end] = rope
        self._seq_len = end
        return self.latent[:, :end], self.rope[:, :end]

    def reset(self) -> None:
        self.latent.zero_()
        self.rope.zero_()
        self._seq_len = 0

    @property
    def seq_len(self) -> int:
        return self._seq_len


class HybridKVCache:
    """Store CSA and HCA projections and expose the active hybrid history."""

    def __init__(
        self,
        max_seq_len: int,
        n_heads: int,
        d_head: int,
        local_window_size: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        if local_window_size < 1:
            raise ValueError("local_window_size must be positive")
        self.max_seq_len = max_seq_len
        self.n_heads = n_heads
        self.d_head = d_head
        self.local_window_size = local_window_size
        self.device = device or torch.device("cpu")
        self.dtype = dtype or torch.float32
        shape = (1, max_seq_len, n_heads, d_head)
        self.k_csa = torch.zeros(shape, device=self.device, dtype=self.dtype)
        self.v_csa = torch.zeros_like(self.k_csa)
        self.k_hca = torch.zeros_like(self.k_csa)
        self.v_hca = torch.zeros_like(self.k_csa)
        self._seq_len = 0

    def append(
        self,
        k_csa: torch.Tensor,
        v_csa: torch.Tensor,
        k_hca: torch.Tensor,
        v_hca: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tensors = (k_csa, v_csa, k_hca, v_hca)
        expected_tail = (self.n_heads, self.d_head)
        if any(t.ndim != 4 or t.shape[2:] != expected_tail for t in tensors):
            raise ValueError("Hybrid cache tensors have an incompatible shape")
        if any(t.shape[:2] != k_csa.shape[:2] for t in tensors[1:]):
            raise ValueError("Hybrid cache tensors must share batch and sequence shapes")

        batch_size, new_len = k_csa.shape[:2]
        if self._seq_len and batch_size != self.k_csa.shape[0]:
            raise ValueError("Hybrid cache batch size cannot change after tokens are cached")
        if self._seq_len and any(
            t.device != self.k_csa.device or t.dtype != self.k_csa.dtype for t in tensors
        ):
            raise ValueError("Hybrid cache device and dtype cannot change after tokens are cached")
        if self._seq_len + new_len > self.max_seq_len:
            raise ValueError(
                f"Hybrid cache capacity exceeded: cached {self._seq_len + new_len} tokens, "
                f"max capacity is {self.max_seq_len}"
            )

        if self._seq_len == 0 and (
            self.k_csa.shape[0] != batch_size
            or self.k_csa.device != k_csa.device
            or self.k_csa.dtype != k_csa.dtype
        ):
            self.device, self.dtype = k_csa.device, k_csa.dtype
            shape = (batch_size, self.max_seq_len, self.n_heads, self.d_head)
            self.k_csa = torch.zeros(shape, device=self.device, dtype=self.dtype)
            self.v_csa = torch.zeros_like(self.k_csa)
            self.k_hca = torch.zeros_like(self.k_csa)
            self.v_hca = torch.zeros_like(self.k_csa)

        end = self._seq_len + new_len
        target = slice(self._seq_len, end)
        self.k_csa[:, target] = k_csa
        self.v_csa[:, target] = v_csa
        self.k_hca[:, target] = k_hca
        self.v_hca[:, target] = v_hca
        self._seq_len = end

        split = max(0, end - self.local_window_size)
        k = torch.cat((self.k_hca[:, :split], self.k_csa[:, split:end]), dim=1)
        v = torch.cat((self.v_hca[:, :split], self.v_csa[:, split:end]), dim=1)
        return k, v

    def reset(self) -> None:
        self.k_csa.zero_()
        self.v_csa.zero_()
        self.k_hca.zero_()
        self.v_hca.zero_()
        self._seq_len = 0

    @property
    def seq_len(self) -> int:
        return self._seq_len


AttentionCache = KVCache | MLAKVCache | HybridKVCache


def build_attention_cache(config, device: torch.device, dtype: torch.dtype | None = None) -> AttentionCache:
    """Build the cache representation required by the configured attention mode."""
    if config.use_mla:
        return MLAKVCache(
            config.context_length,
            config.n_heads,
            config.mla_kv_c_dim,
            config.d_head_rope,
            device,
            dtype,
        )
    if config.use_hybrid_attention:
        return HybridKVCache(
            config.context_length,
            config.n_heads,
            config.d_head,
            config.local_window_size,
            device,
            dtype,
        )
    return KVCache(config.context_length, config.n_heads, config.d_head, device, dtype)
