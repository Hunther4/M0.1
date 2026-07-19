"""Causal Self-Attention with RoPE and KV Cache.

Multi-head causal self-attention with rotary positional embeddings.
Supports KV cache for autoregressive generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from .config import M01Config
from .rope import RotaryPositionalEmbedding
from .kv_cache import KVCache


class CausalSelfAttention(nn.Module):
    """Causal multi-head self-attention with RoPE.
    
    Implements Q/K/V projections, rotary positional embeddings,
    causal masking, and optional KV cache for efficient generation.
    """
    
    def __init__(self, config: M01Config) -> None:
        """Initialize attention module.
        
        Args:
            config: M01Config with d_model, n_heads, d_head
        """
        super().__init__()
        
        self.n_heads = config.n_heads
        self.d_head = config.d_head
        self.d_model = config.d_model
        
        self.use_hybrid_attention = config.use_hybrid_attention
        self.local_window_size = config.local_window_size
        
        # Q projection (no bias)
        self.W_q = nn.Linear(config.d_model, config.d_model, bias=False)
        
        if self.use_hybrid_attention:
            # Compressed Sparse Attention (CSA) - moderate compression
            self.W_kv_csa = nn.Linear(config.d_model, config.csa_kv_dim, bias=False)
            self.W_k_csa_up = nn.Linear(config.csa_kv_dim, config.d_model, bias=False)
            self.W_v_csa_up = nn.Linear(config.csa_kv_dim, config.d_model, bias=False)
            
            # Heavily Compressed Attention (HCA) - high compression
            self.W_kv_hca = nn.Linear(config.d_model, config.hca_kv_dim, bias=False)
            self.W_k_hca_up = nn.Linear(config.hca_kv_dim, config.d_model, bias=False)
            self.W_v_hca_up = nn.Linear(config.hca_kv_dim, config.d_model, bias=False)
        else:
            # Standard projections
            self.W_k = nn.Linear(config.d_model, config.d_model, bias=False)
            self.W_v = nn.Linear(config.d_model, config.d_model, bias=False)
        
        # Output projection
        self.W_o = nn.Linear(config.d_model, config.d_model, bias=False)
        
        # Rotary positional embedding
        self.rope = RotaryPositionalEmbedding(config)
        
        # Scale factor for attention scores
        self.scale = 1.0 / math.sqrt(config.d_head)
    
    def forward(
        self, 
        x: torch.Tensor, 
        kv_cache: Optional[KVCache] = None
    ) -> torch.Tensor:
        """Forward pass of causal self-attention.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            kv_cache: Optional KVCache for autoregressive generation
            
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.W_q(x)
        
        if self.use_hybrid_attention:
            # Compressed projections
            csa_lat = self.W_kv_csa(x)
            k_csa = self.W_k_csa_up(csa_lat)
            v_csa = self.W_v_csa_up(csa_lat)
            
            if kv_cache is not None:
                # During step-by-step decoding, the current token is always local (CSA)
                k = k_csa
                v = v_csa
            elif seq_len > self.local_window_size:
                # Hybrid: HCA for history, CSA for local window
                hca_lat = self.W_kv_hca(x)
                k_hca = self.W_k_hca_up(hca_lat)
                v_hca = self.W_v_hca_up(hca_lat)
                
                split_idx = seq_len - self.local_window_size
                k = torch.cat([k_hca[:, :split_idx], k_csa[:, split_idx:]], dim=1)
                v = torch.cat([v_hca[:, :split_idx], v_csa[:, split_idx:]], dim=1)
            else:
                k = k_csa
                v = v_csa
        else:
            k = self.W_k(x)
            v = self.W_v(x)
        
        # Reshape to (batch, seq, n_heads, d_head)
        q = q.view(batch_size, seq_len, self.n_heads, self.d_head)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_head)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_head)
        
        # Apply RoPE to Q and K
        offset = 0 if kv_cache is None else kv_cache.seq_len
        q = self.rope(q, offset=offset)
        k = self.rope(k, offset=offset)
        
        # Handle KV cache
        if kv_cache is not None:
            # Append to cache and get full K, V
            k, v = kv_cache.append(k, v)
        
        # Compute attention scores and output using PyTorch's native SDPA (Scaled Dot Product Attention)
        # This will automatically dispatch to FlashAttention or Memory Efficient Attention on GPU.
        q_len = seq_len
        kv_len = k.shape[1]

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if kv_cache is None:
            # Standard causal self-attention
            attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            if q_len > 1:
                # Causal masking for parallel prefill in KV Cache
                q_pos = torch.arange(offset, offset + q_len, device=x.device).unsqueeze(1)
                k_pos = torch.arange(kv_len, device=x.device).unsqueeze(0)
                mask = k_pos <= q_pos  # True means keep attention, False means mask out
                attn_mask = mask.unsqueeze(0).unsqueeze(0)  # Broadcast to (1, 1, q_len, kv_len)
                attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
            else:
                # Autoregressive decoding (q_len == 1), no masking needed
                attn_output = F.scaled_dot_product_attention(q, k, v)

        # Transpose back: (batch, n_heads, q_len, d_head) → (batch, q_len, n_heads, d_head)
        attn_output = attn_output.transpose(1, 2)
        
        # Reshape to (batch, seq_len, d_model)
        attn_output = attn_output.reshape(batch_size, seq_len, self.d_model)
        
        # Output projection
        output = self.W_o(attn_output)
        
        return output