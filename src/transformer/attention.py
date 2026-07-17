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
        
        # Q/K/V projections (no bias)
        self.W_q = nn.Linear(config.d_model, config.d_model, bias=False)
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
        
        # Project to Q, K, V: (batch, seq, d_model) → (batch, seq, d_model)
        q = self.W_q(x)
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
        
        # Compute attention scores: Q @ K^T / sqrt(d_head)
        # (batch, q_len, n_heads, d_head) @ (batch, kv_len, n_heads, d_head)^T
        # → (batch, n_heads, q_len, kv_len)
        q_len = seq_len
        kv_len = k.shape[1]
        
        # Transpose for batch matrix multiplication
        # (batch, seq, n_heads, d_head) → (batch, n_heads, seq, d_head)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Attention scores: (batch, n_heads, q_len, kv_len)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply causal mask
        # For self-attention without cache: q_len == kv_len
        # For cached attention: q_len == 1, kv_len == past + 1
        if kv_cache is None:
            # Standard causal mask for self-attention
            mask = torch.triu(torch.ones(q_len, kv_len, device=x.device), diagonal=1)
            mask = mask.masked_fill(mask == 1, float('-inf'))
            attn_scores = attn_scores + mask.unsqueeze(0).unsqueeze(0)
        else:
            # For cached generation, only need to mask future positions
            # Since we're generating one token at a time, no masking needed
            # (each step only attends to past + current token)
            pass
        
        # Softmax over last dimension
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Weighted sum: (batch, n_heads, q_len, kv_len) @ (batch, n_heads, kv_len, d_head)
        # → (batch, n_heads, q_len, d_head)
        attn_output = torch.matmul(attn_weights, v)
        
        # Transpose back: (batch, n_heads, q_len, d_head) → (batch, q_len, n_heads, d_head)
        attn_output = attn_output.transpose(1, 2)
        
        # Reshape to (batch, seq_len, d_model)
        attn_output = attn_output.reshape(batch_size, seq_len, self.d_model)
        
        # Output projection
        output = self.W_o(attn_output)
        
        return output