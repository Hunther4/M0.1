"""Causal Self-Attention with Multi-head Latent Attention (MLA), Hybrid Attention, and standard MHA.

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
    """Causal self-attention supporting MLA, Hybrid Attention, and MHA.
    
    Implements Q/K/V projections, rotary positional embeddings (RoPE),
    causal masking, and optional KV cache.
    """
    
    def __init__(self, config: M01Config) -> None:
        """Initialize attention module.
        
        Args:
            config: M01Config configurations
        """
        super().__init__()
        
        self.n_heads = config.n_heads
        self.d_head = config.d_head
        self.d_model = config.d_model
        
        self.use_mla = config.use_mla
        self.use_hybrid_attention = config.use_hybrid_attention
        self.local_window_size = config.local_window_size
        
        if self.use_mla:
            # MLA Latent and RoPE dimensions
            self.d_head_rope = config.d_head_rope
            self.d_head_no_rope = config.d_head_no_rope
            
            # Query projections
            self.W_q = nn.Linear(config.d_model, config.n_heads * self.d_head_no_rope, bias=False)
            self.W_qr = nn.Linear(config.d_model, config.n_heads * self.d_head_rope, bias=False)
            
            # Key-Value compression and latent norm
            self.W_kv_down = nn.Linear(config.d_model, config.mla_kv_c_dim, bias=False)
            from src.model.rms_norm import RMSNorm
            self.norm_kv = RMSNorm(config.mla_kv_c_dim)
            
            # Key-Value up-projections
            self.W_k_up = nn.Linear(config.mla_kv_c_dim, config.n_heads * self.d_head_no_rope, bias=False)
            self.W_v_up = nn.Linear(config.mla_kv_c_dim, config.n_heads * config.d_head, bias=False)
            
            # Positional Key projection (RoPE part)
            self.W_kr = nn.Linear(config.d_model, config.n_heads * self.d_head_rope, bias=False)
            
        elif self.use_hybrid_attention:
            # Compressed Sparse Attention (CSA) - moderate compression
            self.W_q = nn.Linear(config.d_model, config.d_model, bias=False)
            self.W_kv_csa = nn.Linear(config.d_model, config.csa_kv_dim, bias=False)
            self.W_k_csa_up = nn.Linear(config.csa_kv_dim, config.d_model, bias=False)
            self.W_v_csa_up = nn.Linear(config.csa_kv_dim, config.d_model, bias=False)
            
            # Heavily Compressed Attention (HCA) - high compression
            self.W_kv_hca = nn.Linear(config.d_model, config.hca_kv_dim, bias=False)
            self.W_k_hca_up = nn.Linear(config.hca_kv_dim, config.d_model, bias=False)
            self.W_v_hca_up = nn.Linear(config.hca_kv_dim, config.d_model, bias=False)
        else:
            # Standard projections (MHA)
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
        offset = 0 if kv_cache is None else kv_cache.seq_len
        
        if self.use_mla:
            # 1. Project Query
            q_c = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_head_no_rope)
            q_r = self.W_qr(x).view(batch_size, seq_len, self.n_heads, self.d_head_rope)
            
            # Apply RoPE to Query RoPE part
            q_r = self.rope(q_r, offset=offset)
            
            # Concatenate to retrieve full Query tensor
            q = torch.cat([q_c, q_r], dim=-1)
            
            # 2. Compress and project Key-Value
            kv_latent = self.W_kv_down(x)
            kv_latent = self.norm_kv(kv_latent)
            
            k_c = self.W_k_up(kv_latent).view(batch_size, seq_len, self.n_heads, self.d_head_no_rope)
            v = self.W_v_up(kv_latent).view(batch_size, seq_len, self.n_heads, self.d_head)
            
            # Project positional Key RoPE part
            k_r = self.W_kr(x).view(batch_size, seq_len, self.n_heads, self.d_head_rope)
            k_r = self.rope(k_r, offset=offset)
            
            # Concatenate to retrieve full Key tensor
            k = torch.cat([k_c, k_r], dim=-1)
            
        elif self.use_hybrid_attention:
            q = self.W_q(x)
            # Compressed projections
            csa_lat = self.W_kv_csa(x)
            k_csa = self.W_k_csa_up(csa_lat)
            v_csa = self.W_v_csa_up(csa_lat)
            
            if kv_cache is not None:
                k = k_csa
                v = v_csa
            elif seq_len > self.local_window_size:
                hca_lat = self.W_kv_hca(x)
                k_hca = self.W_k_hca_up(hca_lat)
                v_hca = self.W_v_hca_up(hca_lat)
                split_idx = seq_len - self.local_window_size
                k = torch.cat([k_hca[:, :split_idx], k_csa[:, split_idx:]], dim=1)
                v = torch.cat([v_hca[:, :split_idx], v_csa[:, split_idx:]], dim=1)
            else:
                k = k_csa
                v = v_csa
                
            q = q.view(batch_size, seq_len, self.n_heads, self.d_head)
            k = k.view(batch_size, seq_len, self.n_heads, self.d_head)
            v = v.view(batch_size, seq_len, self.n_heads, self.d_head)
            
            q = self.rope(q, offset=offset)
            k = self.rope(k, offset=offset)
        else:
            # Standard Multi-head attention
            q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_head)
            k = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_head)
            v = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_head)
            
            q = self.rope(q, offset=offset)
            k = self.rope(k, offset=offset)
        
        # Handle KV cache appending
        if kv_cache is not None:
            k, v = kv_cache.append(k, v)
        
        # Transpose for attention computation: (batch, seq, n_heads, d_head) -> (batch, n_heads, seq, d_head)
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
                mask = k_pos <= q_pos
                attn_mask = mask.unsqueeze(0).unsqueeze(0)  # Shape (1, 1, q_len, kv_len)
                attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
            else:
                # Autoregressive decoding (q_len == 1), no masking needed
                attn_output = F.scaled_dot_product_attention(q, k, v)

        # Transpose back: (batch, n_heads, q_len, d_head) -> (batch, q_len, n_heads, d_head)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(batch_size, seq_len, self.d_model)
        
        # Output projection
        output = self.W_o(attn_output)
        
        return output