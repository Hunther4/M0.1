"""Causal Self-Attention with Multi-head Latent Attention (MLA), Hybrid Attention, and standard MHA.

Supports KV cache for autoregressive generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
from .config import M01Config
from .rope import RotaryPositionalEmbedding
from .attention_backend import AttentionDispatcher
from .kv_cache import HybridKVCache, KVCache, MLAKVCache


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
        self.attention_dispatcher = AttentionDispatcher(config.attention_backend)
        
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
        nn.init.normal_(
            self.W_o.weight,
            mean=0.0,
            std=config.residual_init_std,
        )
        
        # Rotary positional embedding
        self.rope = RotaryPositionalEmbedding(config)
        
        # Scale factor for attention scores
        self.scale = 1.0 / math.sqrt(config.d_head)
    
    def forward(
        self, 
        x: torch.Tensor, 
        kv_cache: Optional[KVCache | MLAKVCache | HybridKVCache] = None
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

            if isinstance(kv_cache, MLAKVCache):
                all_latent, all_k_r = kv_cache.append(kv_latent, k_r)
                return self._forward_mla_cached(q_c, q_r, all_latent, all_k_r, offset)
            
            # Concatenate to retrieve full Key tensor
            k = torch.cat([k_c, k_r], dim=-1)
            
        elif self.use_hybrid_attention:
            q = self.W_q(x)
            # Compressed projections
            csa_lat = self.W_kv_csa(x)
            k_csa = self.W_k_csa_up(csa_lat)
            v_csa = self.W_v_csa_up(csa_lat)

            hca_lat = self.W_kv_hca(x)
            k_hca = self.W_k_hca_up(hca_lat)
            v_hca = self.W_v_hca_up(hca_lat)

            q = q.view(batch_size, seq_len, self.n_heads, self.d_head)
            k_csa = k_csa.view(batch_size, seq_len, self.n_heads, self.d_head)
            v_csa = v_csa.view(batch_size, seq_len, self.n_heads, self.d_head)
            k_hca = k_hca.view(batch_size, seq_len, self.n_heads, self.d_head)
            v_hca = v_hca.view(batch_size, seq_len, self.n_heads, self.d_head)

            q = self.rope(q, offset=offset)
            k_csa = self.rope(k_csa, offset=offset)
            k_hca = self.rope(k_hca, offset=offset)

            if isinstance(kv_cache, HybridKVCache):
                k, v = kv_cache.append(k_csa, v_csa, k_hca, v_hca)
            elif kv_cache is not None:
                # Compatibility for callers that still construct the generic cache.
                k, v = kv_cache.append(k_csa, v_csa)
            elif seq_len > self.local_window_size:
                split_idx = seq_len - self.local_window_size
                k = torch.cat([k_hca[:, :split_idx], k_csa[:, split_idx:]], dim=1)
                v = torch.cat([v_hca[:, :split_idx], v_csa[:, split_idx:]], dim=1)
            else:
                k = k_csa
                v = v_csa
        else:
            # Standard Multi-head attention
            q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_head)
            k = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_head)
            v = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_head)
            
            q = self.rope(q, offset=offset)
            k = self.rope(k, offset=offset)
        
        # Handle KV cache appending
        if kv_cache is not None and not (self.use_hybrid_attention and not self.use_mla):
            k, v = kv_cache.append(k, v)
        
        # Transpose for attention computation: (batch, seq, n_heads, d_head) -> (batch, n_heads, seq, d_head)
        q_len = seq_len
        kv_len = k.shape[1]

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if kv_cache is None:
            # Standard causal self-attention
            attn_output = self.attention_dispatcher.dispatch(q, k, v, is_causal=True)
        else:
            if q_len > 1:
                # Causal masking for parallel prefill in KV Cache
                q_pos = torch.arange(offset, offset + q_len, device=x.device).unsqueeze(1)
                k_pos = torch.arange(kv_len, device=x.device).unsqueeze(0)
                mask = k_pos <= q_pos
                attn_mask = mask.unsqueeze(0).unsqueeze(0)  # Shape (1, 1, q_len, kv_len)
                attn_output = self.attention_dispatcher.dispatch(q, k, v, attn_mask=attn_mask)
            else:
                # Autoregressive decoding (q_len == 1), no masking needed
                attn_output = self.attention_dispatcher.dispatch(q, k, v)

        # Transpose back: (batch, n_heads, q_len, d_head) -> (batch, q_len, n_heads, d_head)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(batch_size, seq_len, self.d_model)
        
        # Output projection
        output = self.W_o(attn_output)
        
        return output

    def _forward_mla_cached(
        self,
        q_c: torch.Tensor,
        q_r: torch.Tensor,
        latent: torch.Tensor,
        k_r: torch.Tensor,
        offset: int,
    ) -> torch.Tensor:
        """Run MLA from compressed history without materializing historical K/V."""
        q_len = q_c.shape[1]
        kv_len = latent.shape[1]

        # Cast to FP32 for safe attention score accumulation (FP16 overflows easily
        # in the einsum before self.scale is applied).
        orig_dtype = q_c.dtype
        q_c = q_c.float()
        q_r = q_r.float()
        latent = latent.float()
        k_r = k_r.float()
        k_up = self.W_k_up.weight.float().view(
            self.n_heads, self.d_head_no_rope, latent.shape[-1]
        )

        content_scores = torch.einsum("bqhn,hnc,blc->bhql", q_c, k_up, latent)
        rope_scores = torch.einsum("bqhr,blhr->bhql", q_r, k_r)
        scores = (content_scores + rope_scores) * self.scale

        if q_len > 1:
            q_pos = torch.arange(offset, offset + q_len, device=scores.device).unsqueeze(1)
            k_pos = torch.arange(kv_len, device=scores.device).unsqueeze(0)
            scores = scores.masked_fill(~(k_pos <= q_pos).unsqueeze(0).unsqueeze(0), -torch.inf)

        weights = torch.softmax(scores, dim=-1)
        attended_latent = torch.einsum("bhql,blc->bqhc", weights, latent)
        v_up = self.W_v_up.weight.float().view(self.n_heads, self.d_head, latent.shape[-1])
        output = torch.einsum("bqhc,hdc->bqhd", attended_latent, v_up)
        return self.W_o(output.reshape(q_c.shape[0], q_len, self.d_model).to(orig_dtype))
