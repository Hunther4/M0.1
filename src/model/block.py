"""Transformer Block with Pre-Norm and Conditional MoE.

Pre-norm transformer block: norm → sublayer → residual.
Supports both dense FeedForward and Mixture of Experts (MoE) based on config.
KV Cache passthrough for autoregressive generation.
"""

from typing import Optional
import torch.nn as nn
from torch import Tensor
from src.transformer.config import M01Config
from src.transformer.attention import CausalSelfAttention
from src.transformer.feedforward import FeedForward
from src.transformer.moe import MoELayer
from src.transformer.kv_cache import KVCache
from src.model.rms_norm import RMSNorm


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block with conditional FF/MoE.

    Applies pre-norm self-attention followed by feedforward (or MoE)
    with residual connections around each sublayer.

    Args:
        config: M01Config with d_model, n_heads, d_ff, num_experts
    """

    def __init__(self, config: M01Config) -> None:
        super().__init__()

        # Pre-norm self-attention
        self.norm1 = RMSNorm(config.d_model)
        self.attn = CausalSelfAttention(config)

        # Pre-norm feedforward (dense or MoE)
        self.norm2 = RMSNorm(config.d_model)

        # Dropout applied after attention and FF sublayers
        self.dropout = nn.Dropout(config.dropout)
        if config.num_experts > 1:
            self.ff = MoELayer(config)
        else:
            self.ff = FeedForward(config)

    def forward(
        self,
        x: Tensor,
        kv_cache: Optional[KVCache] = None,
    ) -> Tensor:
        """Forward pass with pre-norm residual connections.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            kv_cache: Optional KV cache for autoregressive generation

        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        # Pre-norm self-attention with residual + dropout
        x = x + self.dropout(self.attn(self.norm1(x), kv_cache=kv_cache))

        # Pre-norm feedforward with residual + dropout
        x = x + self.dropout(self.ff(self.norm2(x)))

        return x
