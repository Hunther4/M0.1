"""Transformer Language Model with Weight-Tied Embeddings.

Assembles TokenEmbedding, a stack of TransformerBlock layers, and RMSNorm
into a complete decoder-only language model. The output projection shares
weights with the embedding layer (weight tying), saving ~21M parameters
and providing a single gradient signal through the embedding matrix.
"""

from typing import List, Optional

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.model.block import TransformerBlock
from src.model.rms_norm import RMSNorm
from src.transformer.config import M01Config
from src.transformer.embeddings import TokenEmbedding
from src.transformer.kv_cache import KVCache


class TransformerLM(nn.Module):
    """Decoder-only Transformer Language Model.

    Architecture:
        TokenEmbedding → [TransformerBlock × n_layers] → RMSNorm → output_head

    The output head is weight-tied with the embedding layer — no separate
    output projection parameters exist. Accepts optional KV caches for
    efficient autoregressive generation.

    Args:
        config: M01Config with vocab_size, d_model, n_layers, etc.
    """

    def __init__(self, config: M01Config) -> None:
        super().__init__()

        self.config = config
        self.embedding = TokenEmbedding(config)
        self.blocks = nn.ModuleList([
            TransformerBlock(config, force_dense=(i < config.num_dense_layers)) 
            for i in range(config.n_layers)
        ])
        self.norm = RMSNorm(config.d_model)

        # Weight-tied output head: references embedding.weight
        # TokenEmbedding.output_head is a method, not a separate Module
        self.output_head = self.embedding.output_head

    def forward(
        self,
        token_ids: Tensor,
        kv_caches: Optional[List[Optional[KVCache]]] = None,
    ) -> Tensor:
        """Forward pass through the full transformer stack.

        Args:
            token_ids: Token IDs of shape (batch, seq_len)
            kv_caches: Optional list of KV caches, one per layer.
                       Must have length equal to n_layers if provided.
                       Individual entries may be None for layers without cache.

        Returns:
            Logits of shape (batch, seq_len, vocab_size)
        """
        # Embed tokens: (batch, seq_len) -> (batch, seq_len, d_model)
        x = self.embedding(token_ids)

        # Pass through each transformer block
        for i, block in enumerate(self.blocks):
            cache = kv_caches[i] if kv_caches is not None else None
            x = block(x, kv_cache=cache)

        # Final normalization
        x = self.norm(x)

        # Weight-tied output projection: (batch, seq_len, vocab_size)
        return self.output_head(x)

    def get_aux_loss(self) -> Tensor:
        """Collect and sum the auxiliary load balancing losses from all child MoELayers."""
        import torch
        aux_loss = torch.tensor(0.0, device=self.norm.gamma.device)
        for block in self.blocks:
            if hasattr(block.ff, "get_aux_loss"):
                aux_loss += block.ff.get_aux_loss()
        return aux_loss

    def get_z_loss(self) -> Tensor:
        """Collect and sum Router Z-Loss from all child MoELayers."""
        import torch

        z_loss = torch.tensor(0.0, device=self.norm.gamma.device)
        count = 0
        for block in self.blocks:
            if hasattr(block.ff, "get_z_loss"):
                z_loss += block.ff.get_z_loss()
                count += 1
        return z_loss / count if count > 0 else z_loss

    def get_moe_metrics(self) -> dict:
        """Compute and return MoE routing metrics for this model.

        Delegates to :func:`src.training.moe_metrics.compute_moe_metrics`.
        Returns an empty dict when no MoE layers exist or none have been
        forwarded in training mode yet.

        Returns:
            dict with per-layer and global metric keys, or {} if no MoE layers.
        """
        from src.training.moe_metrics import compute_moe_metrics
        return compute_moe_metrics(self)
