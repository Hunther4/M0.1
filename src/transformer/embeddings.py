"""Token Embedding with Tied Weights.

This module provides token embedding with weight tying between input and output.
Weight tying saves ~10.5M parameters and ensures single gradient flow.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import M01Config


class TokenEmbedding(nn.Module):
    """Token embedding layer with tied output head.
    
    The embedding matrix is shared between input embedding and output projection.
    This saves parameters and ensures gradient flow through the entire vocabulary.
    """
    
    def __init__(self, config: M01Config) -> None:
        """Initialize token embedding.
        
        Args:
            config: M01Config with vocab_size and d_model
        """
        super().__init__()
        
        # Embedding matrix: vocab_size × d_model
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        # Scaling factor (default to 1.0 to match standard LLaMA embedding dynamics)
        self.scale = 1.0
        
        # Initialize weights (optional, can use default)
        nn.init.normal_(
            self.embedding.weight,
            mean=0.0,
            std=config.initializer_range,
        )
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Embed token IDs and scale.
        
        Args:
            token_ids: (batch, seq_len) integer tensor
            
        Returns:
            (batch, seq_len, d_model) scaled embeddings
        """
        # (batch, seq_len) → (batch, seq_len, d_model)
        embeddings = self.embedding(token_ids)
        
        # Scale by 1/√d_model
        return embeddings * self.scale
    
    def output_head(self, hidden: torch.Tensor) -> torch.Tensor:
        """Project hidden states to vocabulary logits using tied weights.
        
        Args:
            hidden: (batch, seq_len, d_model) hidden states
            
        Returns:
            (batch, seq_len, vocab_size) logits
        """
        # Apply the same scale on both uses of the tied matrix. The default is
        # 1.0, so this is backward-compatible while keeping future scales
        # symmetric between input embeddings and output projection.
        return F.linear(hidden, self.embedding.weight * self.scale, None)
