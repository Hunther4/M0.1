"""M0.1 Transformer Configuration.

This module defines the configuration dataclass for the M0.1 transformer model.
All hyperparameters are set with defaults matching docs/architecture.md.
"""

from dataclasses import dataclass

@dataclass
class M01Config:
    """Configuration for the M0.1 decoder-only transformer.
    
    Attributes:
        vocab_size: Vocabulary size (default: 32768)
        context_length: Maximum sequence length (default: 8192)
        d_model: Embedding dimension (default: 640)
        n_heads: Number of attention heads (default: 10)
        d_ff: Feedforward hidden dimension (default: 1728)
        n_layers: Number of transformer layers (default: 12)
        rope_theta: RoPE theta parameter (default: 10000.0)
        num_experts: Number of experts for MoE (default: 1 for dense)
        dropout: Dropout rate (default: 0.0)
    """
    vocab_size: int = 32768
    context_length: int = 8192
    d_model: int = 640
    n_heads: int = 10
    d_ff: int = 1728
    n_layers: int = 12
    rope_theta: float = 10000.0
    num_experts: int = 1
    dropout: float = 0.0

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        # Ensure d_model is divisible by n_heads
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        
        # Compute head dimension
        self.d_head = self.d_model // self.n_heads  # 64 for default config