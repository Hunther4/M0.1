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
        num_shared_experts: Number of shared experts for DeepSeek-style MoE (default: 1)
        moe_top_k: Number of active routed experts per token (default: 1)
        dropout: Dropout rate (default: 0.0)
    """
    vocab_size: int = 32768
    context_length: int = 8192
    d_model: int = 640
    n_heads: int = 10
    d_ff: int = 1728
    n_layers: int = 12
    rope_theta: float = 10000.0
    # Stage 1: MoE activo con 4 routed + 1 shared + top-1
    num_experts: int = 4
    num_shared_experts: int = 1
    moe_top_k: int = 1
    d_ff_shared: int | None = 1024
    d_ff_routed: int | None = 640
    use_hybrid_attention: bool = False
    csa_kv_dim: int = 128
    hca_kv_dim: int = 32
    local_window_size: int = 64
    dropout: float = 0.0

    # MLA (Multi-head Latent Attention) Configuration
    use_mla: bool = True
    mla_kv_c_dim: int = 128
    mla_rope_dim: int = 16  # RoPE dimension per head

    # Dense layers layout configuration (number of dense layers before MoE starts)
    num_dense_layers: int = 2

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        # Ensure d_model is divisible by n_heads
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        
        # Compute head dimension
        self.d_head = self.d_model // self.n_heads  # 64 for default config
        
        if self.use_mla:
            # Dynamically clamp mla_rope_dim to be an even number strictly less than d_head
            raw_rope = min(self.mla_rope_dim, max(2, self.d_head - 2))
            self.d_head_rope = raw_rope if raw_rope % 2 == 0 else raw_rope - 1
            self.d_head_no_rope = self.d_head - self.d_head_rope