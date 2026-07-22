"""M0.1 Transformer Configuration.

This module defines the configuration dataclass for the M0.1 transformer model.
All hyperparameters are set with defaults matching docs/architecture.md.
"""

import math
from dataclasses import dataclass

@dataclass
class M01Config:
    """Configuration for the M0.1 decoder-only transformer.
    
    Attributes:
        vocab_size: Vocabulary size (default: 16384)
        context_length: Maximum sequence length (default: 8192)
        d_model: Embedding dimension (default: 640)
        n_heads: Number of attention heads (default: 10)
        d_ff: Feedforward hidden dimension (default: 1728)
        n_layers: Number of transformer layers (default: 12)
        rope_theta: RoPE theta parameter (default: 10000.0)
        num_experts: Number of experts for MoE (default: 1 for dense)
        num_shared_experts: Number of shared experts for DeepSeek-style MoE (default: 1)
        moe_top_k: Number of active routed experts per token (default: 2)
        capacity_factor: Routed-expert capacity headroom over ideal balance (default: 1.25)
        capacity_factor_warmup_steps: Steps to anneal capacity from the warmup start
            to the target capacity factor (default: 2000)
        capacity_factor_warmup_start: Initial routed-expert capacity factor during
            warmup (default: 2.0)
        dropout: Dropout rate (default: 0.0)
    """
    vocab_size: int = 16384
    context_length: int = 8192
    d_model: int = 640
    n_heads: int = 10
    d_ff: int = 1728
    n_layers: int = 12
    rope_theta: float = 10000.0
    # Current MoE: 4 routed + 1 shared + top-2
    # d_ff_shared=448, d_ff_routed=784 → ratio FLOPs routed:shared = 3.5×
    num_experts: int = 4
    num_shared_experts: int = 1
    moe_top_k: int = 2
    capacity_factor: float = 1.25
    capacity_factor_warmup_steps: int = 2000
    capacity_factor_warmup_start: float = 2.0
    d_ff_shared: int | None = 448
    d_ff_routed: int | None = 784
    use_hybrid_attention: bool = False
    csa_kv_dim: int = 128
    hca_kv_dim: int = 32
    local_window_size: int = 64
    dropout: float = 0.0
    attention_backend: str = "auto"
    initializer_range: float = 0.02
    scale_residual_projections: bool = True

    # MLA (Multi-head Latent Attention) Configuration
    use_mla: bool = True
    mla_kv_c_dim: int = 128
    mla_rope_dim: int = 16  # RoPE dimension per head

    # Dense layers layout configuration (number of dense layers before MoE starts)
    num_dense_layers: int = 2

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.n_layers <= 0:
            raise ValueError("n_layers must be greater than zero")
        if self.initializer_range <= 0:
            raise ValueError("initializer_range must be greater than zero")
        # Ensure d_model is divisible by n_heads
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if self.attention_backend not in {"auto", "flash", "efficient", "math"}:
            raise ValueError(
                "attention_backend must be one of: auto, flash, efficient, math"
            )
        
        # Compute head dimension
        self.d_head = self.d_model // self.n_heads  # 64 for default config
        
        if self.use_mla:
            # Dynamically clamp mla_rope_dim to be an even number strictly less than d_head
            raw_rope = min(self.mla_rope_dim, max(2, self.d_head - 2))
            self.d_head_rope = raw_rope if raw_rope % 2 == 0 else raw_rope - 1
            self.d_head_no_rope = self.d_head - self.d_head_rope

    @property
    def residual_init_std(self) -> float:
        """Initialization std for projections entering the residual stream."""
        if not self.scale_residual_projections:
            return self.initializer_range
        return self.initializer_range / math.sqrt(2.0 * self.n_layers)
