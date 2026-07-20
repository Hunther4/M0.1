"""Training Configuration.

This module defines the TrainingConfig dataclass containing all
hyperparameters and paths for the M0.1 training loop.
"""

from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Configuration for the M0.1 training loop.

    Attributes:
        batch_size: Number of sequences per batch
        seq_len: Maximum sequence length per sample
        max_lr: Peak learning rate
        min_lr_ratio: Minimum LR as fraction of max_lr (cosine decay floor)
        warmup_steps: Number of linear warmup steps
        max_steps: Total training steps
        weight_decay: AdamW weight decay coefficient
        beta1: Adam beta1
        beta2: Adam beta2
        max_norm: Gradient clipping max norm
        log_interval: Steps between console logging
        save_interval: Steps between checkpoint saves
        checkpoint_dir: Directory for model checkpoints
        data_dir: Directory containing training data
    """
    batch_size: int = 4
    seq_len: int = 1024
    max_lr: float = 3e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 200
    max_steps: int = 100_000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    max_norm: float = 1.0
    log_interval: int = 10
    save_interval: int = 500
    checkpoint_dir: str = "checkpoints"
    data_dir: str = "data"

    # MoE Runtime Monitoring
    log_moe_metrics: bool = True
    moe_collapse_consecutive_steps: int = 50
    moe_collapse_expert_ratio: float = 0.3
    log_metrics_backend: str = "console"
    collapse_streak_threshold: int = 500
