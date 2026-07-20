"""TrainerState: Single source of truth for the M0.1 TrainingEngine."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time
import torch


@dataclass
class TrainerState:
    """Global state of the training session.

    All callbacks, metrics, loggers, and engines read and write to this
    single object.
    """

    step: int = 0
    epoch: int = 0
    global_tokens: int = 0
    loss: float = 0.0
    ce_loss: float = 0.0
    aux_loss: float = 0.0
    lr: float = 0.0
    grad_norm: float = 0.0

    # MoE Metrics (expandidos)
    moe_enabled: bool = False
    router_entropy: float = 0.0
    moe_gini: float = 0.0
    dead_experts: int = 0
    router_collapse: bool = False
    consecutive_collapse_steps: int = 0
    moe_metrics: dict = field(default_factory=dict)

    # Throughput & Hardware
    tokens_per_sec: float = 0.0
    samples_per_sec: float = 0.0
    vram_allocated_mb: float = 0.0
    vram_peak_mb: float = 0.0
    scaler_scale: float = 1.0

    # Timings (ms per component)
    forward_ms: float = 0.0
    backward_ms: float = 0.0
    optimizer_ms: float = 0.0
    start_time: float = field(default_factory=time.time)
    elapsed_seconds: float = 0.0
    eta_hours: float = 0.0

    # Validation
    val_loss: float = float("inf")
    val_ppl: float = float("inf")
    best_val_loss: float = float("inf")

    # Status Flags
    is_interrupted: bool = False
    should_stop: bool = False
    stop_reason: str = ""
    is_done: bool = False

    # RNG states dictionary for robust deterministic resume
    rng_states: Dict[str, Any] = field(default_factory=dict)

    def capture_rng_states(self) -> None:
        """Capture Python, NumPy, PyTorch CPU, and CUDA/ROCm RNG states."""
        import random
        import numpy as np

        self.rng_states = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }

    def restore_rng_states(self) -> None:
        """Restore Python, NumPy, PyTorch CPU, and CUDA/ROCm RNG states."""
        import random
        import numpy as np

        if not self.rng_states:
            return

        if "python" in self.rng_states:
            random.setstate(self.rng_states["python"])
        if "numpy" in self.rng_states:
            np.random.set_state(self.rng_states["numpy"])
        if "torch_cpu" in self.rng_states:
            torch.set_rng_state(self.rng_states["torch_cpu"])
        if "torch_cuda" in self.rng_states and self.rng_states["torch_cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(self.rng_states["torch_cuda"])
