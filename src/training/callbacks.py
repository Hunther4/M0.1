"""Callbacks con solo 3 hooks.

Cada callback implementa los hooks que necesita y decide
cuándo actuar según TrainerState.
"""

import math
from typing import Any

import torch
import torch.nn as nn

from .state import TrainerState
from .moe_metrics import compute_moe_metrics

try:
    from .run_manager import RunManager
except ImportError:
    RunManager = None  # type: ignore


class Callback:
    """Interface mínima. Solo 3 hooks. Cada callback decide cuándo actuar."""

    def on_step_end(self, state: TrainerState) -> None:
        pass

    def on_validation(self, state: TrainerState) -> None:
        pass

    def on_save(self, state: TrainerState) -> None:
        pass


# ── Logger ────────────────────────────────────────────────────────────────────

class LoggerCallback(Callback):
    """Logging a consola cada log_interval steps."""

    def __init__(self, log_interval: int = 10) -> None:
        self.log_interval = log_interval

    def on_step_end(self, state: TrainerState) -> None:
        if (state.step + 1) % self.log_interval != 0:
            return

        parts = [
            f"step {state.step + 1:>6d}",
            f"loss {state.loss:.4f}",
            f"aux {state.aux_loss:.4f}",
            f"lr {state.lr:.2e}",
            f"ppl {state.val_ppl:.2f}" if state.val_ppl < float("inf") else "",
            f"norm {state.grad_norm:.1f}",
            f"tok/s {state.tokens_per_sec:.0f}",
            f"VRAM {state.vram_allocated_mb:.0f}MB",
        ]
        # MoE metrics compactas
        if state.moe_enabled:
            parts.append(f"ent {state.router_entropy:.2f}")
            parts.append(f"gini {state.moe_gini:.3f}")
            parts.append(f"muertos {state.dead_experts}")
            parts.append(f"collapse {'⚠' if state.router_collapse else '✓'}")
        parts.append(f"ETA {state.eta_hours:.1f}h")

        print(" | ".join(parts), flush=True)


# ── MoE Monitor ───────────────────────────────────────────────────────────────

class MoEMonitorCallback(Callback):
    """Monitorea MoE routing: entropy, Gini, CV, collapse detection."""

    def __init__(self, model: nn.Module, collapse_cv_threshold: float = 2.5) -> None:
        self.model = model
        self.collapse_cv_threshold = collapse_cv_threshold

    def on_step_end(self, state: TrainerState) -> None:
        metrics = compute_moe_metrics(self.model)
        if not metrics:
            state.moe_enabled = False
            return

        state.moe_enabled = True
        state.router_entropy = metrics.get("global/mean_entropy", 0.0)
        state.moe_gini = metrics.get("global/mean_gini", 0.0)
        state.dead_experts = metrics.get("global/total_dead", 0)
        state.router_collapse = metrics.get("global/router_collapse", False)

        # Collapse detection
        if state.router_collapse:
            state.consecutive_collapse_steps += 1
            if state.consecutive_collapse_steps >= 50:
                state.should_stop = True
                state.stop_reason = f"ROUTER_COLLAPSE ({state.consecutive_collapse_steps} steps)"
        else:
            state.consecutive_collapse_steps = 0

        # Store full metrics dict for richer logging
        state.moe_metrics = metrics


# ── Early Stop ────────────────────────────────────────────────────────────────

class EarlyStopCallback(Callback):
    """Detiene training si loss es NaN."""

    def on_step_end(self, state: TrainerState) -> None:
        if not math.isfinite(state.loss):
            state.should_stop = True
            state.stop_reason = f"LOSS_NAN ({state.loss:.4e})"


# ── Checkpoint ────────────────────────────────────────────────────────────────

class CheckpointCallback(Callback):
    """Guarda checkpoint cada save_interval steps."""

    def __init__(
        self,
        checkpoint_manager: Any,
        model: torch.nn.Module,
        optimizer: Any,
        scheduler: Any,
        save_interval: int = 500,
        ema_state: dict | None = None,
    ) -> None:
        self.manager = checkpoint_manager
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_interval = save_interval
        self.ema_state = ema_state

    def on_step_end(self, state: TrainerState) -> None:
        if (state.step + 1) % self.save_interval != 0:
            return
        state.capture_rng_states()
        extra = {}
        if self.ema_state is not None:
            extra["ema"] = self.ema_state
        self.manager.save(
            step=state.step,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            loss=state.loss,
            config=getattr(self.model, "config", {}),
            extra=extra,
        )
        self.on_save(state)

    def on_save(self, state: TrainerState) -> None:
        """Callback after checkpoint is saved — override for post-save logic."""
        pass


# ── JSONL Logger ──────────────────────────────────────────────────────────────

class JSONLLoggerCallback(Callback):
    """Writes step-level metrics to runs/run_XXXX/metrics.jsonl.

    Captures loss, aux_loss, lr, grad_norm, and MoE metrics.
    """

    def __init__(self, run_manager: RunManager, log_interval: int = 10) -> None:
        self.run = run_manager
        self.log_interval = log_interval

    def on_step_end(self, state: TrainerState) -> None:
        if (state.step + 1) % self.log_interval != 0:
            return

        metrics = {
            "loss": f"{state.loss:.4f}" if state.loss else 0,
            "ce_loss": f"{state.ce_loss:.4f}",
            "aux_loss": f"{state.aux_loss:.6f}",
            "lr": f"{state.lr:.2e}",
            "norm": f"{state.grad_norm:.2f}",
            "tok/s": int(state.tokens_per_sec),
        }

        if state.moe_enabled:
            moe = state.moe_metrics
            if moe:
                metrics["entropy"] = f"{moe.get('global/mean_entropy', 0):.4f}"
                metrics["gini"] = f"{moe.get('global/mean_gini', 0):.4f}"
                metrics["dead"] = moe.get("global/total_dead", 0)

        if state.val_ppl < float("inf"):
            metrics["val_loss"] = f"{state.val_loss:.4f}"
            metrics["val_ppl"] = f"{state.val_ppl:.2f}"

        self.run.log_metrics(step=state.step + 1, **metrics)
