"""TrainingEngine with simplified Callback (3 hooks) + EMA.

Event & Callback driven training orchestration for M0.1 Transformer.
"""

import os
import sys
import math
import time
import signal
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .state import TrainerState
from .callbacks import Callback
from .amp import AMPContext
from .checkpoint import CheckpointManager
from .ema import ModelEMA


class TrainingEngine:
    """Training engine with Callback lifecycle and optional EMA.

    Only 3 callback hooks: on_step_end, on_validation, on_save.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        callbacks: Optional[List[Callback]] = None,
        config: Optional[Any] = None,
        gradient_accumulation_steps: int = 1,
        max_norm: float = 1.0,
        device: Optional[torch.device] = None,
        ema_decay: float = 0.0,  # 0.0 = sin EMA
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.gradient_accumulation_steps = max(1, gradient_accumulation_steps)
        self.max_norm = max_norm

        # Hardware
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.verify_rocm_hardware_guard()

        # State & AMP
        self.state = TrainerState()
        self.amp_context = AMPContext(self.device, enabled=(self.device.type == "cuda"))

        # Callbacks
        self.callbacks: List[Callback] = callbacks or []
        self.setup_default_callbacks()

        # EMA (optional)
        self.ema: Optional[ModelEMA] = None
        if ema_decay > 0.0:
            self.ema = ModelEMA(self.model, decay=ema_decay)
            print(f"  [EMA] Enabled with decay={ema_decay}")

        # Signal Handler
        self.setup_signal_handler()

    def verify_rocm_hardware_guard(self) -> None:
        """Verify ROCm GPU availability."""
        print("=" * 60)
        print("          M0.1 TrainingEngine: Hardware Verification")
        print("=" * 60)
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"[GPU ACCELERATED] {device_name}")
            print(f"[VRAM] Total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("[WARNING] GPU NOT detected! Running on CPU fallback.")
        print("=" * 60)

    def setup_default_callbacks(self) -> None:
        """Add standard callbacks if none provided."""
        from .callbacks import LoggerCallback, MoEMonitorCallback, EarlyStopCallback

        if not any(isinstance(c, LoggerCallback) for c in self.callbacks):
            self.callbacks.append(LoggerCallback(log_interval=10))
        if not any(isinstance(c, MoEMonitorCallback) for c in self.callbacks):
            self.callbacks.append(MoEMonitorCallback(self.model))
        if not any(isinstance(c, EarlyStopCallback) for c in self.callbacks):
            self.callbacks.append(EarlyStopCallback())

        ckpt_dir = getattr(self.config, "checkpoint_dir", "checkpoints") if self.config else "checkpoints"
        manager = CheckpointManager(ckpt_dir)
        from .callbacks import CheckpointCallback
        if not any(isinstance(c, CheckpointCallback) for c in self.callbacks):
            ema_state = None
            if self.ema is not None:
                ema_state = self.ema.state_dict()
            self.callbacks.append(CheckpointCallback(
                manager, self.model, self.optimizer, self.scheduler, ema_state=ema_state,
            ))

    def setup_signal_handler(self) -> None:
        """Graceful interrupt handling."""
        def sigint_handler(signum, frame):
            print("\n[SIGINT] Gracefully stopping...", flush=True)
            self.state.is_interrupted = True
            self.state.should_stop = True
            self.state.stop_reason = "SIGINT_INTERRUPT"
        signal.signal(signal.SIGINT, sigint_handler)

    def _emit(self, hook: str) -> None:
        """Dispatch a callback hook to all registered callbacks."""
        for cb in self.callbacks:
            fn = getattr(cb, hook, None)
            if callable(fn):
                fn(self.state)

    def fit(self, max_steps: int, warmup_steps: int = 200, val_interval: int = 1000) -> TrainerState:
        """Main training loop."""
        self.state.start_time = time.time()
        self.model.to(self.device)
        self.model.train()

        data_iter = iter(self.train_loader)
        batch_size = self.train_loader.batch_size or 4
        vocab_size = getattr(getattr(self.model, "config", None), "vocab_size", 8192)
        micro_step = 0

        for step in range(self.state.step, max_steps):
            if self.state.should_stop:
                print(f"[STOP] {self.state.stop_reason}", flush=True)
                break

            self.state.step = step
            step_start = time.time()

            # Get batch
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                x, y = next(data_iter)
                self.state.epoch += 1
            x, y = x.to(self.device), y.to(self.device)
            seq_len = x.size(1)

            # ── Forward ──
            t0 = time.time()
            with self.amp_context.autocast():
                logits = self.model(x)
                ce_loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
                router_aux_loss = torch.tensor(0.0, device=self.device)
                if hasattr(self.model, "get_aux_loss"):
                    router_aux_loss = self.model.get_aux_loss()
                loss = (ce_loss + router_aux_loss) / self.gradient_accumulation_steps

            self.state.forward_ms = (time.time() - t0) * 1000
            self.state.ce_loss = ce_loss.item()
            self.state.aux_loss = router_aux_loss.item()
            self.state.loss = (ce_loss + router_aux_loss).item()
            self.state.global_tokens += batch_size * seq_len

            # ── Backward ──
            self.amp_context.scale(loss).backward()
            micro_step += 1

            # ── Optimizer step (on accumulation boundary) ──
            if micro_step % self.gradient_accumulation_steps == 0:
                self.amp_context.unscale_(self.optimizer)
                norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
                self.state.grad_norm = norm.item()
                self.amp_context.step(self.optimizer)
                self.amp_context.update()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

                # EMA update (after optimizer)
                if self.ema is not None:
                    self.ema.update()

            # ── Step Metrics ──
            step_time = time.time() - step_start
            self.state.tokens_per_sec = (batch_size * seq_len) / max(step_time, 1e-6)
            self.state.samples_per_sec = batch_size / max(step_time, 1e-6)
            self.state.elapsed_seconds = time.time() - self.state.start_time
            remaining = max_steps - (step + 1)
            self.state.eta_hours = (remaining * step_time) / 3600
            self.state.lr = self.scheduler.get_last_lr()[0]
            self.state.scaler_scale = self.amp_context.get_scale()

            if self.device.type == "cuda":
                self.state.vram_allocated_mb = torch.cuda.memory_allocated() / 1e6
                self.state.vram_peak_mb = torch.cuda.max_memory_allocated() / 1e6

            # ── Callback hook ──
            self._emit("on_step_end")

            # ── Validation ──
            if self.val_loader is not None and (step + 1) % val_interval == 0:
                self.run_validation(vocab_size)

        self.state.is_done = True
        return self.state

    def run_validation(self, vocab_size: int) -> float:
        """Validation loop. Uses EMA weights if available."""
        if self.val_loader is None:
            return 0.0

        # Apply EMA weights for validation if available
        if self.ema is not None:
            self.ema.apply_shadow()

        self.model.eval()
        total_loss = 0.0
        steps = 0

        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)
                with self.amp_context.autocast():
                    logits = self.model(x)
                    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
                total_loss += loss.item()
                steps += 1
                if steps >= 50:
                    break

        avg_loss = total_loss / max(steps, 1)
        self.state.val_loss = avg_loss
        self.state.val_ppl = math.exp(min(avg_loss, 50.0))
        if avg_loss < self.state.best_val_loss:
            self.state.best_val_loss = avg_loss

        print(f"[VAL] Step {self.state.step + 1} | Loss: {avg_loss:.4f} | PPL: {self.state.val_ppl:.2f}", flush=True)

        self.model.train()

        # Restore original weights
        if self.ema is not None:
            self.ema.restore()

        # Callback hook
        self._emit("on_validation")
        return avg_loss
