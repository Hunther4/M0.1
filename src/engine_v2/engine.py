"""TrainingEngineV2: Enterprise-Grade Hardened Research Framework Core for M0.1."""

import os
import sys
import time
import signal
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .fsm import StateMachine, EngineState
from .bus import EventBus, EngineEvent
from .loss_pipeline import LossPipeline, CrossEntropyLossTerm, RouterAuxLossTerm, RouterZLossTerm
from .metrics import MetricRegistry
from .checkpoint_v2 import AsyncCheckpointManagerV2, normalize_checkpoint_state, safe_load_checkpoint
from .experiment import ExperimentManager
from .profiler import GranularProfiler
from .plugins import BasePlugin
from .amp import AMPContext
from .ema import EMA
from .health import HealthChecker
from .loggers import ConsoleLogger, JSONLLogger, CSVLogger


def get_cuda_memory_metrics_mb() -> Dict[str, float]:
    """Return current and peak allocator memory in MB for CUDA or ROCm."""
    if not torch.cuda.is_available():
        allocated = reserved = peak_reserved = 0.0
    else:
        allocated = torch.cuda.memory_allocated() / 1e6
        reserved = torch.cuda.memory_reserved() / 1e6
        peak_reserved = torch.cuda.max_memory_reserved() / 1e6

    return {
        "vram_mb": allocated,
        "vram_alloc_mb": allocated,
        "vram_reserved_mb": reserved,
        "vram_reserved_peak_mb": peak_reserved,
    }


def reset_cuda_peak_memory_stats() -> None:
    """Reset peak allocator memory once before a CUDA or ROCm training run."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


class TrainingEngineV2:
    """Enterprise-Grade Hardened Model-Agnostic FSM Training Engine."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        loss_pipeline: Optional[LossPipeline] = None,
        callbacks: Optional[List[BasePlugin]] = None,
        plugins: Optional[List[BasePlugin]] = None,
        config: Optional[Any] = None,
        experiment_manager: Optional[ExperimentManager] = None,
        gradient_accumulation_steps: int = 1,
        max_norm: float = 1.0,
        enable_hooks: bool = True,
        enable_ema: bool = True,
        device: Optional[torch.device] = None,
    ) -> None:
        self.fsm = StateMachine(EngineState.INIT)
        self.bus = EventBus()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.gradient_accumulation_steps = max(1, gradient_accumulation_steps)
        self.max_norm = max_norm
        self.enable_hooks = enable_hooks
        self.enable_ema = enable_ema
        self.max_recovery_attempts = max(
            1, int(getattr(config, "max_recovery_attempts", 3))
        )
        self.recovery_lr_factor = float(getattr(config, "recovery_lr_factor", 0.5))
        if not 0.0 < self.recovery_lr_factor <= 1.0:
            raise ValueError("recovery_lr_factor must be in the interval (0, 1]")

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.experiment = experiment_manager or ExperimentManager()
        self.checkpoint_mgr = AsyncCheckpointManagerV2(str(self.experiment.checkpoints_dir))
        self.profiler = GranularProfiler()
        self.metrics = MetricRegistry()
        self.health_checker = HealthChecker(self.model)

        # 1. AMP Context
        self.amp_context = AMPContext(self.device, enabled=(self.device.type == "cuda"))

        # 2. EMA Setup
        self.ema = EMA(self.model) if self.enable_ema else None

        # 3. Loss Pipeline
        vocab_size = getattr(getattr(self.model, "config", None), "vocab_size", 8192)
        self.loss_pipeline = loss_pipeline or LossPipeline([
            CrossEntropyLossTerm(vocab_size=vocab_size),
            RouterAuxLossTerm(weight=0.2),
            RouterZLossTerm(weight=0.01),
        ])

        # 4. Multichannel Loggers
        self.console_logger = ConsoleLogger(log_interval=10)
        self.jsonl_logger = JSONLLogger(self.experiment.run_dir / "metrics.jsonl")
        self.csv_logger = CSVLogger(self.experiment.run_dir / "metrics.csv")

        # 5. Forward Hooks (Configurable)
        if self.enable_hooks:
            self.setup_model_hooks()

        # 6. Plugins
        self.plugins = plugins or callbacks or []
        for p in self.plugins:
            p.register(self)

        # 7. Environment Traceability
        env_meta = {
            **self.checkpoint_mgr.capture_environment_metadata(),
            **self.experiment.capture_full_metadata(),
        }
        self.experiment.log_environment(env_meta)

        # 8. Graceful Signal Handlers
        self.should_stop = False
        self.current_step = 0
        self.global_tokens = 0
        self._dataset_hash_cache: Optional[str] = None
        self._tokenizer_hash_cache: Optional[str] = None
        self._moe_aux_loss_ema: Optional[float] = None
        self._moe_collapse_counters: Dict[str, int] = {}
        self.setup_signal_handlers()

        self.bus.publish(EngineEvent.ENGINE_INIT, engine=self)

    def setup_signal_handlers(self) -> None:
        """Catch SIGINT and SIGTERM for graceful shutdown."""
        def _handler(signum, frame):
            print("\n[SIGINT/SIGTERM] Catching shutdown signal. Saving canonical checkpoint...", flush=True)
            self.should_stop = True

        signal.signal(signal.SIGINT, _handler)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, _handler)

    def setup_model_hooks(self) -> None:
        """Register PyTorch Forward Hooks to capture MoE and attention telemetry automatically."""
        def _hook(module, input_tensor, output_tensor):
            if hasattr(module, "gate_probs"):
                self.metrics.update("router_gate_probs", module.gate_probs)
                self.metrics.update("router_kl_div", MetricRegistry.compute_kl_divergence(module.gate_probs))
            if hasattr(module, "expert_mask"):
                self.metrics.update("expert_gini_index", MetricRegistry.compute_gini_index(module.expert_mask.sum(dim=1)))

        for block in getattr(self.model, "blocks", []):
            ff = getattr(block, "ff", None)
            if ff is not None:
                ff.register_forward_hook(_hook)

    def resume(self, checkpoint_path: Optional[str] = None) -> int:
        """Resume training from canonical checkpoint or a given path, restoring full state and RNGs.

        If ``checkpoint_path`` is provided and exists, it is loaded directly (used to stack
        knowledge on top of a previous run's checkpoint). Otherwise the current run's canonical
        checkpoint is used.
        """
        if checkpoint_path is not None:
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Explicit resume checkpoint not found: {checkpoint_path}")
            print(f"[RESUME] Loading checkpoint from {checkpoint_path}")
            state = safe_load_checkpoint(Path(checkpoint_path))
        else:
            state = self.checkpoint_mgr.load_canonical()
        state = normalize_checkpoint_state(state, require_architecture=True)
        self._assert_config_compatible(state)

        model_state = state["model_state"]
        current_state = self.model.state_dict()
        compatible = {}
        skipped = []
        for key, param in current_state.items():
            if key in model_state and model_state[key].shape == param.shape:
                compatible[key] = model_state[key]
            elif key in model_state:
                skipped.append(
                    f"  {key}: checkpoint {list(model_state[key].shape)} "
                    f"vs model {list(param.shape)}"
                )
            else:
                skipped.append(f"  {key}: not found in checkpoint")

        missing, unexpected = self.model.load_state_dict(compatible, strict=False)

        if skipped:
            print(
                f"[RESUME] Partial load — {len(compatible)} keys loaded, "
                f"{len(skipped)} MoE keys skipped (shape mismatch):"
            )
            for entry in skipped:
                print(entry)
        if missing:
            print(f"[RESUME] {len(missing)} keys missing from checkpoint (expected for re-initialized MoE):")
            for key in missing:
                print(f"  {key}")
        if unexpected:
            print(f"[RESUME] {len(unexpected)} unexpected keys in checkpoint (old MoE shapes):")
            for key in unexpected:
                print(f"  {key}")

        if "optimizer_state" in state:
            model_device = next(self.model.parameters()).device
            try:
                # Move checkpoint optimizer state to model device BEFORE load_state_dict
                # so all tensors are on the correct device from the start.
                ckpt_opt = state["optimizer_state"]
                for param_state in ckpt_opt.get("state", {}).values():
                    for key, value in param_state.items():
                        if torch.is_tensor(value):
                            param_state[key] = value.to(model_device, non_blocking=True)
                self.optimizer.load_state_dict(ckpt_opt)
            except ValueError as error:
                if "different number of parameter groups" not in str(error):
                    raise
                print(
                    "[RESUME] Skipped optimizer state because its parameter groups are incompatible; "
                    "using a fresh optimizer state."
                )
            else:
                skipped_optimizer_tensors = 0
                reinitialized_params = set()
                for parameter, parameter_state in list(self.optimizer.state.items()):
                    for state_key, value in list(parameter_state.items()):
                        if torch.is_tensor(value) and value.ndim and value.shape != parameter.shape:
                            del parameter_state[state_key]
                            skipped_optimizer_tensors += 1
                            reinitialized_params.add(id(parameter))
                # Wipe the entire state for params that lost ANY buffer — the
                # optimizer's lazy init will create fresh ones on next step().
                for parameter in list(self.optimizer.state.keys()):
                    if id(parameter) in reinitialized_params:
                        del self.optimizer.state[parameter]
                if skipped_optimizer_tensors:
                    print(
                        f"[RESUME] Reinitialized {skipped_optimizer_tensors} optimizer state tensors "
                        "with incompatible model shapes."
                    )
        if "scheduler_state" in state:
            self.scheduler.load_state_dict(state["scheduler_state"])

        if self.ema and isinstance(state.get("ema_state"), dict):
            saved_ema_state = dict(state["ema_state"])
            saved_shadows = saved_ema_state.get("shadow", {})
            compatible_shadows = {
                key: value
                for key, value in saved_shadows.items()
                if key in self.ema.shadow and value.shape == self.ema.shadow[key].shape
            }
            skipped_ema_shadows = len(saved_shadows) - len(compatible_shadows)
            saved_ema_state["shadow"] = {**self.ema.shadow, **compatible_shadows}
            self.ema.load_state_dict(saved_ema_state)
            if skipped_ema_shadows:
                print(
                    f"[RESUME] Reinitialized {skipped_ema_shadows} EMA shadows "
                    "with incompatible model shapes."
                )

        if "amp_scaler_state" in state:
            self.amp_context.load_state_dict(state["amp_scaler_state"])

        if "rng_states" in state:
            AsyncCheckpointManagerV2.restore_rng_states(state["rng_states"])

        resumed_step = int(state.get("step", 0))
        self.current_step = resumed_step
        self.global_tokens = int(state.get("global_tokens", state.get("tokens_seen", 0)))
        print(f"[RESUME] Restored execution state at step {resumed_step + 1}")
        return resumed_step

    def _assert_config_compatible(self, state: Dict[str, Any]) -> None:
        """Fail loudly if the resumed checkpoint's architecture differs from the current model.

        Prevents silently corrupting a "stacked" model by resuming onto an incompatible checkpoint.
        """
        saved = state.get("model_config") or state.get("config")
        if not saved:
            return
        cur = getattr(self.model, "config", None)
        if cur is None:
            return
        for key in ("vocab_size", "n_layers", "d_model", "num_experts", "num_shared_experts"):
            if getattr(cur, key, None) != saved.get(key):
                raise ValueError(
                    f"[RESUME] Architecture mismatch on '{key}': checkpoint={saved.get(key)} "
                    f"vs current model={getattr(cur, key, None)}. "
                    f"Cannot stack layers on an incompatible checkpoint."
                )

    def save_checkpoint(self, step: int = 0, global_tokens: int = 0) -> None:
        """Save canonical checkpoint explicitly with dataset/tokenizer hashes."""
        self.fsm.transition_to(EngineState.SAVE)
        self.bus.publish(EngineEvent.CHECKPOINT_START)
        
        # Fingerprint the actual loader sources once per engine instance.
        data_hash = self._dataset_hash_cache or "unknown"
        tok_hash = self._tokenizer_hash_cache or "unknown"
        try:
            if self._dataset_hash_cache is None:
                dataset = getattr(self.train_loader, "dataset", None)
                source_files = [
                    Path(path) for path in getattr(dataset, "source_files", [])
                    if Path(path).is_file()
                ]
                if source_files:
                    self._dataset_hash_cache = self.checkpoint_mgr.calculate_manifest_sha256(
                        source_files
                    )
                    data_hash = self._dataset_hash_cache
            if self._tokenizer_hash_cache is None:
                data_dir = Path(getattr(self.config, "data_dir", "data"))
                tok_file = data_dir / "tokenizers" / "tokenizer.json"
                if not tok_file.exists():
                    tok_file = Path("data") / "tokenizers" / "tokenizer.json"
                if tok_file.exists():
                    self._tokenizer_hash_cache = self.checkpoint_mgr.calculate_sha256(tok_file)
                    tok_hash = self._tokenizer_hash_cache
        except Exception:
            pass

        state_to_save = {
            "step": step,
            "global_tokens": global_tokens,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "ema_state": self.ema.state_dict() if self.ema else None,
            "amp_scaler_state": self.amp_context.state_dict(),
            "rng_states": AsyncCheckpointManagerV2.capture_rng_states(),
            "metrics": self.metrics.get_all(),
            "env": self.checkpoint_mgr.capture_environment_metadata(),
            "dataset_hash": data_hash,
            "tokenizer_hash": tok_hash,
            "model_config": dict(self.model.config.__dict__) if hasattr(self.model, "config") else None,
        }
        self.checkpoint_mgr.save_canonical_async(state_to_save)
        self.checkpoint_mgr.wait_completion()
        self.bus.publish(EngineEvent.CHECKPOINT_COMPLETE, file="checkpoint.pt")
        self.fsm.transition_to(EngineState.TRAIN)

    def _data_iterator_at_step(self, step: int) -> Iterator[Any]:
        """Build an iterator positioned at the batch represented by ``step``."""
        data_iter = iter(self.train_loader)
        try:
            loader_length = len(self.train_loader)
        except TypeError:
            loader_length = 0

        batches_to_skip = step % loader_length if loader_length else 0
        for _ in range(batches_to_skip):
            try:
                next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                next(data_iter)
        return data_iter

    def _scale_recovery_lr(self, factor: float) -> None:
        """Scale optimizer LR and scheduler anchors after checkpoint restore."""
        for param_group in self.optimizer.param_groups:
            param_group["lr"] *= factor
            if "initial_lr" in param_group:
                param_group["initial_lr"] *= factor

        if hasattr(self.scheduler, "base_lrs"):
            self.scheduler.base_lrs = [lr * factor for lr in self.scheduler.base_lrs]
        if hasattr(self.scheduler, "_last_lr"):
            self.scheduler._last_lr = [
                param_group["lr"] for param_group in self.optimizer.param_groups
            ]

    def _recover_from_checkpoint(
        self, recovery_attempt: int
    ) -> Tuple[int, int, Iterator[Any]]:
        """Restore a clean checkpoint and return coherent execution counters."""
        self.fsm.transition_to(EngineState.RECOVERING)
        self.checkpoint_mgr.wait_completion()
        print("[RECOVERY ROLLBACK] Restoring last clean canonical checkpoint...")
        try:
            restored_step = self.resume()
        except Exception as error:
            self.optimizer.zero_grad(set_to_none=True)
            self.fsm.transition_to(EngineState.ERROR)
            raise RuntimeError("Recovery checkpoint could not be restored") from error

        # The checkpoint owns optimizer and scheduler state. Apply the retry
        # reduction only after both have been restored, and keep their anchors aligned.
        self._scale_recovery_lr(self.recovery_lr_factor ** recovery_attempt)
        self.optimizer.zero_grad(set_to_none=True)
        self.model.train()
        data_iter = self._data_iterator_at_step(restored_step)
        self.fsm.transition_to(EngineState.TRAIN)
        return restored_step, self.global_tokens, data_iter

    def fit(self, max_steps: int) -> Dict[str, Any]:
        """Run FSM training cycle with automatic recovery on NaN/Inf."""
        self.fsm.transition_to(EngineState.LOAD)
        self.model.to(self.device)
        if self.device.type == "cuda":
            reset_cuda_peak_memory_stats()
        
        # Save baseline checkpoint at start of fit for safe recovery rollback
        if not self.checkpoint_mgr.canonical_path.exists():
            print("[CHECKPOINT] Saving baseline step 0 checkpoint for recovery protection...")
            self.save_checkpoint(step=0, global_tokens=0)

        self.fsm.transition_to(EngineState.TRAIN)
        self.bus.publish(EngineEvent.TRAIN_START)

        self.model.train()
        data_iter = self._data_iterator_at_step(self.current_step)
        batch_size = self.train_loader.batch_size or 4
        global_tokens = self.global_tokens
        step_loss = 0.0
        start_time = time.time()
        step = self.current_step
        recovery_attempts = 0

        while step < max_steps:
            if self.should_stop:
                print(f"[GRACEFUL SHUTDOWN] Stopping at step {step}")
                self.current_step = step
                break

            self.bus.publish(EngineEvent.STEP_START, step=step)
            self.profiler.start("dataloader")

            self.bus.publish(EngineEvent.STEP_START, step=step)
            self.profiler.start("dataloader")
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                x, y = next(data_iter)
            self.profiler.stop("dataloader")

            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            seq_len = x.size(1)

            # Forward pass under AMP
            self.profiler.start("forward")
            if hasattr(self.model, "set_moe_step"):
                self.model.set_moe_step(step)
            with self.amp_context.autocast():
                logits = self.model(x)
                loss = self.loss_pipeline(logits, y, model=self.model) / self.gradient_accumulation_steps
            self.profiler.stop("forward")

            step_loss = loss.detach().item() * self.gradient_accumulation_steps

            # ── Loss Health Check ─────────────────────────────────────
            loss_healthy, loss_reason = self.health_checker.check_loss(step_loss)
            if not loss_healthy:
                print(f"[RECOVERY TRIGGERED] {loss_reason} at step {step+1}")
                self.bus.publish(EngineEvent.LOSS_NAN, reason=loss_reason)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.optimizer.zero_grad(set_to_none=True)
                recovery_attempts += 1
                if recovery_attempts >= self.max_recovery_attempts:
                    self.checkpoint_mgr.wait_completion()
                    self.fsm.transition_to(EngineState.ERROR)
                    raise RuntimeError(
                        "Training recovery exhausted after "
                        f"{recovery_attempts} consecutive loss-NaN checks "
                        f"at step {step + 1}: {loss_reason}"
                    )
                step, global_tokens, data_iter = self._recover_from_checkpoint(
                    recovery_attempts
                )
                continue

            # Backward pass under AMP
            self.profiler.start("backward")
            self.bus.publish(EngineEvent.BEFORE_BACKWARD)
            self.amp_context.scale(loss).backward()
            self.bus.publish(EngineEvent.AFTER_BACKWARD)
            self.profiler.stop("backward")

            if (step + 1) % self.gradient_accumulation_steps == 0:
                self.amp_context.unscale_(self.optimizer)
                grad_stats = self.health_checker.monitor_gradients()
                self.metrics.update("grad_stats", grad_stats)

                healthy, reason = self.health_checker.check_health()
                if not healthy:
                    print(f"[RECOVERY TRIGGERED] Health check failed at step {step+1}: {reason}")
                    self.bus.publish(EngineEvent.LOSS_NAN, reason=reason)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    recovery_attempts += 1
                    if recovery_attempts >= self.max_recovery_attempts:
                        self.checkpoint_mgr.wait_completion()
                        self.optimizer.zero_grad(set_to_none=True)
                        self.fsm.transition_to(EngineState.ERROR)
                        raise RuntimeError(
                            "Training recovery exhausted after "
                            f"{recovery_attempts} consecutive health-check failures "
                            f"at step {step + 1}: {reason}"
                        )
                    step, global_tokens, data_iter = self._recover_from_checkpoint(
                        recovery_attempts
                    )
                    continue

                recovery_attempts = 0

                self.profiler.start("optimizer")
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
                # Safety net: ensure optimizer state is on model device before step.
                # This catches any edge case where resume's device fix didn't apply.
                model_device = next(self.model.parameters()).device
                for state_data in self.optimizer.state.values():
                    for k, v in list(state_data.items()):
                        if torch.is_tensor(v) and v.device != model_device:
                            state_data[k] = v.to(model_device, non_blocking=True)
                self.amp_context.step(self.optimizer)
                self.amp_context.update()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

                if self.ema:
                    self.ema.update()

                self.profiler.stop("optimizer")

            global_tokens += batch_size * seq_len
            self.current_step = step + 1
            self.global_tokens = global_tokens
            elapsed_time = time.time() - start_time
            tok_s = int(global_tokens / max(elapsed_time, 1e-6))
            current_lr = self.scheduler.get_last_lr()[0]

            log_data = {
                "loss": step_loss,
                "lr": current_lr,
                "tok_s": tok_s,
                **get_cuda_memory_metrics_mb(),
                **self.loss_pipeline.last_breakdown,
            }

            self._log_moe_metrics(step + 1, log_data)

            self.console_logger.log(step, log_data)
            self.jsonl_logger.log(step, log_data)
            self.csv_logger.log(step, log_data)

            self.bus.publish(EngineEvent.STEP_END, step=step, loss=step_loss)

            # Validation with EMA weights
            val_interval = int(getattr(self.config, "val_interval", 500))
            if val_interval > 0 and (step + 1) % val_interval == 0 and self.val_loader is not None:
                self.fsm.transition_to(EngineState.VALIDATE)
                self.bus.publish(EngineEvent.VALIDATION_START)
                val_loss = self._validate_with_ema()
                self.metrics.update("val_loss", val_loss)
                self.bus.publish(EngineEvent.VALIDATION_END, val_loss=val_loss)
                self.fsm.transition_to(EngineState.TRAIN)

            # Save Canonical Checkpoint (Async Background Save)
            save_interval = int(getattr(self.config, "save_interval", 1000))
            if (save_interval > 0 and (step + 1) % save_interval == 0) or self.should_stop:
                self.save_checkpoint(step=self.current_step, global_tokens=global_tokens)

            step = self.current_step

        # Ensure canonical checkpoint is saved at end of fit()
        self.save_checkpoint(step=self.current_step, global_tokens=global_tokens)
        self.checkpoint_mgr.wait_completion()
        self.profiler.export(self.experiment.run_dir / "training_profile.json", total_tokens=global_tokens, total_batches=self.current_step)

        self.fsm.transition_to(EngineState.FINISHED)
        self.bus.publish(EngineEvent.TRAIN_END)

        summary = {"final_step": self.current_step, "final_loss": step_loss, "total_tokens": global_tokens}
        self.experiment.save_summary(summary)
        return summary

    def _log_moe_metrics(self, step: int, log_data: Dict[str, Any]) -> None:
        """Log MoE metrics and track configured auxiliary-loss EMA."""
        if not hasattr(self.model, "get_moe_metrics"):
            return
        if self.config is not None and not getattr(self.config, "log_moe_metrics", True):
            return
        log_interval = max(1, int(getattr(self.config, "log_interval", 10)))
        if step % log_interval != 0:
            return

        metrics = self.model.get_moe_metrics()
        if not metrics:
            return

        from src.training.moe_metrics import ConsoleLogger as MoeConsoleLogger
        from src.training.moe_metrics import aux_loss_ema, detect_router_collapse

        MoeConsoleLogger().log(metrics, step)
        current_aux = float(self.model.get_aux_loss().detach().item())
        previous_aux = self._moe_aux_loss_ema
        if previous_aux is None:
            previous_aux = current_aux
        self._moe_aux_loss_ema = aux_loss_ema(current_aux, previous_aux)
        self.metrics.update("moe_aux_loss_ema", self._moe_aux_loss_ema)
        log_data["moe_aux_loss_ema"] = self._moe_aux_loss_ema

        threshold = int(getattr(self.config, "moe_collapse_consecutive_steps", 50))
        expert_ratio = float(getattr(self.config, "moe_collapse_expert_ratio", 0.3))
        if threshold <= 0:
            return
        for key, histogram in metrics.items():
            if not key.endswith("/histogram"):
                continue
            counter = self._moe_collapse_counters.get(key, 0)
            stop, counter = detect_router_collapse(
                histogram, counter, threshold, expert_ratio=expert_ratio
            )
            self._moe_collapse_counters[key] = counter
            if stop:
                self.should_stop = True
                print(f"Router collapse detected in {key} at step {step}. Stopping training.", flush=True)
                break

    def _validate_with_ema(self) -> float:
        """Run validation using EMA weights, then restore original model weights."""
        if self.ema:
            self.ema.apply_shadow()

        self.model.eval()
        total_loss = 0.0
        steps = 0
        vocab_size = getattr(getattr(self.model, "config", None), "vocab_size", 8192)

        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                loss = CrossEntropyLossTerm(vocab_size=vocab_size)(logits, y)
                total_loss += loss.item()
                steps += 1
                if steps >= 30:
                    break

        self.model.train()

        if self.ema:
            self.ema.restore()

        return total_loss / max(steps, 1)
