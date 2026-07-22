"""Shared training loop for M0.1 training scripts.

Provides a unified train function that handles the standard training loop
pattern used across all training scripts, with optional AMP, validation,
and target loss early stopping.
"""

import time

import torch


def train(
    model,
    dataloader,
    optimizer,
    criterion,
    steps,
    device,
    log_interval=250,
    scaler=None,
    val_loader=None,
    target_loss=None,
    max_batches=30,
    collapse_streak_threshold=None,
    config=None,
):
    """Run the shared training loop.

    Iterates over the dataloader for up to `steps` training steps. Supports
    optional mixed-precision training via GradScaler, periodic validation
    loss evaluation, and early stopping when a target loss is reached.

    The loop follows the standard pattern used across all M0.1 training
    scripts: `while not done: for x, y in dataloader`.

    When `collapse_streak_threshold > 0`, the loop computes MoE metrics at
    each log interval, logs them to the console, and stops training if
    router collapse is detected (consecutive zero-expert steps >= threshold).

    Args:
        model: nn.Module to train (must be in train mode before calling).
        dataloader: DataLoader yielding (input, target) batches.
        optimizer: Optimizer instance.
        criterion: Loss function (e.g. nn.CrossEntropyLoss()).
        steps: Total number of training steps.
        device: Device to use for computation.
        log_interval: Steps between log prints (default 250).
        scaler: Optional torch.cuda.amp.GradScaler for AMP training.
        val_loader: Optional DataLoader for validation loss computation.
        target_loss: Optional float; stop training if loss drops below this.
        max_batches: Maximum number of validation batches to use (default 30).
        collapse_streak_threshold: Optional explicit collapse threshold. When
            ``config`` is supplied, its MoE setting takes precedence.
        config: Optional TrainingConfig. Its MoE collapse threshold and dead
            expert ratio are read when supplied.

    Returns:
        dict with keys:
            - last_loss: float, the loss from the final step
            - steps_completed: int, total steps completed
            - elapsed: float, total time in seconds
            - stop_reason: str or None, why training stopped (if collapse)
    """
    step = 0
    start_time = time.time()
    done = False
    last_loss = 0.0
    stop_reason = None
    collapse_counters: dict[str, int] = {}
    collapse_expert_ratio = 0.0
    if config is not None:
        collapse_streak_threshold = config.moe_collapse_consecutive_steps
        collapse_expert_ratio = config.moe_collapse_expert_ratio
    elif collapse_streak_threshold is None:
        collapse_streak_threshold = 0

    while not done:
        for x, y in dataloader:
            if step >= steps:
                done = True
                break

            x, y = x.to(device), y.to(device)

            if scaler is not None:
                with torch.amp.autocast(device_type=device.type, enabled=True):
                    logits = model(x)
                    loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(x)
                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            last_loss = loss.item()

            if (step + 1) % log_interval == 0:
                elapsed = time.time() - start_time
                steps_per_sec = (step + 1) / max(elapsed, 1e-6)
                msg = (
                    f"Step {step + 1}/{steps} | Loss: {loss.item():.4f} "
                    f"| Speed: {steps_per_sec:.1f} steps/s | Time: {elapsed:.1f}s"
                )
                if val_loader is not None:
                    from src.training.eval import evaluate_val_loss

                    val_loss = evaluate_val_loss(model, val_loader, device, criterion, max_batches=max_batches)
                    val_ppl = torch.exp(torch.tensor(val_loss)).item()
                    msg += f" | Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}"
                print(msg)

                # MoE metrics logging + collapse detection
                _log_moe_and_check_collapse(
                    model, step + 1, collapse_counters,
                    collapse_streak_threshold,
                    collapse_expert_ratio,
                )
                # Check if collapse was detected by checking counters
                if collapse_streak_threshold > 0:
                    for key, ctr in list(collapse_counters.items()):
                        if ctr >= collapse_streak_threshold:
                            stop_reason = f"Router collapse in {key} at step {step + 1}"
                            print(stop_reason)
                            done = True
                            break
                if done:
                    break

            if target_loss is not None and loss.item() < target_loss:
                print(
                    f"Target loss {target_loss} reached at step {step + 1} "
                    f"(loss: {loss.item():.4f}). Stopping early."
                )
                done = True
                break

            step += 1

    elapsed = time.time() - start_time
    result: dict = {
        "last_loss": last_loss,
        "steps_completed": step,
        "elapsed": elapsed,
    }
    if stop_reason is not None:
        result["stop_reason"] = stop_reason
    return result


def _log_moe_and_check_collapse(
    model,
    step: int,
    collapse_counters: dict[str, int],
    collapse_streak_threshold: int,
    collapse_expert_ratio: float = 0.0,
) -> None:
    """Compute MoE metrics, log them, and update collapse counters.

    Args:
        model: The model (must have ``get_moe_metrics()`` method).
        step: Current training step (1-indexed) for logging.
        collapse_counters: Dict mapping layer keys to current streak counts.
            Modified in-place.
        collapse_streak_threshold: Threshold for collapse detection.
            0 disables detection.
        collapse_expert_ratio: Fraction of unused experts required to mark a
            step as collapsed. Zero means any unused expert.
    """
    # Check if model exposes MoE metrics
    if not hasattr(model, "get_moe_metrics"):
        return

    metrics = model.get_moe_metrics()
    if not metrics:
        return

    from src.training.moe_metrics import ConsoleLogger

    ConsoleLogger().log(metrics, step)

    if collapse_streak_threshold <= 0:
        return

    from src.training.moe_metrics import detect_router_collapse

    for key, val in metrics.items():
        if not key.endswith("/histogram"):
            continue
        ctr = collapse_counters.get(key, 0)
        _stop, ctr = detect_router_collapse(
            val, ctr, collapse_streak_threshold, expert_ratio=collapse_expert_ratio
        )
        collapse_counters[key] = ctr
