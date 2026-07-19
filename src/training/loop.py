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
):
    """Run the shared training loop.

    Iterates over the dataloader for up to `steps` training steps. Supports
    optional mixed-precision training via GradScaler, periodic validation
    loss evaluation, and early stopping when a target loss is reached.

    The loop follows the standard pattern used across all M0.1 training
    scripts: `while not done: for x, y in dataloader`.

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

    Returns:
        dict with keys:
            - last_loss: float, the loss from the final step
            - steps_completed: int, total steps completed
            - elapsed: float, total time in seconds
    """
    step = 0
    start_time = time.time()
    done = False
    last_loss = 0.0

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
                steps_per_sec = (step + 1) / elapsed
                msg = (
                    f"Step {step + 1}/{steps} | Loss: {loss.item():.4f} "
                    f"| Speed: {steps_per_sec:.1f} steps/s | Time: {elapsed:.1f}s"
                )
                if val_loader is not None:
                    from src.training.eval import evaluate_val_loss

                    val_loss = evaluate_val_loss(model, val_loader, device, criterion)
                    val_ppl = torch.exp(torch.tensor(val_loss)).item()
                    msg += f" | Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}"
                print(msg)

            if target_loss is not None and loss.item() < target_loss:
                print(
                    f"Target loss {target_loss} reached at step {step + 1} "
                    f"(loss: {loss.item():.4f}). Stopping early."
                )
                done = True
                break

            step += 1

    elapsed = time.time() - start_time
    return {
        "last_loss": last_loss,
        "steps_completed": step,
        "elapsed": elapsed,
    }
