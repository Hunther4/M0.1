"""Evaluation utilities for M0.1 training scripts.

Provides the shared validation loss computation used across training
scripts that perform periodic evaluation during training.
"""

import torch


def evaluate_val_loss(model, val_loader, device, criterion, max_batches=30):
    """Compute average validation loss over a limited number of batches.

    Sets the model to eval mode, computes cross-entropy loss on up to
    max_batches from the validation loader, then sets the model back to
    train mode.

    Args:
        model: nn.Module to evaluate.
        val_loader: DataLoader yielding (input, target) batches.
        device: Device to use for computation.
        criterion: Loss function (e.g. nn.CrossEntropyLoss()).
        max_batches: Maximum number of validation batches to use (default 30).

    Returns:
        float: Average validation loss across the evaluated batches.
    """
    model.eval()
    total_loss = 0.0
    steps = 0
    with torch.no_grad():
        for x, y in val_loader:
            if steps >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item()
            steps += 1
    model.train()
    return total_loss / max(steps, 1)
