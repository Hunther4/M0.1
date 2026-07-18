"""Atomic checkpointing for M0.1 training.

Provides the CheckpointManager class for saving and loading training
checkpoints with atomic file operations to prevent corruption.
"""

import os
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.optim as optim


class CheckpointManager:
    """Manages atomic save/load of training checkpoints.

    Uses an atomic write pattern: writes to a ``.checkpoint.tmp`` file
    first, then uses ``os.replace`` to make the checkpoint visible
    atomically. This prevents partial/corrupted checkpoints from being
    loaded after a crash during save.

    Args:
        checkpoint_dir: Directory path for checkpoint files. Created
            automatically if it does not exist.
    """

    def __init__(self, checkpoint_dir: str) -> None:
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save(
        self,
        step: int,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler.LRScheduler,
        loss: float,
        config: Dict[str, Any],
        epoch: int = 0,
    ) -> None:
        """Save a training checkpoint atomically.

        Writes to ``.checkpoint.tmp`` in the checkpoint directory, then
        atomically renames to ``checkpoint.pt`` via ``os.replace``.

        Args:
            step: Current training step.
            model: Model whose state_dict will be saved.
            optimizer: Optimizer whose state_dict will be saved.
            scheduler: Scheduler whose state_dict will be saved.
            loss: Current loss value.
            config: Model/training configuration dict.
            epoch: Current epoch (default 0).
        """
        checkpoint = {
            "epoch": epoch,
            "step": step,
            "loss": loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": config,
        }

        tmp_path = os.path.join(self.checkpoint_dir, ".checkpoint.tmp")
        final_path = os.path.join(self.checkpoint_dir, "checkpoint.pt")

        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, final_path)

    def load(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler.LRScheduler,
    ) -> Dict[str, Any]:
        """Load a training checkpoint and restore states.

        Args:
            model: Model whose state_dict will be restored.
            optimizer: Optimizer whose state_dict will be restored.
            scheduler: Scheduler whose state_dict will be restored.

        Returns:
            Dict with keys ``epoch``, ``step``, ``loss``, and ``config``
            from the saved checkpoint.

        Raises:
            FileNotFoundError: If ``checkpoint.pt`` does not exist in the
                checkpoint directory.
        """
        final_path = os.path.join(self.checkpoint_dir, "checkpoint.pt")
        if not os.path.exists(final_path):
            raise FileNotFoundError(
                f"No checkpoint found at {final_path}. "
                "Train the model first or check the checkpoint_dir path."
            )

        checkpoint = torch.load(final_path, weights_only=True)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        return {
            "epoch": checkpoint["epoch"],
            "step": checkpoint["step"],
            "loss": checkpoint["loss"],
            "config": checkpoint["config"],
        }
