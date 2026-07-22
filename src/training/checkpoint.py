"""Checkpoint utilities for M0.1 training scripts.

Provides config_to_dict, save_checkpoint, and load_checkpoint helpers
used across all training scripts, plus the CheckpointManager class for
atomic checkpoint operations.
"""

import os
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.optim as optim

from src.transformer.config import M01Config
from src.engine_v2.checkpoint_v2 import safe_load_checkpoint


def config_to_dict(config):
    """Convert an M01Config instance to a serialization dict.

    Args:
        config: M01Config instance.

    Returns:
        dict with the 11 core M01Config fields.
    """
    return {field.name: getattr(config, field.name) for field in fields(config)}


def save_checkpoint(model, config, path):
    """Save model state dict and config dict to a checkpoint file.

    Creates the parent directory if it does not exist. The checkpoint dict
    contains 'model_state_dict' and 'config' keys.

    Args:
        model: nn.Module whose state_dict will be saved.
        config: M01Config instance to serialize.
        path: File path for the checkpoint.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config_to_dict(config),
        },
        path,
    )


def load_checkpoint(path, device="cpu"):
    """Load a checkpoint and reconstruct the M01Config and TransformerLM model.

    The checkpoint must contain 'model_state_dict' and 'config' keys. The
    config dict is used to build an M01Config, which in turn builds a
    TransformerLM whose state_dict is then loaded.

    Args:
        path: File path to the checkpoint.
        device: Device to load the model onto (default 'cpu').

    Returns:
        Tuple of (model, config) where model is a TransformerLM instance
        with loaded weights and config is the reconstructed M01Config.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found at {path}")

    from src.model.lm import TransformerLM

    checkpoint = safe_load_checkpoint(Path(path), map_location=device)
    state_key = "model_state" if "model_state" in checkpoint else "model_state_dict"
    config_key = "model_config" if "model_config" in checkpoint else "config"
    if state_key not in checkpoint or config_key not in checkpoint:
        raise ValueError("Checkpoint must contain model state and configuration metadata")
    ckpt_config = checkpoint[config_key]
    if not isinstance(ckpt_config, dict):
        raise ValueError("Checkpoint configuration metadata must be a dictionary")
    valid_fields = {field.name for field in fields(M01Config)}
    config = M01Config(**{key: value for key, value in ckpt_config.items() if key in valid_fields})
    model = TransformerLM(config).to(device)
    model.load_state_dict(checkpoint[state_key])

    return model, config


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
        **kwargs,
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
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        checkpoint = {
            "epoch": epoch,
            "step": step,
            "loss": loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": config,
        }
        if "extra" in kwargs and kwargs["extra"]:
            checkpoint.update(kwargs["extra"])

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

        checkpoint = safe_load_checkpoint(Path(final_path))

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        return {
            "epoch": checkpoint["epoch"],
            "step": checkpoint["step"],
            "loss": checkpoint["loss"],
            "config": checkpoint["config"],
        }
