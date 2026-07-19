"""M0.1 Training Components.

Training pipeline for the M0.1 decoder-only model.
"""

from .config import TrainingConfig
from .dataset import TinyShakespeareDataset
from .checkpoint import CheckpointManager
from .checkpoint import config_to_dict, save_checkpoint, load_checkpoint
from .loop import train
from .datasets import AmplifiedDialogueDataset, JsonlDataset
from .eval import evaluate_val_loss
from .setup import setup_device, setup_stdout

__all__ = [
    "TrainingConfig",
    "TinyShakespeareDataset",
    "CheckpointManager",
    "config_to_dict",
    "save_checkpoint",
    "load_checkpoint",
    "train",
    "AmplifiedDialogueDataset",
    "JsonlDataset",
    "evaluate_val_loss",
    "setup_device",
    "setup_stdout",
]
