"""M0.1 Training Components.

The main training entry point is ``src.training.train``, which uses the V2
engine (``src.engine_v2``). The V1 legacy modules (loop, eval, setup, datasets)
still exist for backward compatibility with scripts under ``scripts/training/``
but are NOT imported here.
"""

from .config import TrainingConfig
from .dataset import TinyShakespeareDataset, BinaryCorpusDataset
from .checkpoint import CheckpointManager
from .moe_metrics import compute_moe_metrics

# V1 legacy — still available via direct import for script compatibility:
#   from src.training.loop import train
#   from src.training.eval import evaluate_val_loss
#   from src.training.setup import setup_device, setup_stdout
#   from src.training.datasets import AmplifiedDialogueDataset, JsonlDataset

__all__ = [
    "TrainingConfig",
    "TinyShakespeareDataset",
    "BinaryCorpusDataset",
    "CheckpointManager",
    "compute_moe_metrics",
]
