"""Event System definitions for M0.1 TrainingEngine."""

from enum import Enum, auto


class TrainingEvent(Enum):
    """Event types emitted by TrainingEngine during execution."""

    TRAIN_BEGIN = auto()
    TRAIN_END = auto()
    STEP_BEGIN = auto()
    STEP_END = auto()
    BEFORE_BACKWARD = auto()
    AFTER_BACKWARD = auto()
    BEFORE_STEP = auto()
    AFTER_STEP = auto()
    BEFORE_ZERO_GRAD = auto()
    VALIDATION_BEGIN = auto()
    VALIDATION_END = auto()
    CHECKPOINT_SAVED = auto()
    ROUTER_COLLAPSE = auto()
    LOSS_NAN = auto()
