"""Decoupled EventBus (Publish/Subscribe Architecture) for TrainingEngine v2."""

from collections import defaultdict
from typing import Any, Callable, Dict, List
from enum import Enum, auto


class EngineEvent(Enum):
    """Event types published over the EventBus."""

    ENGINE_INIT = auto()
    TRAIN_START = auto()
    TRAIN_END = auto()
    STEP_START = auto()
    STEP_END = auto()
    BEFORE_BACKWARD = auto()
    AFTER_BACKWARD = auto()
    VALIDATION_START = auto()
    VALIDATION_END = auto()
    CHECKPOINT_START = auto()
    CHECKPOINT_COMPLETE = auto()
    ROUTER_COLLAPSE = auto()
    LOSS_NAN = auto()


class EventBus:
    """Decoupled Event Bus supporting subscriber registration and event publishing."""

    def __init__(self) -> None:
        self._subscribers: Dict[EngineEvent, List[Callable[..., None]]] = defaultdict(list)

    def subscribe(self, event: EngineEvent, callback: Callable[..., None]) -> None:
        """Subscribe a handler to an event."""
        self._subscribers[event].append(callback)

    def publish(self, event: EngineEvent, **payload: Any) -> None:
        """Publish an event with payload to all registered subscribers."""
        for callback in self._subscribers[event]:
            callback(**payload)
