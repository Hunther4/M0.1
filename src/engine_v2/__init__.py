"""M0.1 Training Engine V2 — Enterprise-Grade Hardened Training Framework.

This is the active training engine used by ``src.training.train`` (the main entry
point). It replaces the V1 callback-based engine with a Finite State Machine
(FSM), EventBus, composable LossPipeline, and plugin architecture.

Safe imports only — modules requiring optional dependencies (e.g. ``psutil`` in
``engine`` and ``experiment``) are available via direct import:
    from src.engine_v2.engine import TrainingEngineV2
    from src.engine_v2.experiment import ExperimentManager
"""

from .fsm import StateMachine, EngineState
from .bus import EventBus, EngineEvent
from .loss_pipeline import LossPipeline, CrossEntropyLossTerm, RouterAuxLossTerm, RouterZLossTerm
from .metrics import MetricRegistry
from .profiler import GranularProfiler
from .loggers import ConsoleLogger, JSONLLogger, CSVLogger

__all__ = [
    "StateMachine",
    "EngineState",
    "EventBus",
    "EngineEvent",
    "LossPipeline",
    "CrossEntropyLossTerm",
    "RouterAuxLossTerm",
    "RouterZLossTerm",
    "MetricRegistry",
    "GranularProfiler",
    "ConsoleLogger",
    "JSONLLogger",
    "CSVLogger",
]
