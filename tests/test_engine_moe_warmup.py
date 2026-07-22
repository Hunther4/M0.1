"""Integration tests for MoE capacity warmup scheduling in TrainingEngineV2."""

from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


from src.engine_v2.amp import AMPContext
from src.engine_v2.bus import EventBus
from src.engine_v2.engine import TrainingEngineV2
from src.engine_v2.fsm import EngineState, StateMachine
from src.engine_v2.health import HealthChecker
from src.engine_v2.loss_pipeline import CrossEntropyLossTerm, LossPipeline
from src.engine_v2.metrics import MetricRegistry


class StepAwareToyModel(nn.Module):
    """Records the MoE step active when each forward pass begins."""

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(vocab_size=4)
        self.projection = nn.Linear(1, self.config.vocab_size)
        self.steps: list[int] = []
        self.forward_steps: list[int] = []
        self._active_step = -1

    def set_moe_step(self, step: int) -> None:
        self.steps.append(step)
        self._active_step = step

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.forward_steps.append(self._active_step)
        return self.projection(x.float().unsqueeze(-1))


class PlainToyModel(nn.Module):
    """A non-MoE model accepted by the generic training engine."""

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(vocab_size=4)
        self.projection = nn.Linear(1, self.config.vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x.float().unsqueeze(-1))


class _NoOpProfiler:
    def start(self, _: str) -> None:
        pass

    def stop(self, _: str) -> None:
        pass

    def export(self, _: Path, **__: int) -> None:
        pass


class _NoOpLogger:
    def log(self, *_: object) -> None:
        pass


def _build_test_engine(model: nn.Module, tmp_path: Path) -> TrainingEngineV2:
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    train_loader = DataLoader(
        TensorDataset(torch.ones(1, 2, dtype=torch.long), torch.ones(1, 2, dtype=torch.long)),
        batch_size=1,
    )
    engine = TrainingEngineV2.__new__(TrainingEngineV2)
    engine.fsm = StateMachine(EngineState.INIT)
    engine.model = model
    engine.optimizer = optimizer
    engine.scheduler = scheduler
    engine.train_loader = train_loader
    engine.val_loader = None
    engine.loss_pipeline = LossPipeline([CrossEntropyLossTerm(vocab_size=4)])
    engine.config = SimpleNamespace(val_interval=0, save_interval=0)
    engine.gradient_accumulation_steps = 1
    engine.max_norm = 1.0
    engine.device = torch.device("cpu")
    engine.checkpoint_mgr = SimpleNamespace(
        canonical_path=tmp_path / "checkpoint.pt",
        wait_completion=lambda: None,
    )
    engine.profiler = _NoOpProfiler()
    engine.metrics = MetricRegistry()
    engine.health_checker = HealthChecker(model)
    engine.amp_context = AMPContext(engine.device, enabled=False)
    engine.ema = None
    engine.console_logger = _NoOpLogger()
    engine.jsonl_logger = _NoOpLogger()
    engine.csv_logger = _NoOpLogger()
    engine.bus = EventBus()
    engine.should_stop = False
    engine.current_step = 0
    engine.global_tokens = 0
    engine.experiment = SimpleNamespace(run_dir=tmp_path, save_summary=lambda _: None)
    engine.save_checkpoint = lambda **_: None
    return engine


def test_training_engine_sets_moe_step_before_each_forward(tmp_path: Path) -> None:
    model = StepAwareToyModel()
    engine = _build_test_engine(model, tmp_path)

    summary = engine.fit(max_steps=2)

    assert summary["final_step"] == 2
    assert model.steps == [0, 1]
    assert model.forward_steps == [0, 1]


def test_training_engine_supports_models_without_moe_step(tmp_path: Path) -> None:
    engine = _build_test_engine(PlainToyModel(), tmp_path)

    summary = engine.fit(max_steps=1)

    assert summary["final_step"] == 1
