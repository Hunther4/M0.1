"""Focused regression tests for bounded TrainingEngineV2 recovery."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.engine_v2.engine import TrainingEngineV2
from src.engine_v2.experiment import ExperimentManager
from src.engine_v2.fsm import EngineState, StateMachine
from src.engine_v2.loss_pipeline import CrossEntropyLossTerm, LossPipeline


class TinyLanguageModel(nn.Module):
    def __init__(self, vocab_size: int = 8) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 4)
        self.projection = nn.Linear(4, vocab_size)
        self.config = SimpleNamespace(
            vocab_size=vocab_size,
            n_layers=1,
            d_model=4,
            num_experts=0,
            num_shared_experts=0,
        )
        self.seen_batches = []

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        self.seen_batches.append(tokens.detach().cpu().clone())
        return self.projection(self.embedding(tokens))


def make_engine(tmp_path, *, max_recovery_attempts: int = 3) -> TrainingEngineV2:
    model = TinyLanguageModel()
    x = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]])
    y = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 0]])
    loader = DataLoader(TensorDataset(x, y), batch_size=1, shuffle=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    config = SimpleNamespace(
        max_recovery_attempts=max_recovery_attempts,
        recovery_lr_factor=0.5,
        save_interval=0,
        val_interval=0,
    )
    return TrainingEngineV2(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=loader,
        loss_pipeline=LossPipeline([CrossEntropyLossTerm(vocab_size=8)]),
        config=config,
        experiment_manager=ExperimentManager(base_dir=str(tmp_path / "runs")),
        enable_hooks=False,
        enable_ema=False,
        device=torch.device("cpu"),
    )


def install_health_sequence(engine: TrainingEngineV2, outcomes) -> None:
    sequence = iter(outcomes)
    engine.health_checker.monitor_gradients = lambda: {}
    engine.health_checker.check_health = lambda: next(sequence)


def test_recovery_replays_batch_and_restores_counters_and_lr(tmp_path):
    engine = make_engine(tmp_path)
    install_health_sequence(engine, [(False, "synthetic failure"), (True, "ok")])

    summary = engine.fit(max_steps=1)

    assert summary["final_step"] == 1
    assert engine.current_step == 1
    assert engine.global_tokens == 2
    assert len(engine.model.seen_batches) == 2
    assert torch.equal(engine.model.seen_batches[0], engine.model.seen_batches[1])
    assert engine.optimizer.param_groups[0]["lr"] == pytest.approx(0.05)
    assert engine.scheduler.base_lrs[0] == pytest.approx(0.05)
    assert engine.scheduler.get_last_lr()[0] == pytest.approx(0.05)
    assert EngineState.RECOVERING in engine.fsm.history
    assert engine.fsm.current_state is EngineState.FINISHED


def test_three_consecutive_failures_are_terminal_and_wait_for_checkpoint(tmp_path):
    engine = make_engine(tmp_path, max_recovery_attempts=3)
    install_health_sequence(
        engine,
        [(False, "always unhealthy"), (False, "always unhealthy"), (False, "always unhealthy")],
    )
    wait_calls = 0
    original_wait = engine.checkpoint_mgr.wait_completion

    def tracked_wait():
        nonlocal wait_calls
        wait_calls += 1
        return original_wait()

    engine.checkpoint_mgr.wait_completion = tracked_wait

    with pytest.raises(RuntimeError, match="3 consecutive health-check failures"):
        engine.fit(max_steps=1)

    assert wait_calls >= 3
    assert engine.current_step == 0
    assert engine.global_tokens == 0
    assert engine.fsm.current_state is EngineState.ERROR
    with pytest.raises(RuntimeError, match="terminal"):
        engine.fsm.transition_to(EngineState.TRAIN)


def test_healthy_step_resets_consecutive_recovery_counter(tmp_path):
    engine = make_engine(tmp_path, max_recovery_attempts=2)
    install_health_sequence(
        engine,
        [
            (False, "first failure"),
            (True, "healthy"),
            (False, "new failure streak"),
            (True, "replayed healthy step"),
            (True, "healthy"),
        ],
    )

    summary = engine.fit(max_steps=2)

    assert summary["final_step"] == 2
    assert summary["total_tokens"] == 4
    assert engine.fsm.current_state is EngineState.FINISHED


def test_error_state_is_terminal():
    fsm = StateMachine(EngineState.ERROR)

    with pytest.raises(RuntimeError, match="terminal"):
        fsm.transition_to(EngineState.LOAD)
