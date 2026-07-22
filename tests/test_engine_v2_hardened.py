"""Comprehensive Hardened Test Suite for TrainingEngine v2."""

import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

from src.engine_v2.fsm import StateMachine, EngineState
from src.engine_v2.bus import EventBus, EngineEvent
from src.engine_v2.loss_pipeline import LossPipeline, CrossEntropyLossTerm, RouterAuxLossTerm, RouterZLossTerm
from src.engine_v2.metrics import MetricRegistry
from src.engine_v2.checkpoint_v2 import AsyncCheckpointManagerV2
from src.engine_v2.amp import AMPContext
from src.engine_v2.ema import EMA
from src.engine_v2.health import HealthChecker
from src.engine_v2.engine import TrainingEngineV2
from src.transformer.config import M01Config
from src.model.lm import TransformerLM


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(16, 100)

    def forward(self, x):
        return self.fc(x.float())


class MismatchedWeightModel(nn.Module):
    def __init__(self, rows: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(rows, 3))
        self.bias = nn.Parameter(torch.zeros(2))


@pytest.mark.gpu
def test_loss_pipeline_registration():
    """Verify dynamic registration in LossPipeline."""
    pipeline = LossPipeline([CrossEntropyLossTerm(vocab_size=100)])
    assert len(pipeline.terms) == 1
    pipeline.register(RouterZLossTerm(weight=0.01))
    assert len(pipeline.terms) == 2


@pytest.mark.gpu
def test_ema_shadow_and_restore():
    """Verify EMA shadow parameter application and restoration."""
    model = DummyModel()
    ema = EMA(model, decay=0.9)
    initial_val = model.fc.weight.data.clone()

    # Mutate weights
    model.fc.weight.data += 1.0
    ema.update()

    # Apply shadow
    ema.apply_shadow()
    assert not torch.equal(model.fc.weight.data, initial_val)

    # Restore
    ema.restore()
    assert torch.equal(model.fc.weight.data, initial_val + 1.0)


@pytest.mark.gpu
def test_health_checker_detection():
    """Verify HealthChecker catches NaNs in parameters."""
    model = DummyModel()
    checker = HealthChecker(model)
    healthy, _ = checker.check_health()
    assert healthy

    # Inject NaN
    model.fc.weight.data[0, 0] = float("nan")
    healthy, reason = checker.check_health()
    assert not healthy
    assert "NaN parameter" in reason


@pytest.mark.gpu
def test_canonical_checkpoint_sha256_integrity(tmp_path):
    """Verify canonical checkpoint saving, SHA256 calculation, and restoration."""
    mgr = AsyncCheckpointManagerV2(str(tmp_path))
    state = {"step": 42, "model_state": {}}
    mgr.save_canonical_async(state)
    mgr.wait_completion()

    assert (tmp_path / "checkpoint.pt").exists()
    assert (tmp_path / "checkpoint.pt.sha256").exists()

    restored = mgr.load_canonical()
    assert restored["step"] == 42


def test_resume_loads_compatible_weights_when_moe_shapes_changed(tmp_path, monkeypatch, capsys):
    """Resume keeps compatible state while reinitializing changed MoE weights."""
    model = nn.Linear(3, 2)
    model.config = SimpleNamespace(
        vocab_size=16,
        n_layers=1,
        d_model=3,
        num_experts=4,
        num_shared_experts=1,
        moe_top_k=2,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    engine = TrainingEngineV2.__new__(TrainingEngineV2)
    engine.model = model
    engine.optimizer = optimizer
    engine.scheduler = scheduler
    engine.ema = None
    engine.current_step = 0
    engine.global_tokens = 0

    original_weight = model.weight.detach().clone()
    expected_bias = torch.tensor([3.0, -2.0])
    checkpoint_state = {
        "step": 17,
        "model_config": {
            "vocab_size": 16,
            "n_layers": 1,
            "d_model": 3,
            "num_experts": 4,
            "num_shared_experts": 1,
            "moe_top_k": 1,
        },
        "model_state": {
            "weight": torch.ones(3, 3),
            "bias": expected_bias,
        },
    }
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.touch()
    monkeypatch.setattr("src.engine_v2.engine.safe_load_checkpoint", lambda _: checkpoint_state)

    assert engine.resume(str(checkpoint_path)) == 17
    assert torch.equal(model.weight, original_weight)
    assert torch.equal(model.bias, expected_bias)
    output = capsys.readouterr().out
    assert "1 keys loaded, 1 MoE keys skipped" in output
    assert "weight: checkpoint [3, 3] vs model [2, 3]" in output


def test_resume_loads_all_weights_when_checkpoint_shapes_match(tmp_path, monkeypatch, capsys):
    """Resume restores every model weight without a partial-load report when shapes match."""
    model = nn.Linear(3, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    engine = TrainingEngineV2.__new__(TrainingEngineV2)
    engine.model = model
    engine.optimizer = optimizer
    engine.scheduler = scheduler
    engine.ema = None
    engine.current_step = 0
    engine.global_tokens = 0

    expected_weight = torch.full_like(model.weight, 5.0)
    expected_bias = torch.tensor([1.0, 2.0])
    checkpoint_state = {
        "step": 23,
        "model_config": {},
        "model_state": {"weight": expected_weight, "bias": expected_bias},
    }
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.touch()
    monkeypatch.setattr("src.engine_v2.engine.safe_load_checkpoint", lambda _: checkpoint_state)

    assert engine.resume(str(checkpoint_path)) == 23
    assert torch.equal(model.weight, expected_weight)
    assert torch.equal(model.bias, expected_bias)
    assert "Partial load" not in capsys.readouterr().out


def test_resume_reinitializes_incompatible_optimizer_and_ema_state(tmp_path, monkeypatch):
    """Resume keeps fresh optimizer and EMA state for changed expert parameters."""
    source_model = MismatchedWeightModel(rows=3)
    source_optimizer = torch.optim.SGD(source_model.parameters(), lr=0.1, momentum=0.9)
    for parameter in source_model.parameters():
        parameter.grad = torch.ones_like(parameter)
    source_optimizer.step()

    model = MismatchedWeightModel(rows=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    engine = TrainingEngineV2.__new__(TrainingEngineV2)
    engine.model = model
    engine.optimizer = optimizer
    engine.scheduler = scheduler
    engine.ema = EMA(model)
    engine.current_step = 0
    engine.global_tokens = 0

    checkpoint_state = {
        "step": 11,
        "model_config": {},
        "model_state": source_model.state_dict(),
        "optimizer_state": source_optimizer.state_dict(),
        "ema_state": EMA(source_model).state_dict(),
    }
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.touch()
    monkeypatch.setattr("src.engine_v2.engine.safe_load_checkpoint", lambda _: checkpoint_state)

    engine.resume(str(checkpoint_path))

    assert "momentum_buffer" not in optimizer.state[model.weight]
    assert torch.equal(optimizer.state[model.bias]["momentum_buffer"], torch.ones_like(model.bias))
    assert engine.ema.shadow["weight"].shape == model.weight.shape
    assert torch.equal(engine.ema.shadow["bias"], source_model.bias)


def test_resume_restores_optimizer_and_ema_state_when_shapes_match(tmp_path, monkeypatch, capsys):
    """Resume preserves optimizer and EMA state when every parameter shape matches."""
    source_model = MismatchedWeightModel(rows=2)
    source_optimizer = torch.optim.SGD(source_model.parameters(), lr=0.1, momentum=0.9)
    for parameter in source_model.parameters():
        parameter.grad = torch.ones_like(parameter)
    source_optimizer.step()

    model = MismatchedWeightModel(rows=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    engine = TrainingEngineV2.__new__(TrainingEngineV2)
    engine.model = model
    engine.optimizer = optimizer
    engine.scheduler = scheduler
    engine.ema = EMA(model)
    engine.current_step = 0
    engine.global_tokens = 0

    checkpoint_state = {
        "step": 12,
        "model_config": {},
        "model_state": source_model.state_dict(),
        "optimizer_state": source_optimizer.state_dict(),
        "ema_state": EMA(source_model).state_dict(),
    }
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.touch()
    monkeypatch.setattr("src.engine_v2.engine.safe_load_checkpoint", lambda _: checkpoint_state)

    engine.resume(str(checkpoint_path))

    assert torch.equal(optimizer.state[model.weight]["momentum_buffer"], torch.ones_like(model.weight))
    assert torch.equal(engine.ema.shadow["weight"], source_model.weight)
    assert "Reinitialized" not in capsys.readouterr().out


def test_resume_reinitializes_optimizer_when_parameter_groups_changed(tmp_path, monkeypatch, capsys):
    """Resume retains loaded weights when the checkpoint optimizer groups are incompatible."""
    source_model = MismatchedWeightModel(rows=2)
    source_optimizer = torch.optim.SGD(
        [{"params": [source_model.weight]}, {"params": [source_model.bias]}],
        lr=0.1,
        momentum=0.9,
    )
    for parameter in source_model.parameters():
        parameter.grad = torch.ones_like(parameter)
    source_optimizer.step()

    model = MismatchedWeightModel(rows=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    engine = TrainingEngineV2.__new__(TrainingEngineV2)
    engine.model = model
    engine.optimizer = optimizer
    engine.scheduler = scheduler
    engine.ema = None
    engine.current_step = 0
    engine.global_tokens = 0

    checkpoint_state = {
        "step": 13,
        "model_config": {},
        "model_state": source_model.state_dict(),
        "optimizer_state": source_optimizer.state_dict(),
    }
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.touch()
    monkeypatch.setattr("src.engine_v2.engine.safe_load_checkpoint", lambda _: checkpoint_state)

    assert engine.resume(str(checkpoint_path)) == 13
    assert optimizer.state == {}
    assert "Skipped optimizer state" in capsys.readouterr().out


@pytest.mark.gpu
def test_full_engine_v2_run(tmp_path):
    """Verify TrainingEngineV2 end-to-end execution on GPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = M01Config(vocab_size=256, d_model=128, n_heads=4, n_layers=2)
    model = TransformerLM(model_config)

    x = torch.randint(0, 256, (8, 32))
    y = torch.randint(0, 256, (8, 32))
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 1.0)

    engine = TrainingEngineV2(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=loader,
        device=device,
    )

    summary = engine.fit(max_steps=5)
    assert summary["final_step"] == 5
