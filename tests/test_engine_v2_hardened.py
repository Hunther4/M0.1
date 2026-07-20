"""Comprehensive Hardened Test Suite for TrainingEngine v2."""

import os
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
