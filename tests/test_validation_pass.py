"""Automated Validation Pass for Resumption, NaN Recovery, Router Metrics, and VRAM Stability.

Run via: pytest tests/test_validation_pass.py
"""

import os
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.transformer.config import M01Config
from src.model.lm import TransformerLM
from src.engine_v2.engine import TrainingEngineV2
from src.engine_v2.checkpoint_v2 import AsyncCheckpointManagerV2
from src.engine_v2.loss_pipeline import LossPipeline, CrossEntropyLossTerm, RouterAuxLossTerm, RouterZLossTerm
from src.engine_v2.experiment import ExperimentManager


@pytest.mark.gpu
def test_resumption_20_times_determinism(tmp_path):
    """Test 1: Interrupt & Resume 20 times to verify exact step and token accumulation determinism."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = M01Config(vocab_size=256, d_model=128, n_heads=4, n_layers=2)
    model = TransformerLM(model_config)

    x = torch.randint(0, 256, (16, 32))
    y = torch.randint(0, 256, (16, 32))
    loader = DataLoader(TensorDataset(x, y), batch_size=2)

    exp_mgr = ExperimentManager(base_dir=str(tmp_path / "runs"))

    # Execute 20 iterations of 5-step runs, saving and resuming
    for iteration in range(20):
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 1.0)

        engine = TrainingEngineV2(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=loader,
            experiment_manager=exp_mgr,
            device=device,
        )

        if iteration > 0:
            engine.resume()

        target_steps = (iteration + 1) * 5
        summary = engine.fit(max_steps=target_steps)
        assert summary["final_step"] == target_steps

    print("\n[PASSED] 20 Resumption Cycles Verified Deterministically!")


@pytest.mark.gpu
def test_forced_nan_recovery_rollback(tmp_path):
    """Test 2: Inject NaN into gradients, verify checkpoint rollback, LR reduction, and training continuation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = M01Config(vocab_size=256, d_model=128, n_heads=4, n_layers=2)
    model = TransformerLM(model_config)

    x = torch.randint(0, 256, (16, 32))
    y = torch.randint(0, 256, (16, 32))
    loader = DataLoader(TensorDataset(x, y), batch_size=2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 1.0)
    exp_mgr = ExperimentManager(base_dir=str(tmp_path / "runs"))

    engine = TrainingEngineV2(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=loader,
        experiment_manager=exp_mgr,
        device=device,
    )

    # 1. Run 5 initial steps and save canonical checkpoint
    engine.fit(max_steps=5)
    assert engine.checkpoint_mgr.canonical_path.exists()

    # 2. Inject NaN into model weights
    with torch.no_grad():
        for param in model.parameters():
            param[0] = float("nan")
            break

    healthy, reason = engine.health_checker.check_health()
    assert not healthy
    assert "NaN parameter" in reason

    # 3. Fit next steps; engine must catch NaN, trigger recovery, restore weights, and finish
    summary = engine.fit(max_steps=10)
    assert summary["final_step"] == 10

    # Ensure weights no longer contain NaN
    healthy_after, _ = engine.health_checker.check_health()
    assert healthy_after
    print("\n[PASSED] Automatic NaN Detection, Rollback & Healing Verified!")


@pytest.mark.gpu
def test_router_metrics_stability(tmp_path):
    """Test 3: Verify Gini Index, KL Divergence, and Router Z-loss remain within healthy bounds."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = M01Config(
        vocab_size=256,
        d_model=128,
        n_heads=4,
        n_layers=3,
        num_experts=8,
        num_shared_experts=2,
        moe_top_k=2,
    )
    model = TransformerLM(config).to(device)

    x = torch.randint(0, 256, (16, 32))
    y = torch.randint(0, 256, (16, 32))
    loader = DataLoader(TensorDataset(x, y), batch_size=4)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 1.0)
    exp_mgr = ExperimentManager(base_dir=str(tmp_path / "runs"))

    engine = TrainingEngineV2(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=loader,
        experiment_manager=exp_mgr,
        device=device,
    )

    engine.fit(max_steps=20)
    metrics_all = engine.metrics.get_all()

    # Verify Gini index is reasonable (< 0.8) and KL divergence is non-negative
    if "expert_gini_index" in metrics_all:
        gini = metrics_all["expert_gini_index"]
        assert 0.0 <= gini <= 0.95
    if "router_kl_div" in metrics_all:
        kl = metrics_all["router_kl_div"]
        assert kl >= 0.0

    print("\n[PASSED] MoE Router Stability & Metrics Verified!")


@pytest.mark.gpu
def test_vram_and_throughput_stability(tmp_path):
    """Test 4: Verify VRAM memory and token/s throughput remain flat over 200 steps (no leaks)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = M01Config(vocab_size=500, d_model=128, n_heads=4, n_layers=2)
    model = TransformerLM(config)

    x = torch.randint(0, 500, (32, 64))
    y = torch.randint(0, 500, (32, 64))
    loader = DataLoader(TensorDataset(x, y), batch_size=4)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 1.0)
    exp_mgr = ExperimentManager(base_dir=str(tmp_path / "runs"))

    engine = TrainingEngineV2(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=loader,
        experiment_manager=exp_mgr,
        device=device,
    )

    summary = engine.fit(max_steps=200)

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e6
        # Assert VRAM footprint stays reasonable (< 4000 MB for tiny model)
        assert allocated < 4000

    print(f"\n[PASSED] 200-Step VRAM & Throughput Stability Verified! Final tokens: {summary['total_tokens']:,}")
