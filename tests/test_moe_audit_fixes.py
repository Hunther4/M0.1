"""Regression tests for the critical MoE audit fixes."""

from unittest.mock import patch

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, TensorDataset

from src.engine_v2.engine import TrainingEngineV2
from src.model.lm import TransformerLM
from src.training.config import TrainingConfig
from src.training.loop import train
from src.training.moe_metrics import aux_loss_ema, detect_router_collapse
from src.transformer.config import M01Config
from src.transformer.moe import MoELayer


def _tiny_moe_config(**overrides) -> M01Config:
    values = dict(
        vocab_size=32,
        context_length=8,
        d_model=16,
        n_heads=2,
        d_ff=32,
        n_layers=2,
        num_experts=4,
        num_shared_experts=1,
        moe_top_k=1,
        num_dense_layers=0,
        use_mla=False,
    )
    values.update(overrides)
    return M01Config(**values)


def test_transformer_lm_aux_loss_is_mean_across_moe_layers() -> None:
    model = TransformerLM(_tiny_moe_config())
    model(torch.randint(0, 32, (1, 2)))

    model.blocks[0].ff.current_aux_loss = torch.tensor(2.0)
    model.blocks[1].ff.current_aux_loss = torch.tensor(4.0)

    assert torch.isclose(model.get_aux_loss(), torch.tensor(3.0))


def test_default_engine_router_aux_weight_compensates_for_layer_mean(tmp_path) -> None:
    config = _tiny_moe_config(n_layers=1)
    model = TransformerLM(config)
    x = torch.randint(0, 32, (2, 4))
    loader = DataLoader(TensorDataset(x, x.clone()), batch_size=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = LambdaLR(optimizer, lambda _: 1.0)

    from src.engine_v2.experiment import ExperimentManager

    engine = TrainingEngineV2(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=loader,
        config=TrainingConfig(log_interval=1, save_interval=0, val_interval=0),
        experiment_manager=ExperimentManager(base_dir=str(tmp_path)),
        enable_hooks=False,
        enable_ema=False,
        device=torch.device("cpu"),
    )

    aux_term = next(term for term in engine.loss_pipeline.terms if term.name == "RouterAuxLoss")
    assert aux_term.weight == 0.2
    z_term = next(term for term in engine.loss_pipeline.terms if term.name == "RouterZLoss")
    assert z_term.weight == 0.01


def test_gate_noise_is_used_only_for_routing_not_router_losses() -> None:
    config = _tiny_moe_config(n_layers=1)
    moe = MoELayer(config)
    x = torch.randn(1, 3, config.d_model)
    clean_logits = moe.gate(x.flatten(0, 1))
    clean_probs = torch.softmax(clean_logits, dim=-1)

    moe.train()
    torch.manual_seed(0)
    moe(x)

    assert torch.allclose(moe.gate_logits, clean_logits)
    assert torch.allclose(moe.gate_probs, clean_probs)
    assert torch.equal(
        moe.topk_indices,
        torch.topk(moe.routing_probs, k=config.moe_top_k, dim=-1).indices,
    )

    expected_aux = config.num_experts * torch.sum(
        moe.expert_mask.mean(dim=1) * clean_probs.mean(dim=0)
    )
    assert torch.allclose(moe.get_aux_loss(), expected_aux)
    expected_z = torch.mean(torch.logsumexp(clean_logits, dim=-1) ** 2)
    assert torch.allclose(moe.get_z_loss(), expected_z)


def test_collapse_detection_uses_dead_expert_ratio() -> None:
    mostly_active = torch.tensor([0, 10, 20, 5])
    mostly_dead = torch.tensor([0, 0, 20, 5])

    stop, counter = detect_router_collapse(mostly_active, 0, 5, expert_ratio=0.3)
    assert (stop, counter) == (False, 0)

    stop, counter = detect_router_collapse(mostly_dead, 0, 5, expert_ratio=0.3)
    assert (stop, counter) == (False, 1)


def test_shared_loop_reads_moe_collapse_config() -> None:
    class FakeMoEModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.projection = nn.Linear(4, 4)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.projection(inputs)

        def get_moe_metrics(self) -> dict:
            return {
                "layer_0/histogram": [0, 0, 1, 1],
                "global/n_layers_moe": 1,
            }

    model = FakeMoEModel()
    data = TensorDataset(torch.randn(8, 4), torch.randint(0, 4, (8,)))
    result = train(
        model=model,
        dataloader=DataLoader(data, batch_size=2),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        criterion=nn.CrossEntropyLoss(),
        steps=10,
        device=torch.device("cpu"),
        log_interval=1,
        config=TrainingConfig(
            moe_collapse_consecutive_steps=2,
            moe_collapse_expert_ratio=0.3,
        ),
    )

    assert result["steps_completed"] < 10
    assert "collapse" in result["stop_reason"].lower()


def test_engine_tracks_aux_loss_ema(tmp_path) -> None:
    config = _tiny_moe_config(n_layers=1)
    model = TransformerLM(config)
    x = torch.randint(0, 32, (2, 4))
    loader = DataLoader(TensorDataset(x, x.clone()), batch_size=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = LambdaLR(optimizer, lambda _: 1.0)

    from src.engine_v2.experiment import ExperimentManager

    engine = TrainingEngineV2(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=loader,
        config=TrainingConfig(log_interval=1, save_interval=0, val_interval=0),
        experiment_manager=ExperimentManager(base_dir=str(tmp_path)),
        enable_hooks=False,
        enable_ema=False,
        device=torch.device("cpu"),
    )

    with patch(
        "src.training.moe_metrics.aux_loss_ema",
        wraps=aux_loss_ema,
    ) as ema:
        engine.fit(max_steps=1)

    assert ema.call_count == 1


def test_engine_uses_collapse_defaults_when_config_is_none(tmp_path) -> None:
    class FakeMoEModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.parameter = nn.Parameter(torch.ones(1))

        def get_moe_metrics(self) -> dict:
            return {"layer_0/histogram": [0, 0, 1, 1]}

        def get_aux_loss(self) -> torch.Tensor:
            return torch.tensor(1.0)

    model = FakeMoEModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = LambdaLR(optimizer, lambda _: 1.0)

    from src.engine_v2.experiment import ExperimentManager

    engine = TrainingEngineV2(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=DataLoader(TensorDataset(torch.ones(1, 1)), batch_size=1),
        config=None,
        experiment_manager=ExperimentManager(base_dir=str(tmp_path)),
        enable_hooks=False,
        enable_ema=False,
        device=torch.device("cpu"),
    )

    with patch(
        "src.training.moe_metrics.detect_router_collapse",
        wraps=detect_router_collapse,
    ) as detector:
        engine._log_moe_metrics(step=10, log_data={})

    detector.assert_called_once()
    assert detector.call_args.args[2] == 50
    assert detector.call_args.kwargs["expert_ratio"] == 0.3


def test_gate_noise_is_training_only() -> None:
    config = _tiny_moe_config(n_layers=1)
    moe = MoELayer(config)
    with torch.no_grad():
        moe.gate.weight.zero_()
    x = torch.ones(1, 2, config.d_model)

    moe.train()
    torch.manual_seed(0)
    moe(x)
    assert torch.allclose(moe.gate_logits, torch.zeros_like(moe.gate_logits))
    assert not torch.allclose(
        moe.routing_probs,
        torch.full_like(moe.routing_probs, 1 / config.num_experts),
    )

    moe.eval()
    moe(x)
    assert torch.allclose(moe.gate_logits, torch.zeros_like(moe.gate_logits))
    assert torch.allclose(
        moe.routing_probs,
        torch.full_like(moe.routing_probs, 1 / config.num_experts),
    )
