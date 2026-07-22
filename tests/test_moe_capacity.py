"""Behavioral tests for capacity-limited MoE routing."""

import pytest
import torch
import torch.nn as nn

from src.transformer.config import M01Config
from src.transformer.moe import MoELayer
from src.model.lm import TransformerLM


def _tiny_moe_config(**overrides) -> M01Config:
    values = dict(
        d_model=16,
        n_heads=2,
        d_ff=32,
        num_experts=4,
        num_shared_experts=1,
        moe_top_k=2,
        d_ff_shared=16,
        d_ff_routed=16,
        use_mla=False,
    )
    values.update(overrides)
    return M01Config(**values)


def _prefer_first_two_experts(moe: MoELayer) -> None:
    with torch.no_grad():
        moe.gate.weight.zero_()
        moe.gate.weight[0].fill_(3.0)
        moe.gate.weight[1].fill_(2.0)


def test_capacity_factor_default() -> None:
    config = _tiny_moe_config()
    moe = MoELayer(config)
    x = torch.randn(1, 3, config.d_model)

    moe.eval()
    output = moe(x)

    assert config.capacity_factor == 1.25
    assert config.capacity_factor_warmup_steps == 2000
    assert config.capacity_factor_warmup_start == 2.0
    assert output.shape == x.shape


@pytest.mark.parametrize(
    ("step", "expected_capacity_factor"),
    [(0, 2.0), (1000, 1.625), (2000, 1.25), (5000, 1.25)],
)
def test_capacity_factor_warmup_anneals_to_target(
    step: int,
    expected_capacity_factor: float,
) -> None:
    config = _tiny_moe_config()
    moe = MoELayer(config)

    moe.set_step(step)

    assert moe.effective_capacity_factor == pytest.approx(expected_capacity_factor)


def test_capacity_uses_warmed_up_capacity_factor() -> None:
    config = _tiny_moe_config(
        capacity_factor=1.0,
        capacity_factor_warmup_start=2.0,
        capacity_factor_warmup_steps=4,
    )
    moe = MoELayer(config)
    x = torch.ones(1, 4, config.d_model)

    _prefer_first_two_experts(moe)
    moe.eval()
    moe.set_step(0)
    _ = moe(x)
    warmup_capacity = moe.capacity
    moe.set_step(4)
    _ = moe(x)

    assert warmup_capacity == 4
    assert moe.capacity == 2


def test_capacity_warmup_can_be_disabled() -> None:
    config = _tiny_moe_config(
        capacity_factor=1.0,
        capacity_factor_warmup_start=2.0,
        capacity_factor_warmup_steps=0,
    )
    moe = MoELayer(config)

    moe.set_step(0)

    assert moe.effective_capacity_factor == 1.0


def test_transformer_lm_propagates_moe_step_to_all_moe_blocks() -> None:
    config = _tiny_moe_config(n_layers=3, num_dense_layers=2)
    model = TransformerLM(config)

    model.set_moe_step(1000)

    moe_layers = [
        block.ff for block in model.blocks if hasattr(block.ff, "effective_capacity_factor")
    ]
    assert len(moe_layers) == 1
    assert moe_layers[0].effective_capacity_factor == pytest.approx(1.625)


@pytest.mark.parametrize(
    ("num_tokens", "expected_capacity"),
    [(1, 1), (4, 2), (10, 6)],
)
def test_capacity_factor_limits(num_tokens: int, expected_capacity: int) -> None:
    config = _tiny_moe_config()
    moe = MoELayer(config)
    x = torch.ones(1, num_tokens, config.d_model)

    _prefer_first_two_experts(moe)
    moe.eval()
    _ = moe(x)

    assert moe.capacity == expected_capacity
    assert torch.all(moe.expert_mask.sum(dim=1) <= expected_capacity)


def test_capacity_prefers_higher_routing_weight() -> None:
    config = _tiny_moe_config(capacity_factor=0.5)
    moe = MoELayer(config)
    x = torch.zeros(1, 4, config.d_model)
    x[0, :, 0] = torch.tensor([1.0, 2.0, 3.0, 4.0])

    with torch.no_grad():
        moe.gate.weight.zero_()
        moe.gate.weight[0, 0] = 1.0
        moe.gate.weight[1, 0] = 0.0
        moe.gate.weight[2, 0] = -1.0
        moe.gate.weight[3, 0] = -2.0

    moe.eval()
    _ = moe(x)

    assert moe.capacity == 1
    assert torch.equal(torch.where(moe.expert_mask[0].bool())[0], torch.tensor([3]))


def test_capacity_no_tokens_dropped() -> None:
    config = _tiny_moe_config(capacity_factor=2.0)
    moe = MoELayer(config)
    x = torch.ones(1, 8, config.d_model)

    _prefer_first_two_experts(moe)
    moe.eval()
    output = moe(x)

    assert output.shape == x.shape
    assert torch.equal(
        moe.expert_mask.sum(dim=0),
        torch.full(
            (x.size(0) * x.size(1),),
            config.moe_top_k,
            dtype=x.dtype,
        ),
    )


def test_capacity_tokens_filtered_keep_shared_output() -> None:
    config = _tiny_moe_config(capacity_factor=0.25)
    moe = MoELayer(config)
    moe.shared_experts = nn.ModuleList([nn.Identity()])
    moe.experts = nn.ModuleList([nn.Identity() for _ in range(config.num_experts)])
    x = torch.ones(1, 8, config.d_model)

    _prefer_first_two_experts(moe)
    moe.eval()
    output = moe(x)

    dropped = moe.expert_mask.sum(dim=0).eq(0)
    assert dropped.any()
    assert torch.allclose(output.view(-1, config.d_model)[dropped], x.view(-1, config.d_model)[dropped])
    assert torch.isfinite(output).all()
