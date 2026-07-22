"""Focused behavioral tests for real Top-K MoE routing."""

import torch
import torch.nn as nn

from src.transformer.config import M01Config
from src.transformer.moe import MoELayer


class _ScaledIdentity(nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


def _config(**overrides) -> M01Config:
    values = dict(
        d_model=8,
        n_heads=2,
        d_ff=16,
        num_experts=4,
        num_shared_experts=0,
        moe_top_k=2,
        capacity_factor=4.0,
        capacity_factor_warmup_steps=0,
        d_ff_routed=8,
        use_mla=False,
    )
    values.update(overrides)
    return M01Config(**values)


def _fixed_gate(moe: MoELayer) -> None:
    with torch.no_grad():
        moe.gate.weight.zero_()
        moe.gate.weight[0].fill_(2.0)
        moe.gate.weight[1].fill_(1.0)
        moe.gate.weight[2].fill_(-1.0)
        moe.gate.weight[3].fill_(-2.0)


def test_top2_accepts_two_distinct_assignments_per_token() -> None:
    moe = MoELayer(_config())
    _fixed_gate(moe)
    moe.eval()

    x = torch.ones(1, 5, moe.config.d_model)
    _ = moe(x)

    assert moe.assigned_experts.shape == (5, 2)
    assert moe.assigned_weights.shape == (5, 2)
    assert torch.all(moe.assigned_experts.ge(0))
    assert torch.all(moe.assigned_experts[:, 0] != moe.assigned_experts[:, 1])
    assert torch.allclose(moe.assigned_weights.sum(dim=-1), torch.ones(5))
    assert torch.equal(moe.accepted_assignments_per_rank, torch.tensor([5, 5]))
    assert moe.dropped_tokens.item() == 0


def test_output_accumulates_both_experts_with_accepted_weights() -> None:
    moe = MoELayer(_config())
    moe.experts = nn.ModuleList([_ScaledIdentity(float(i + 1)) for i in range(4)])
    _fixed_gate(moe)
    moe.eval()

    x = torch.ones(1, 3, moe.config.d_model)
    output = moe(x).view(-1, moe.config.d_model)
    expected_scale = (
        (moe.assigned_experts.to(x.dtype) + 1) * moe.assigned_weights
    ).sum(dim=-1, keepdim=True)

    assert torch.allclose(output, x.view_as(output) * expected_scale)


def test_limited_capacity_drops_assignments_and_renormalizes() -> None:
    moe = MoELayer(_config(capacity_factor=0.25))
    _fixed_gate(moe)
    moe.eval()

    x = torch.ones(1, 8, moe.config.d_model)
    output = moe(x)
    accepted_count = moe.assigned_experts.ge(0).sum(dim=-1)
    accepted_weight_sum = moe.assigned_weights.sum(dim=-1)

    assert torch.all(moe.expert_counts <= moe.capacity)
    assert torch.allclose(
        accepted_weight_sum[accepted_count > 0],
        torch.ones_like(accepted_weight_sum[accepted_count > 0]),
    )
    assert torch.equal(
        accepted_weight_sum[accepted_count == 0],
        torch.zeros_like(accepted_weight_sum[accepted_count == 0]),
    )
    assert torch.isfinite(moe.assigned_weights).all()
    assert torch.isfinite(output).all()


def test_gradients_reach_both_selected_experts() -> None:
    moe = MoELayer(_config())
    moe.experts = nn.ModuleList([_ScaledIdentity(float(i + 1)) for i in range(4)])
    _fixed_gate(moe)
    moe.eval()

    output = moe(torch.ones(1, 4, moe.config.d_model))
    output.sum().backward()

    selected = set(moe.assigned_experts.flatten().tolist()) - {-1}
    assert selected == {0, 1}
    assert all(moe.experts[index].scale.grad is not None for index in selected)
    assert all(moe.experts[index].scale.grad.abs().item() > 0 for index in selected)


def test_top1_keeps_single_assignment_and_unit_weight() -> None:
    moe = MoELayer(_config(moe_top_k=1))
    moe.experts = nn.ModuleList([_ScaledIdentity(float(i + 1)) for i in range(4)])
    _fixed_gate(moe)
    moe.eval()

    x = torch.ones(1, 4, moe.config.d_model)
    output = moe(x)

    assert moe.assigned_experts.shape == (4, 1)
    assert torch.equal(moe.assigned_experts, torch.zeros(4, 1, dtype=torch.long))
    assert torch.equal(moe.assigned_weights, torch.ones(4, 1))
    assert torch.allclose(output, x)
