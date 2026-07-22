"""Unit tests for gradient health monitoring."""

import pytest
import torch
import torch.nn as nn

from src.engine_v2.health import HealthChecker


class GradientModel(nn.Module):
    """Small model with independently controllable parameter gradients."""

    def __init__(self):
        super().__init__()
        self.first = nn.Parameter(torch.zeros(3))
        self.second = nn.Parameter(torch.zeros(3))


def test_monitor_gradients_collects_stats_without_per_layer_item_calls(monkeypatch):
    """Gradient statistics are transferred only after GPU-side aggregation."""
    model = GradientModel()
    model.first.grad = torch.tensor([1.0, -2.0, 0.0])
    model.second.grad = torch.tensor([3.0, -4.0, 0.0])

    def fail_on_item(*_args, **_kwargs):
        raise AssertionError("monitor_gradients must not call Tensor.item() per layer")

    monkeypatch.setattr(torch.Tensor, "item", fail_on_item)

    stats = HealthChecker(model).monitor_gradients()

    assert stats["grad_mean"] == pytest.approx(-1 / 3)
    assert stats["grad_std"] == pytest.approx(2.4221203)
    assert stats["grad_max"] == 3.0
    assert stats["grad_min"] == -4.0
    assert stats["grad_sparsity"] == pytest.approx(2 / 6)
    assert stats["largest_grad_norm"] == 5.0
    assert stats["layer_with_biggest_grad"] == "second"


def test_monitor_gradients_returns_empty_stats_without_gradients():
    """Models without trainable gradients do not trigger summary aggregation."""
    model = GradientModel()

    assert HealthChecker(model).monitor_gradients() == {}
