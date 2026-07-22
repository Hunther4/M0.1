"""Tests for RMSNorm layer.

RMSNorm requirements:
- Output shape MUST be (B, S, D) for input (B, S, D)
- Gradients MUST flow through the layer (loss.sum().backward())
- gamma.grad MUST exist and be non-zero (gamma is learnable)
- eps MUST prevent division by zero, even with zero input
"""

import torch
import pytest
from src.model.rms_norm import RMSNorm


class TestRMSNormShape:
    """RMSNorm output shape invariants."""

    def test_output_shape_basic(self) -> None:
        """Output MUST match input shape (B, S, D)."""
        d_model = 640
        rms = RMSNorm(d_model)
        x = torch.randn(2, 5, d_model)
        out = rms(x)
        assert out.shape == (2, 5, d_model), (
            f"Expected (2, 5, {d_model}), got {out.shape}"
        )

    def test_output_shape_various_dims(self) -> None:
        """Output MUST match input shape for various batch and seq dims."""
        d_model = 640
        rms = RMSNorm(d_model)
        shapes = [(1, 1, 640), (4, 128, 640), (8, 1024, 640)]
        for batch, seq, d in shapes:
            x = torch.randn(batch, seq, d)
            out = rms(x)
            assert out.shape == (batch, seq, d), (
                f"Expected ({batch}, {seq}, {d}), got {out.shape}"
            )


class TestRMSNormGradientFlow:
    """Gradient flow through RMSNorm."""

    def test_backward_passes(self) -> None:
        """loss.sum().backward() MUST succeed (no NaN, no exception)."""
        d_model = 640
        rms = RMSNorm(d_model)
        x = torch.randn(2, 5, d_model, requires_grad=True)
        out = rms(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None, "Input gradient is None"
        assert x.grad.shape == x.shape, (
            f"Input grad shape {x.grad.shape} != {x.shape}"
        )
        assert torch.isfinite(x.grad).all(), "Input grad contains NaN or Inf"

    def test_gamma_is_learnable(self) -> None:
        """RMSNorm.gamma MUST be a learnable parameter with non-zero gradient."""
        d_model = 640
        rms = RMSNorm(d_model)
        assert isinstance(rms.gamma, torch.nn.Parameter), (
            "gamma is not a Parameter"
        )
        assert rms.gamma.requires_grad, "gamma does not require grad"

        x = torch.randn(2, 5, d_model)
        out = rms(x)
        loss = out.sum()
        loss.backward()

        assert rms.gamma.grad is not None, "gamma.grad is None"
        assert rms.gamma.grad.shape == (d_model,), (
            f"gamma.grad shape {rms.gamma.grad.shape} != ({d_model},)"
        )
        # gamma should receive meaningful gradient (non-zero for varied input)
        assert rms.gamma.grad.abs().sum().item() > 0, (
            "gamma.grad is all zero — no learning happening"
        )


class TestRMSNormEps:
    """EPS parameter behavior."""

    def test_eps_prevents_nan_with_zero_input(self) -> None:
        """With zero input and any eps>0, output MUST be zero (not NaN)."""
        d_model = 640
        rms = RMSNorm(d_model, eps=1e-5)
        x = torch.zeros(2, 5, d_model)
        out = rms(x)
        assert not torch.isnan(out).any(), "Output contains NaN with zero input"
        assert not torch.isinf(out).any(), "Output contains Inf with zero input"
        # When x is all zeros: sqrt(mean(0) + eps) = sqrt(eps), x / sqrt(eps) * gamma = 0
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-7), (
            "RMSNorm(zeros) should be zero"
        )

    def test_eps_very_small_still_stable(self) -> None:
        """Even with tiny eps=1e-10, RMSNorm MUST NOT produce NaN."""
        d_model = 640
        rms = RMSNorm(d_model, eps=1e-10)
        x = torch.randn(2, 5, d_model)
        out = rms(x)
        assert not torch.isnan(out).any(), (
            "Output contains NaN with eps=1e-10"
        )
        assert not torch.isinf(out).any(), (
            "Output contains Inf with eps=1e-10"
        )
        # Gradient should still flow
        loss = out.sum()
        loss.backward()
        assert rms.gamma.grad is not None, "gamma.grad is None after backward"
        assert torch.isfinite(rms.gamma.grad).all(), (
            "gamma.grad is not finite with tiny eps"
        )


class TestRMSNormGammaInit:
    """Gamma parameter initialization."""

    def test_gamma_initialized_to_ones(self) -> None:
        """gamma MUST be initialized to all ones (RMSNorm default)."""
        d_model = 640
        rms = RMSNorm(d_model)
        expected = torch.ones(d_model)
        assert torch.allclose(rms.gamma.data, expected), (
            f"gamma initial values deviate from 1: {rms.gamma.data[:5]}..."
        )

    def test_gamma_has_correct_size(self) -> None:
        """gamma MUST have exactly d_model elements."""
        d_model = 640
        rms = RMSNorm(d_model)
        assert rms.gamma.numel() == d_model, (
            f"gamma has {rms.gamma.numel()} elements, expected {d_model}"
        )
        assert rms.gamma.shape == (d_model,), (
            f"gamma shape {rms.gamma.shape} != ({d_model},)"
        )


class TestRMSNormNumerics:
    """Numerical correctness of RMSNorm."""

    def test_normalized_output_magnitude(self) -> None:
        """RMSNorm output MUST have RMS ~= 1 along the last dim (before gamma)."""
        d_model = 640
        rms = RMSNorm(d_model)
        x = torch.randn(2, 5, d_model)

        # Run forward
        out = rms(x)

        # Compute RMS of each position vector (before gamma scaling)
        # out = (x / sqrt(mean(x**2) + eps)) * gamma
        # So out / gamma should have RMS ≈ 1
        rms_per_pos = out / rms.gamma  # undo gamma scaling
        actual_rms = torch.sqrt(
            torch.mean(rms_per_pos ** 2, dim=-1)
        )
        # Should be close to 1 (within tolerance)
        assert torch.allclose(actual_rms, torch.ones_like(actual_rms), atol=1e-4), (
            f"RMS deviates from 1: range [{actual_rms.min().item():.6f}, "
            f"{actual_rms.max().item():.6f}]"
        )

    def test_shape_invariant_different_d_model(self) -> None:
        """RMSNorm MUST work with various d_model sizes."""
        for d_model in [64, 128, 640, 1024]:
            rms = RMSNorm(d_model)
            x = torch.randn(2, 4, d_model)
            out = rms(x)
            assert out.shape == (2, 4, d_model), (
                f"For d_model={d_model}: expected (2, 4, {d_model}), got {out.shape}"
            )
            # Gradient flow must succeed
            out.sum().backward()
            assert rms.gamma.grad is not None, (
                f"gamma.grad is None for d_model={d_model}"
            )

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_mixed_precision_output_preserves_input_dtype(self, dtype) -> None:
        """An FP32 gamma MUST NOT promote mixed-precision activations."""
        rms = RMSNorm(64)
        x = torch.randn(2, 4, 64, dtype=dtype)

        out = rms(x)

        assert rms.gamma.dtype == torch.float32
        assert out.dtype == dtype
        assert torch.isfinite(out).all()
