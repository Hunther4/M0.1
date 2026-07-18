"""Tests for TransformerBlock layer.

TransformerBlock requirements:
- Output shape MUST be (B, S, D) for input (B, S, D)
- Pre-norm residual flow: x = x + attn(norm1(x)), then x = x + ff(norm2(x))
- Conditional FF/MoE: num_experts=1 uses FeedForward, num_experts>1 uses MoELayer
- Gradients MUST flow through the layer (loss.sum().backward())
"""

import torch
import pytest
from src.transformer.config import M01Config
from src.model.block import TransformerBlock


@pytest.fixture
def base_config() -> M01Config:
    """Default config with num_experts=1 (dense FF)."""
    config = M01Config(
        vocab_size=32768,
        context_length=512,
        d_model=128,  # smaller for faster tests
        n_heads=4,
        d_ff=344,
        n_layers=2,
        num_experts=1,
    )
    return config


@pytest.fixture
def moe_config() -> M01Config:
    """Config with num_experts=4 (MoE)."""
    config = M01Config(
        vocab_size=32768,
        context_length=512,
        d_model=128,
        n_heads=4,
        d_ff=344,
        n_layers=2,
        num_experts=4,
    )
    return config


@pytest.fixture
def sample_input() -> torch.Tensor:
    """Random input tensor (B, S, D)."""
    return torch.randn(2, 8, 128)


class TestBlockShape:
    """TransformerBlock output shape invariants."""

    def test_output_shape_with_ff(self, base_config, sample_input) -> None:
        """With num_experts=1 (dense FF), output MUST match input shape (B, S, D)."""
        block = TransformerBlock(base_config)
        block.eval()
        out = block(sample_input)
        assert out.shape == sample_input.shape, (
            f"Expected {sample_input.shape}, got {out.shape}"
        )

    def test_output_shape_with_moe(self, moe_config, sample_input) -> None:
        """With num_experts=4 (MoE), output MUST match input shape (B, S, D)."""
        block = TransformerBlock(moe_config)
        block.eval()
        out = block(sample_input)
        assert out.shape == sample_input.shape, (
            f"Expected {sample_input.shape}, got {out.shape}"
        )

    def test_output_shape_various_dims(self, base_config) -> None:
        """Output MUST match input shape for various batch and seq dims."""
        block = TransformerBlock(base_config)
        block.eval()
        shapes = [(1, 1, 128), (4, 16, 128), (8, 64, 128)]
        for batch, seq, d in shapes:
            x = torch.randn(batch, seq, d)
            out = block(x)
            assert out.shape == (batch, seq, d), (
                f"Expected ({batch}, {seq}, {d}), got {out.shape}"
            )


class TestBlockResidual:
    """Residual connection behavior."""

    def test_residual_preserves_shape(self, base_config, sample_input) -> None:
        """Residual connections MUST preserve shape across the block."""
        block = TransformerBlock(base_config)
        block.eval()
        out = block(sample_input)
        assert out.shape == sample_input.shape, (
            f"Residual output shape {out.shape} != input shape {sample_input.shape}"
        )

    def test_residual_does_something(self, base_config, sample_input) -> None:
        """Block output MUST differ from input (non-identity transformation)."""
        block = TransformerBlock(base_config)
        block.eval()
        out = block(sample_input)
        # With random weights, the output should be different from input
        assert not torch.allclose(out, sample_input, atol=1e-4), (
            "Block output equals input — no transformation applied"
        )


class TestBlockConditionalMoE:
    """Conditional MoE vs FeedForward selection."""

    def test_dense_ff_when_num_experts_equals_one(self, base_config) -> None:
        """When num_experts=1, block MUST use FeedForward (check by type path)."""
        block = TransformerBlock(base_config)
        # The ff module should NOT be an instance of MoELayer
        from src.transformer.moe import MoELayer
        assert not isinstance(block.ff, MoELayer), (
            "num_experts=1 should use FeedForward, not MoELayer"
        )

    def test_moe_when_num_experts_greater_than_one(self, moe_config) -> None:
        """When num_experts>1, block MUST use MoELayer."""
        block = TransformerBlock(moe_config)
        from src.transformer.moe import MoELayer
        assert isinstance(block.ff, MoELayer), (
            "num_experts>1 should use MoELayer, not FeedForward"
        )

    def test_moe_and_ff_both_output_correct_shape(self, base_config, moe_config, sample_input) -> None:
        """Both MoE and FF paths MUST produce output shape (B, S, D)."""
        ff_block = TransformerBlock(base_config)
        moe_block = TransformerBlock(moe_config)

        ff_block.eval()
        moe_block.eval()

        ff_out = ff_block(sample_input)
        moe_out = moe_block(sample_input)

        assert ff_out.shape == sample_input.shape, (
            f"FF block output shape {ff_out.shape} != {sample_input.shape}"
        )
        assert moe_out.shape == sample_input.shape, (
            f"MoE block output shape {moe_out.shape} != {sample_input.shape}"
        )


class TestBlockGradientFlow:
    """Gradient flow through TransformerBlock."""

    def test_backward_passes(self, base_config, sample_input) -> None:
        """loss.sum().backward() MUST succeed through TransformerBlock."""
        block = TransformerBlock(base_config)
        x = sample_input.clone().requires_grad_(True)
        out = block(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None, "Input gradient is None"
        assert x.grad.shape == x.shape, (
            f"Input grad shape {x.grad.shape} != {x.shape}"
        )
        assert torch.isfinite(x.grad).all(), "Input grad contains NaN or Inf"

    def test_all_params_have_grad(self, base_config, sample_input) -> None:
        """All TransformerBlock parameters MUST receive gradient after backward."""
        block = TransformerBlock(base_config)
        x = sample_input.clone().requires_grad_(True)
        out = block(x)
        loss = out.sum()
        loss.backward()

        for name, param in block.named_parameters():
            assert param.grad is not None, (
                f"Parameter '{name}' has no gradient"
            )
            assert torch.isfinite(param.grad).all(), (
                f"Parameter '{name}' has non-finite gradient"
            )
            assert param.grad.abs().sum().item() > 0, (
                f"Parameter '{name}' has zero gradient — not learning"
            )

    def test_backward_with_moe(self, moe_config, sample_input) -> None:
        """loss.sum().backward() MUST succeed through MoE block.

        Note: MoELayer is a placeholder routing to experts[0] only,
        so unused experts may not receive gradients. We verify input
        gradient and at least one MoE parameter receives gradient.
        """
        block = TransformerBlock(moe_config)
        x = sample_input.clone().requires_grad_(True)
        out = block(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None, "Input gradient is None for MoE block"
        assert torch.isfinite(x.grad).all(), (
            "Input grad contains NaN or Inf for MoE block"
        )
        # At least the first expert (used by placeholder) should have grad
        assert block.ff.experts[0].gate_proj.weight.grad is not None, (
            "MoE active expert has no gradient"
        )
        # And it should be finite
        assert torch.isfinite(block.ff.experts[0].gate_proj.weight.grad).all(), (
            "MoE active expert has non-finite gradient"
        )


class TestBlockKVCachePassthrough:
    """KV cache passthrough to CausalSelfAttention."""

    def test_kv_cache_passthrough(self, base_config) -> None:
        """Block MUST accept and pass KV cache to attention (no crash)."""
        from src.transformer.kv_cache import KVCache

        block = TransformerBlock(base_config)
        block.eval()

        # Create a KV cache
        cache = KVCache(
            max_seq_len=512,
            n_heads=base_config.n_heads,
            d_head=base_config.d_head,
        )

        # First call with no cache (stores into cache)
        x1 = torch.randn(1, 1, 128)
        out1 = block(x1, kv_cache=cache)
        assert out1.shape == (1, 1, 128), (
            f"First call with cache: expected (1, 1, 128), got {out1.shape}"
        )

        # Second call adds to cache
        x2 = torch.randn(1, 1, 128)
        out2 = block(x2, kv_cache=cache)
        assert out2.shape == (1, 1, 128), (
            f"Second call with cache: expected (1, 1, 128), got {out2.shape}"
        )

        # Cache should have length 2 after two calls
        assert cache.seq_len == 2, (
            f"Cache should have seq_len=2, got {cache.seq_len}"
        )

    def test_kv_cache_default_none(self, base_config, sample_input) -> None:
        """Block MUST work without KV cache (kv_cache=None default)."""
        block = TransformerBlock(base_config)
        block.eval()
        out = block(sample_input)
        assert out.shape == sample_input.shape, (
            f"Output without KV cache: expected {sample_input.shape}, got {out.shape}"
        )
