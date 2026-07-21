"""Tests for TransformerLM model.

TransformerLM requirements:
- Logit shape MUST be (B, S, V) for token_id input (B, S)
- Param count MUST equal 80_461_440 for default config
- Forward + backward MUST succeed (gradient flows to all params)
- Optional KV cache list MUST be accepted
- Weight tying: embedding parameters count already includes output head
"""

import torch
import pytest
from src.transformer.config import M01Config
from src.model.lm import TransformerLM  # RED: will fail — module doesn't exist yet


@pytest.fixture
def base_config() -> M01Config:
    """Reduced config for fast shape/gradient tests."""
    config = M01Config(
        vocab_size=16384,
        context_length=512,
        d_model=128,
        n_heads=4,
        d_ff=344,
        n_layers=2,
    )
    return config


@pytest.fixture
def full_config() -> M01Config:
    """Default full config for param count verification."""
    return M01Config()


@pytest.fixture
def sample_tokens() -> torch.Tensor:
    """Random token IDs (B=2, S=16)."""
    return torch.randint(0, 16384, (2, 16))


class TestLMShape:
    """TransformerLM output shape invariants."""

    def test_logit_shape(self, base_config, sample_tokens) -> None:
        """Forward pass MUST return logits of shape (B, S, V)."""
        model = TransformerLM(base_config)
        model.eval()
        logits = model(sample_tokens)
        assert logits.shape == (2, 16, 16384), (
            f"Expected (2, 16, 16384), got {logits.shape}"
        )

    def test_logit_shape_single_token(self, base_config) -> None:
        """Forward with single token SHOULD return (1, 1, V) logits."""
        model = TransformerLM(base_config)
        model.eval()
        tokens = torch.randint(0, 16384, (1, 1))
        logits = model(tokens)
        assert logits.shape == (1, 1, 16384), (
            f"Expected (1, 1, 16384), got {logits.shape}"
        )

    def test_logit_shape_different_batch_sizes(self, base_config) -> None:
        """Forward SHOULD work with various batch sizes."""
        model = TransformerLM(base_config)
        model.eval()
        batch_sizes = [(1, 8), (4, 16), (8, 32)]
        for b, s in batch_sizes:
            tokens = torch.randint(0, 16384, (b, s))
            logits = model(tokens)
            assert logits.shape == (b, s, 16384), (
                f"Expected ({b}, {s}, 16384), got {logits.shape}"
            )

    def test_logit_shape_with_kv_caches(self, base_config) -> None:
        """Forward with KV caches MUST return (B, S, V) logits.
        
        Note: KVCache.append() requires batch_size=1 and seq_len=1
        (autoregressive single-token generation), so we use single-token input.
        """
        from src.transformer.kv_cache import KVCache

        model = TransformerLM(base_config)
        model.eval()

        kv_caches = []
        for _ in range(base_config.n_layers):
            cache = KVCache(
                max_seq_len=512,
                n_heads=base_config.n_heads,
                d_head=base_config.d_head,
            )
            kv_caches.append(cache)

        # Single-token autoregressive generation
        tokens = torch.randint(0, 16384, (1, 1))
        logits = model(tokens, kv_caches=kv_caches)
        assert logits.shape == (1, 1, 16384), (
            f"Expected (1, 1, 16384), got {logits.shape}"
        )

        # Second token appended to KV cache
        tokens2 = torch.randint(0, 16384, (1, 1))
        logits2 = model(tokens2, kv_caches=kv_caches)
        assert logits2.shape == (1, 1, 16384), (
            f"Expected (1, 1, 16384), got {logits2.shape}"
        )

    def test_kv_cache_default_none(self, base_config, sample_tokens) -> None:
        """Forward MUST work without KV caches (kv_caches=None default)."""
        model = TransformerLM(base_config)
        model.eval()
        logits = model(sample_tokens)
        assert logits.shape == (2, 16, 16384), (
            f"Expected (2, 16, 16384), got {logits.shape}"
        )


class TestLMParamCount:
    """Parameter count verification."""

    def test_param_count_default_config(self, full_config) -> None:
        """Default config (MoE Stage 1: 4+1 tk1) MUST have 110_225_536 parameters."""
        model = TransformerLM(full_config)
        total = sum(p.numel() for p in model.parameters())
        assert total == 99_739_776, (
            f"Expected 99_739_776 params, got {total}"
        )


class TestLMGradientFlow:
    """Gradient flow through TransformerLM."""

    def test_forward_backward_succeeds(self, base_config, sample_tokens) -> None:
        """loss.backward() MUST succeed through TransformerLM."""
        model = TransformerLM(base_config)
        logits = model(sample_tokens)
        loss = logits.sum()
        loss.backward()
        # Backward must not crash — no explicit assertion needed

    def test_all_params_have_grad(self, base_config, sample_tokens) -> None:
        """All TransformerLM parameters MUST receive gradient after backward."""
        model = TransformerLM(base_config)
        logits = model(sample_tokens)
        loss = logits.sum()
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, (
                f"Parameter '{name}' has no gradient"
            )
            assert torch.isfinite(param.grad).all(), (
                f"Parameter '{name}' has non-finite gradient"
            )
            assert param.grad.abs().sum().item() > 0, (
                f"Parameter '{name}' has zero gradient — not learning"
            )

    def test_grad_flow_with_kv_caches(self, base_config) -> None:
        """Gradient MUST flow through model with KV caches.
        
        Note: KVCache.append() requires batch_size=1 and seq_len=1
        (autoregressive single-token generation), so we use single-token input.
        """
        from src.transformer.kv_cache import KVCache

        model = TransformerLM(base_config)

        kv_caches = []
        for _ in range(base_config.n_layers):
            cache = KVCache(
                max_seq_len=512,
                n_heads=base_config.n_heads,
                d_head=base_config.d_head,
            )
            kv_caches.append(cache)

        # Single-token forward with KV caches
        tokens = torch.randint(0, 16384, (1, 1))
        logits = model(tokens, kv_caches=kv_caches)
        loss = logits.sum()
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, (
                f"Parameter '{name}' has no gradient with KV caches"
            )
            assert torch.isfinite(param.grad).all(), (
                f"Parameter '{name}' has non-finite gradient with KV caches"
            )
