import torch

from src.transformer.attention import CausalSelfAttention
from src.transformer.attention_backend import AttentionDispatcher
from src.transformer.config import M01Config
from src.transformer.kv_cache import HybridKVCache, MLAKVCache, build_attention_cache


def _tiny_config(**overrides) -> M01Config:
    values = {
        "vocab_size": 64,
        "context_length": 16,
        "d_model": 32,
        "n_heads": 4,
        "d_ff": 64,
        "n_layers": 1,
        "num_experts": 1,
        "num_shared_experts": 0,
        "mla_kv_c_dim": 12,
        "mla_rope_dim": 4,
        "dropout": 0.0,
        "attention_backend": "math",
    }
    values.update(overrides)
    return M01Config(**values)


def test_mla_cache_matches_prefix_reference() -> None:
    torch.manual_seed(11)
    config = _tiny_config(use_mla=True, use_hybrid_attention=False)
    attention = CausalSelfAttention(config).eval()
    cache = build_attention_cache(config, torch.device("cpu"))
    assert isinstance(cache, MLAKVCache)
    x = torch.randn(1, 6, config.d_model)

    for position in range(x.shape[1]):
        cached = attention(x[:, position : position + 1], kv_cache=cache)
        reference = attention(x[:, : position + 1])[:, -1:]
        torch.testing.assert_close(cached, reference, atol=2e-5, rtol=2e-5)


def test_mla_cache_storage_is_compressed() -> None:
    config = _tiny_config(use_mla=True, use_hybrid_attention=False)
    cache = build_attention_cache(config, torch.device("cpu"))
    expanded_elements = 2 * config.context_length * config.n_heads * config.d_head
    compressed_elements = cache.latent.numel() + cache.rope.numel()

    assert cache.latent.shape[-1] == config.mla_kv_c_dim
    assert cache.rope.shape[-1] == config.d_head_rope
    assert compressed_elements < expanded_elements


def test_hybrid_cache_matches_prefix_reference() -> None:
    torch.manual_seed(17)
    config = _tiny_config(
        use_mla=False,
        use_hybrid_attention=True,
        csa_kv_dim=10,
        hca_kv_dim=6,
        local_window_size=3,
    )
    attention = CausalSelfAttention(config).eval()
    cache = build_attention_cache(config, torch.device("cpu"))
    assert isinstance(cache, HybridKVCache)
    x = torch.randn(1, 7, config.d_model)

    for position in range(x.shape[1]):
        cached = attention(x[:, position : position + 1], kv_cache=cache)
        reference = attention(x[:, : position + 1])[:, -1:]
        torch.testing.assert_close(cached, reference, atol=2e-5, rtol=2e-5)


def test_math_dispatch_matches_torch_sdpa() -> None:
    torch.manual_seed(23)
    q = torch.randn(1, 2, 4, 8)
    k = torch.randn(1, 2, 4, 8)
    v = torch.randn(1, 2, 4, 8)
    dispatcher = AttentionDispatcher("math")

    actual = dispatcher.dispatch(q, k, v, is_causal=True)
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.testing.assert_close(actual, expected)


def test_cache_factory_preserves_standard_mha() -> None:
    from src.transformer.kv_cache import KVCache

    config = _tiny_config(use_mla=False, use_hybrid_attention=False)
    assert isinstance(build_attention_cache(config, torch.device("cpu")), KVCache)
