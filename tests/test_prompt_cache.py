import pytest
import torch

from src.inference.generate import GenerationMetrics, generate
from src.inference.prompt_cache import PromptPrefixCache
from src.model.lm import TransformerLM
from src.transformer.config import M01Config
from src.transformer.kv_cache import build_attention_cache


class _Tokenizer:
    special_tokens = {"<|endoftext|>": 63}

    def __init__(self, prompts: dict[str, list[int]]) -> None:
        self.prompts = prompts

    def encode(self, text: str) -> list[int]:
        return list(self.prompts[text])

    def decode(self, ids: list[int]) -> str:
        return ",".join(str(token) for token in ids)


def _model(**overrides) -> TransformerLM:
    values = {
        "vocab_size": 64,
        "context_length": 32,
        "d_model": 32,
        "n_heads": 4,
        "d_ff": 64,
        "n_layers": 2,
        "num_experts": 1,
        "num_shared_experts": 0,
        "mla_kv_c_dim": 12,
        "mla_rope_dim": 4,
        "dropout": 0.0,
        "attention_backend": "math",
    }
    values.update(overrides)
    return TransformerLM(M01Config(**values)).eval()


@pytest.mark.parametrize(
    "mode",
    [
        {"use_mla": True, "use_hybrid_attention": False},
        {"use_mla": False, "use_hybrid_attention": False},
        {
            "use_mla": False,
            "use_hybrid_attention": True,
            "csa_kv_dim": 12,
            "hca_kv_dim": 8,
            "local_window_size": 2,
        },
    ],
)
def test_cached_generation_matches_uncached_for_all_attention_modes(mode) -> None:
    torch.manual_seed(7)
    model = _model(**mode)
    tokenizer = _Tokenizer({"first": [1, 2, 3, 4], "second": [1, 2, 3, 5]})
    cache = PromptPrefixCache()

    generate(model, tokenizer, "first", max_gen_len=3, temperature=0.0, prompt_cache=cache)
    metrics = GenerationMetrics()
    cached = generate(
        model,
        tokenizer,
        "second",
        max_gen_len=3,
        temperature=0.0,
        prompt_cache=cache,
        metrics=metrics,
    )
    uncached = generate(model, tokenizer, "second", max_gen_len=3, temperature=0.0)

    assert cached == uncached
    assert metrics.reused_prompt_tokens == 3
    assert cache.stats.hits == 1
    assert cache.stats.misses == 1


def test_cache_invalidates_after_weight_change() -> None:
    model = _model()
    tokenizer = _Tokenizer({"prompt": [1, 2, 3, 4]})
    cache = PromptPrefixCache()
    generate(model, tokenizer, "prompt", max_gen_len=1, temperature=0.0, prompt_cache=cache)

    with torch.no_grad():
        model.embedding.embedding.weight.add_(0.01)

    metrics = GenerationMetrics()
    generate(
        model,
        tokenizer,
        "prompt",
        max_gen_len=1,
        temperature=0.0,
        prompt_cache=cache,
        metrics=metrics,
    )

    assert metrics.reused_prompt_tokens == 0
    assert cache.stats.entries == 1


def test_cache_rejects_training_mode() -> None:
    model = _model().train()
    cache = PromptPrefixCache()
    caches = [
        build_attention_cache(model.config, torch.device("cpu"))
        for _ in range(model.config.n_layers)
    ]

    with pytest.raises(ValueError, match=r"model\.eval"):
        cache.restore_longest_prefix(model, [1, 2], caches)


def test_cache_enforces_memory_limit() -> None:
    model = _model()
    tokenizer = _Tokenizer({"prompt": [1, 2, 3, 4]})
    cache = PromptPrefixCache(max_bytes=1)

    generate(model, tokenizer, "prompt", max_gen_len=1, temperature=0.0, prompt_cache=cache)

    assert cache.stats.entries == 0
    assert cache.stats.rejected_stores == 1
