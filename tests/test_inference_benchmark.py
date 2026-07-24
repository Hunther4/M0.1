import torch

from src.eval.inference_benchmark import benchmark_inference, compare_prompt_cache
from src.model.lm import TransformerLM
from src.transformer.config import M01Config


class _Tokenizer:
    special_tokens = {"<|endoftext|>": 63}

    def encode(self, prompt: str) -> list[int]:
        return {
            "alpha": [1, 2, 3, 4],
            "beta": [1, 2, 3, 5],
        }[prompt]

    def decode(self, ids: list[int]) -> str:
        return ",".join(map(str, ids))


def _model() -> TransformerLM:
    config = M01Config(
        vocab_size=64,
        context_length=24,
        d_model=32,
        n_heads=4,
        d_ff=64,
        n_layers=1,
        num_experts=1,
        num_shared_experts=0,
        mla_kv_c_dim=12,
        mla_rope_dim=4,
        dropout=0.0,
        attention_backend="math",
    )
    return TransformerLM(config).eval()


def test_benchmark_reports_cache_telemetry() -> None:
    report = benchmark_inference(
        _model(),
        _Tokenizer(),
        ["alpha", "beta"],
        max_gen_len=2,
        repetitions=2,
        use_prompt_cache=True,
    )

    assert report["summary"]["requests"] == 4
    assert report["summary"]["reused_prompt_tokens"] == 9
    assert report["cache"]["hits"] == 3
    assert report["cache"]["misses"] == 1
    assert report["cache"]["hit_rate"] == 0.75
    assert len(report["samples"]) == 4


def test_compare_verifies_deterministic_outputs() -> None:
    torch.manual_seed(19)
    report = compare_prompt_cache(
        _model(),
        _Tokenizer(),
        ["alpha", "beta"],
        max_gen_len=2,
        repetitions=2,
    )

    assert report["outputs_match"] is True
    assert report["cached"]["summary"]["reused_prompt_tokens"] > 0
    assert report["uncached"]["cache"] is None
