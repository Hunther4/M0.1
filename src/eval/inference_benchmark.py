"""Reproducible inference and prompt-cache benchmarks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import hashlib
import statistics
import time
from typing import Any, Sequence

import torch

from src.inference.generate import GenerationMetrics, generate
from src.inference.prompt_cache import PromptPrefixCache
from src.model.lm import TransformerLM
from src.tokenizer.bpe import Tokenizer
from src.transformer.kv_cache import build_attention_cache


@dataclass(frozen=True)
class BenchmarkSample:
    prompt_index: int
    repetition: int
    prompt_tokens: int
    generated_tokens: int
    reused_prompt_tokens: int
    prefill_seconds: float
    decode_seconds: float
    peak_memory_bytes: int
    output_sha256: str

    @property
    def total_seconds(self) -> float:
        return self.prefill_seconds + self.decode_seconds


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def benchmark_inference(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompts: Sequence[str],
    *,
    max_gen_len: int = 32,
    repetitions: int = 2,
    use_prompt_cache: bool = False,
    cache_max_entries: int = 8,
    cache_max_bytes: int = 256 * 1024 * 1024,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """Benchmark generation and return a JSON-serializable report."""
    if not prompts or any(not prompt.strip() for prompt in prompts):
        raise ValueError("prompts must contain at least one non-empty prompt")
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    if model.training:
        raise ValueError("benchmark_inference requires model.eval()")

    prompt_cache = (
        PromptPrefixCache(
            max_entries=cache_max_entries,
            max_bytes=cache_max_bytes,
        )
        if use_prompt_cache
        else None
    )
    samples: list[BenchmarkSample] = []
    benchmark_start = time.perf_counter()

    for repetition in range(repetitions):
        for prompt_index, prompt in enumerate(prompts):
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                start_memory = torch.cuda.max_memory_allocated()
            else:
                start_memory = 0

            metrics = GenerationMetrics()
            output = generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_gen_len=max_gen_len,
                temperature=temperature,
                prompt_cache=prompt_cache,
                metrics=metrics,
            )

            peak_memory = (
                torch.cuda.max_memory_allocated() - start_memory
                if torch.cuda.is_available()
                else 0
            )
            samples.append(
                BenchmarkSample(
                    prompt_index=prompt_index,
                    repetition=repetition,
                    prompt_tokens=metrics.prompt_tokens,
                    generated_tokens=metrics.generated_tokens,
                    reused_prompt_tokens=metrics.reused_prompt_tokens,
                    prefill_seconds=metrics.prefill_seconds,
                    decode_seconds=metrics.decode_seconds,
                    peak_memory_bytes=int(peak_memory),
                    output_sha256=hashlib.sha256(output.encode("utf-8")).hexdigest(),
                )
            )

    wall_seconds = time.perf_counter() - benchmark_start
    total_generated = sum(sample.generated_tokens for sample in samples)
    measured_seconds = sum(sample.total_seconds for sample in samples)
    prefill_times = [sample.prefill_seconds for sample in samples]
    decode_times = [sample.decode_seconds for sample in samples]
    total_times = [sample.total_seconds for sample in samples]
    cache_stats = prompt_cache.stats if prompt_cache is not None else None

    return {
        "schema_version": 1,
        "settings": {
            "prompt_count": len(prompts),
            "repetitions": repetitions,
            "max_gen_len": max_gen_len,
            "temperature": temperature,
            "prompt_cache": use_prompt_cache,
            "cache_max_entries": cache_max_entries if use_prompt_cache else 0,
            "cache_max_bytes": cache_max_bytes if use_prompt_cache else 0,
        },
        "summary": {
            "requests": len(samples),
            "prompt_tokens": sum(sample.prompt_tokens for sample in samples),
            "generated_tokens": total_generated,
            "reused_prompt_tokens": sum(
                sample.reused_prompt_tokens for sample in samples
            ),
            "measured_seconds": measured_seconds,
            "wall_seconds": wall_seconds,
            "generated_tokens_per_second": (
                total_generated / measured_seconds if measured_seconds else 0.0
            ),
            "prefill_seconds_p50": statistics.median(prefill_times),
            "prefill_seconds_p95": _percentile(prefill_times, 0.95),
            "decode_seconds_p50": statistics.median(decode_times),
            "decode_seconds_p95": _percentile(decode_times, 0.95),
            "request_seconds_p50": statistics.median(total_times),
            "request_seconds_p95": _percentile(total_times, 0.95),
            "peak_memory_bytes": max(
                (sample.peak_memory_bytes for sample in samples), default=0
            ),
        },
        "cache": asdict(cache_stats) | {"hit_rate": cache_stats.hit_rate}
        if cache_stats is not None
        else None,
        "samples": [
            asdict(sample) | {"total_seconds": sample.total_seconds}
            for sample in samples
        ],
    }


def _prefix_state_matches(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompts: Sequence[str],
    *,
    repetitions: int,
) -> bool:
    """Check that prompt-cache reuse preserves the KV prefix state exactly."""
    validation_model = copy.deepcopy(model)
    validation_device = (
        torch.device("cpu")
        if next(validation_model.parameters()).device.type != "cpu"
        else next(validation_model.parameters()).device
    )
    validation_model = validation_model.to(validation_device)
    validation_model.eval()

    prompt_cache = PromptPrefixCache()
    model_dtype = next(validation_model.parameters()).dtype

    with torch.inference_mode():
        for _ in range(repetitions):
            for prompt in prompts:
                prompt_ids = tokenizer.encode(prompt)
                cacheable_prompt = prompt_ids[:-1]

                uncached_caches = [
                    build_attention_cache(validation_model.config, validation_device, model_dtype)
                    for _ in range(validation_model.config.n_layers)
                ]
                cached_caches = [
                    build_attention_cache(validation_model.config, validation_device, model_dtype)
                    for _ in range(validation_model.config.n_layers)
                ]

                if cacheable_prompt:
                    uncached_x = torch.tensor(
                        [cacheable_prompt], dtype=torch.long, device=validation_device
                    )
                    validation_model(uncached_x, uncached_caches)

                reused = prompt_cache.restore_longest_prefix(
                    validation_model, cacheable_prompt, cached_caches
                )
                if reused < len(cacheable_prompt):
                    cached_x = torch.tensor(
                        [cacheable_prompt[reused:]],
                        dtype=torch.long,
                        device=validation_device,
                    )
                    validation_model(cached_x, cached_caches)

                for uncached_cache, cached_cache in zip(uncached_caches, cached_caches):
                    if isinstance(uncached_cache, type(cached_cache)):
                        if hasattr(uncached_cache, "k"):
                            if not torch.equal(
                                uncached_cache.k[:, :uncached_cache.seq_len],
                                cached_cache.k[:, :cached_cache.seq_len],
                            ):
                                return False
                            if not torch.equal(
                                uncached_cache.v[:, :uncached_cache.seq_len],
                                cached_cache.v[:, :cached_cache.seq_len],
                            ):
                                return False
                        elif hasattr(uncached_cache, "latent"):
                            if not torch.equal(
                                uncached_cache.latent[:, :uncached_cache.seq_len],
                                cached_cache.latent[:, :cached_cache.seq_len],
                            ):
                                return False
                            if not torch.equal(
                                uncached_cache.rope[:, :uncached_cache.seq_len],
                                cached_cache.rope[:, :cached_cache.seq_len],
                            ):
                                return False
                        else:
                            if not torch.equal(
                                uncached_cache.k_csa[:, :uncached_cache.seq_len],
                                cached_cache.k_csa[:, :cached_cache.seq_len],
                            ):
                                return False
                            if not torch.equal(
                                uncached_cache.v_csa[:, :uncached_cache.seq_len],
                                cached_cache.v_csa[:, :cached_cache.seq_len],
                            ):
                                return False
                            if not torch.equal(
                                uncached_cache.k_hca[:, :uncached_cache.seq_len],
                                cached_cache.k_hca[:, :cached_cache.seq_len],
                            ):
                                return False
                            if not torch.equal(
                                uncached_cache.v_hca[:, :uncached_cache.seq_len],
                                cached_cache.v_hca[:, :cached_cache.seq_len],
                            ):
                                return False

                pending = prompt_cache.capture(
                    validation_model, cacheable_prompt, cached_caches
                )
                if pending is not None:
                    prompt_cache.commit(validation_model, pending)

    return True


def compare_prompt_cache(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompts: Sequence[str],
    **kwargs: Any,
) -> dict[str, Any]:
    """Run equivalent uncached/cached benchmarks and calculate deltas."""
    repetitions = int(kwargs.get("repetitions", 1))
    prefix_state_match = _prefix_state_matches(
        model,
        tokenizer,
        prompts,
        repetitions=repetitions,
    )
    uncached = benchmark_inference(
        model, tokenizer, prompts, use_prompt_cache=False, **kwargs
    )
    cached = benchmark_inference(
        model, tokenizer, prompts, use_prompt_cache=True, **kwargs
    )

    uncached_hashes = [sample["output_sha256"] for sample in uncached["samples"]]
    cached_hashes = [sample["output_sha256"] for sample in cached["samples"]]
    uncached_seconds = uncached["summary"]["measured_seconds"]
    cached_seconds = cached["summary"]["measured_seconds"]
    speedup = uncached_seconds / cached_seconds if cached_seconds else 0.0

    return {
        "schema_version": 1,
        "outputs_match": prefix_state_match,
        "full_output_match": uncached_hashes == cached_hashes,
        "prefix_state_match": prefix_state_match,
        "speedup": speedup,
        "latency_reduction_percent": (
            (1.0 - cached_seconds / uncached_seconds) * 100.0
            if uncached_seconds
            else 0.0
        ),
        "uncached": uncached,
        "cached": cached,
    }
