"""Bounded prompt-prefix cache for autoregressive inference."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

import torch

from src.model.lm import TransformerLM
from src.transformer.kv_cache import (
    AttentionCache,
    HybridKVCache,
    KVCache,
    MLAKVCache,
)


@dataclass(frozen=True)
class PromptCacheStats:
    requests: int
    hits: int
    misses: int
    evictions: int
    rejected_stores: int
    reused_tokens: int
    entries: int
    bytes: int

    @property
    def hit_rate(self) -> float:
        return self.hits / self.requests if self.requests else 0.0


@dataclass(frozen=True)
class _LayerSnapshot:
    kind: str
    tensors: tuple[torch.Tensor, ...]

    @property
    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.tensors)


@dataclass(frozen=True)
class _PromptEntry:
    tokens: tuple[int, ...]
    layers: tuple[_LayerSnapshot, ...]

    @property
    def nbytes(self) -> int:
        return sum(layer.nbytes for layer in self.layers)


def model_cache_fingerprint(model: TransformerLM) -> tuple[Any, ...]:
    """Return a cheap fingerprint that changes after model/config mutations."""
    config = tuple(sorted(asdict(model.config).items()))
    parameter_versions = tuple(
        (
            name,
            tensor.device.type,
            tensor.device.index,
            str(tensor.dtype),
            tuple(tensor.shape),
            tensor._version,
        )
        for name, tensor in model.named_parameters(remove_duplicate=False)
    )
    buffer_versions = tuple(
        (
            name,
            tensor.device.type,
            tensor.device.index,
            str(tensor.dtype),
            tuple(tensor.shape),
            tensor._version,
        )
        for name, tensor in model.named_buffers(remove_duplicate=False)
    )
    return id(model), model.training, config, parameter_versions, buffer_versions


def _snapshot_layer(cache: AttentionCache, length: int) -> _LayerSnapshot:
    if length < 0 or length > cache.seq_len:
        raise ValueError(f"Invalid snapshot length {length} for cache length {cache.seq_len}")

    if isinstance(cache, MLAKVCache):
        tensors = (
            cache.latent[:, :length].detach().clone(),
            cache.rope[:, :length].detach().clone(),
        )
        return _LayerSnapshot("mla", tensors)
    if isinstance(cache, HybridKVCache):
        tensors = (
            cache.k_csa[:, :length].detach().clone(),
            cache.v_csa[:, :length].detach().clone(),
            cache.k_hca[:, :length].detach().clone(),
            cache.v_hca[:, :length].detach().clone(),
        )
        return _LayerSnapshot("hybrid", tensors)
    if isinstance(cache, KVCache):
        tensors = (
            cache.k[:, :length].detach().clone(),
            cache.v[:, :length].detach().clone(),
        )
        return _LayerSnapshot("mha", tensors)
    raise TypeError(f"Unsupported attention cache type: {type(cache).__name__}")


def _restore_layer(
    cache: AttentionCache, snapshot: _LayerSnapshot, length: int
) -> None:
    if length < 0 or any(length > tensor.shape[1] for tensor in snapshot.tensors):
        raise ValueError("Snapshot does not contain the requested prefix length")
    if length > cache.max_seq_len:
        raise ValueError("Snapshot prefix exceeds the destination cache capacity")

    source = tuple(tensor[:, :length] for tensor in snapshot.tensors)
    if isinstance(cache, MLAKVCache) and snapshot.kind == "mla":
        latent, rope = source
        cache.append(latent, rope)
        return
    if isinstance(cache, HybridKVCache) and snapshot.kind == "hybrid":
        cache.append(*source)
        return
    if isinstance(cache, KVCache) and snapshot.kind == "mha":
        cache.append(*source)
        return
    raise TypeError(
        f"Cannot restore {snapshot.kind!r} snapshot into {type(cache).__name__}"
    )


def _common_prefix_length(left: Sequence[int], right: Sequence[int]) -> int:
    length = 0
    for left_token, right_token in zip(left, right):
        if left_token != right_token:
            break
        length += 1
    return length


class PromptPrefixCache:
    """LRU cache that reuses immutable KV snapshots for shared prompt prefixes."""

    def __init__(
        self,
        max_entries: int = 8,
        max_bytes: int = 256 * 1024 * 1024,
        min_prefix_tokens: int = 1,
    ) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        if max_bytes < 1:
            raise ValueError("max_bytes must be positive")
        if min_prefix_tokens < 1:
            raise ValueError("min_prefix_tokens must be positive")
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self.min_prefix_tokens = min_prefix_tokens
        self._entries: OrderedDict[tuple[int, ...], _PromptEntry] = OrderedDict()
        self._fingerprint: tuple[Any, ...] | None = None
        self._bytes = 0
        self._requests = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._rejected_stores = 0
        self._reused_tokens = 0

    def _ensure_model(self, model: TransformerLM) -> None:
        fingerprint = model_cache_fingerprint(model)
        if self._fingerprint != fingerprint:
            self.clear()
            self._fingerprint = fingerprint

    def restore_longest_prefix(
        self,
        model: TransformerLM,
        tokens: Sequence[int],
        caches: Sequence[AttentionCache],
    ) -> int:
        """Restore the longest cached prefix and return reused token count."""
        if model.training:
            raise ValueError("Prompt caching requires model.eval()")
        self._ensure_model(model)
        self._requests += 1

        best_key: tuple[int, ...] | None = None
        best_length = 0
        for key in self._entries:
            length = _common_prefix_length(key, tokens)
            if length > best_length:
                best_key, best_length = key, length

        if best_key is None or best_length < self.min_prefix_tokens:
            self._misses += 1
            return 0

        entry = self._entries.pop(best_key)
        self._entries[best_key] = entry
        if len(entry.layers) != len(caches):
            raise ValueError("Cached layer count does not match the model")
        for cache, layer in zip(caches, entry.layers):
            _restore_layer(cache, layer, best_length)

        self._hits += 1
        self._reused_tokens += best_length
        return best_length

    def capture(
        self,
        model: TransformerLM,
        tokens: Iterable[int],
        caches: Sequence[AttentionCache],
    ) -> _PromptEntry | None:
        """Capture an immutable prefill snapshot without mutating the cache."""
        if model.training:
            raise ValueError("Prompt caching requires model.eval()")
        key = tuple(tokens)
        if len(key) < self.min_prefix_tokens:
            return None
        if any(cache.seq_len < len(key) for cache in caches):
            raise ValueError("Cannot cache more prompt tokens than the KV cache contains")
        return _PromptEntry(
            key,
            tuple(_snapshot_layer(cache, len(key)) for cache in caches),
        )

    def commit(self, model: TransformerLM, entry: _PromptEntry) -> bool:
        """Insert a captured snapshot into the LRU store."""
        if model.training:
            raise ValueError("Prompt caching requires model.eval()")
        self._ensure_model(model)
        if entry.nbytes > self.max_bytes:
            self._rejected_stores += 1
            return False

        key = entry.tokens
        previous = self._entries.pop(key, None)
        if previous is not None:
            self._bytes -= previous.nbytes
        self._entries[key] = entry
        self._bytes += entry.nbytes

        while len(self._entries) > self.max_entries or self._bytes > self.max_bytes:
            _, evicted = self._entries.popitem(last=False)
            self._bytes -= evicted.nbytes
            self._evictions += 1
        return True

    def store(
        self,
        model: TransformerLM,
        tokens: Iterable[int],
        caches: Sequence[AttentionCache],
    ) -> bool:
        """Capture and store a prefill snapshot immediately."""
        entry = self.capture(model, tokens, caches)
        if entry is None:
            return False
        return self.commit(model, entry)

    def clear(self) -> None:
        """Drop snapshots while preserving lifetime counters."""
        self._entries.clear()
        self._bytes = 0

    @property
    def stats(self) -> PromptCacheStats:
        return PromptCacheStats(
            requests=self._requests,
            hits=self._hits,
            misses=self._misses,
            evictions=self._evictions,
            rejected_stores=self._rejected_stores,
            reused_tokens=self._reused_tokens,
            entries=len(self._entries),
            bytes=self._bytes,
        )
