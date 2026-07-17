# Spec: transformer-attention

## Overview

Causal multi-head self-attention with optional KV-Cache for autoregressive generation.

## Requirements

### Requirement: CausalSelfAttention Module

The system MUST provide a `CausalSelfAttention` module in `src/transformer/attention.py` that implements multi-head self-attention with a causal mask.

**Scenarios:**

- **Given** a `CausalSelfAttention` with default M01Config, **When** `forward(x)` is called with shape `(batch, seq_len, 640)`, **Then** output shape MUST be `(batch, seq_len, 640)`.
- **Given** a `CausalSelfAttention`, **When** `forward(x)` is called, **Then** the causal mask MUST prevent attention to future tokens (attention weights `attn[i][j] = 0` for `j > i`).
- **Given** a `CausalSelfAttention`, **When** `forward(x)` is called, **Then** Q, K, V projections MUST use `bias=False` Linear layers.
- **Given** a `CausalSelfAttention`, **When** `forward(x)` is called, **Then** attention scores MUST be divided by `sqrt(d_head)` = `sqrt(64)`.
- **Given** a `CausalSelfAttention`, **When** `forward(x)` is called, **Then** output MUST pass through an output projection `W_o`.

### Requirement: KV-Cache Integration

`CausalSelfAttention` MUST accept an optional `KVCache` parameter for autoregressive generation.

**Scenarios:**

- **Given** a `CausalSelfAttention` and a `KVCache`, **When** `forward(x, kv_cache=cache)` is called, **Then** new K/V MUST be appended to the cache.
- **Given** a `CausalSelfAttention` and a `KVCache` with prior entries, **When** `forward(x, kv_cache=cache)` is called, **Then** attention MUST attend to both cached and new K/V entries.
- **Given** a `CausalSelfAttention` without a cache, **When** `forward(x)` is called, **Then** it MUST produce the same output as with a fresh cache (no-cache baseline).
- **Given** a `CausalSelfAttention` with a cache, **When** the cache is reset between calls, **Then** the second call MUST produce output independent of the first call.

### Requirement: KVCache Class

The system MUST provide a `KVCache` class in `src/transformer/kv_cache.py` for storing past key/value tensors during autoregressive generation.

**Scenarios:**

- **Given** a `KVCache(max_seq_len=8192, n_heads=10, d_head=64)`, **When** initialized, **Then** internal buffers MUST have shape `(1, 8192, 10, 64)`.
- **Given** a `KVCache`, **When** `append(k, v)` is called with `k, v` shape `(batch, 1, n_heads, d_head)`, **Then** `seq_len` property MUST increase by 1.
- **Given** a `KVCache`, **When** `reset()` is called, **Then** buffers MUST be zeroed and `seq_len` MUST reset to 0.
- **Given** a `KVCache`, **When** `append(k, v)` is called N times, **Then** returned K/V tensors MUST have shape `(batch, N, n_heads, d_head)`.
- **Given** a `KVCache` after N appends, **When** compared to a no-cache attention call with sequence length N, **Then** outputs MUST match (within tolerance).

### Requirement: No HuggingFace Dependencies

The attention implementation MUST NOT use `transformers` library, `xops`, or any HuggingFace-specific packages. All operations MUST use standard PyTorch.

**Scenarios:**

- **Given** the `attention.py` source code, **When** inspected, **Then** it MUST NOT import from `transformers`, `xops`, or `flash_attn`.
- **Given** the `attention.py` source code, **When** inspected, **Then** it MUST use `torch.nn.functional.scaled_dot_product_attention` or manual `Q @ K^T` computation.
