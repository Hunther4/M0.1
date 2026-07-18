# Spec: model-transformer-block

## Overview

Pre-norm Transformer block with residual connections and conditional feedforward/MoE routing. Each block applies self-attention followed by dense or mixture-of-experts feedforward, both wrapped with pre-norm RMSNorm and residual connections.

## Requirements

### Requirement: TransformerBlock Module

The system MUST provide a `TransformerBlock` module in `src/model/block.py` that implements a pre-norm transformer layer.

**Interface:**
```python
class TransformerBlock(nn.Module):
    def __init__(self, config: M01Config) -> None: ...
    def forward(self, x: Tensor, kv_cache: Optional[KVCache] = None) -> Tensor: ...
```

**Scenarios:**

- **Given** a `TransformerBlock` with default M01Config (`d_model=640`, `num_experts=1`), **When** `forward(x)` is called with shape `(batch, seq_len, 640)`, **Then** output shape MUST be `(batch, seq_len, 640)`.
- **Given** a `TransformerBlock` with `num_experts=4`, **When** `forward(x)` is called with shape `(batch, seq_len, 640)`, **Then** output shape MUST be `(batch, seq_len, 640)`.
- **Given** a `TransformerBlock`, **When** `forward(x)` is called with various batch sizes and sequence lengths, **Then** output shape MUST match input shape in all dimensions.
- **Given** a `TransformerBlock`, **When** `loss.sum().backward()` is called, **Then** backward MUST succeed.
- **Given** a `TransformerBlock` after a forward+backward pass, **When** all parameter gradients are inspected, **Then** every parameter MUST have a finite (non-NaN, non-inf) gradient.

### Requirement: Pre-Norm Residual Connections

The block MUST use pre-norm architecture: RMSNorm before each sublayer (attention and feedforward), with the sublayer output added to the residual (skip connection).

**Flow:**
```
x = x + dropout(attn(norm1(x)))
x = x + dropout(ff(norm2(x)))
```

**Scenarios:**

- **Given** a `TransformerBlock`, **When** `forward(x)` is called, **Then** output shape MUST equal input shape (guaranteed by residual connections).
- **Given** a `TransformerBlock` with `dropout=0.0`, **When** `forward(x)` is called with zero input, **Then** output MUST be zero (residual of zero + (sub)layers that output zero).
- **Given** a `TransformerBlock` with `dropout=0.0`, **When** `forward(x)` is called with non-zero input, **Then** output MUST differ from input (sublayers contribute signal).

### Requirement: Conditional MoE Routing

The block MUST support conditional feedforward routing: when `config.num_experts > 1`, use `MoELayer`; otherwise use `FeedForward`.

**Scenarios:**

- **Given** a `TransformerBlock` with `num_experts=1`, **When** instantiated, **Then** `self.ff` MUST be an instance of `FeedForward`.
- **Given** a `TransformerBlock` with `num_experts=4`, **When** instantiated, **Then** `self.ff` MUST be an instance of `MoELayer`.
- **Given** a `TransformerBlock` with `num_experts=1`, **When** `forward(x)` is called, **Then** output shape MUST be `(batch, seq_len, d_model)`.
- **Given** a `TransformerBlock` with `num_experts=4` (MoE), **When** `forward(x)` is called, **Then** output shape MUST be `(batch, seq_len, d_model)`.
- **Given** a `TransformerBlock` with `num_experts=1` (FF), **When** `forward(x)` is called, **Then** output shape MUST be `(batch, seq_len, d_model)`.

### Requirement: KV Cache Passthrough

The block MUST accept an optional `KVCache` parameter and pass it through to `CausalSelfAttention`.

**Scenarios:**

- **Given** a `TransformerBlock` and a `KVCache`, **When** `forward(x, kv_cache=cache)` is called, **Then** output shape MUST be `(batch, seq_len, d_model)` and the cache MUST be updated.
- **Given** a `TransformerBlock`, **When** `forward(x)` is called without a cache, **Then** it MUST produce a valid output (kv_cache defaults to None).
