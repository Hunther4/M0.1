# Spec: model-transformer-lm

## Overview

Decoder-only Transformer Language Model assembled from TokenEmbedding, a stack of TransformerBlock layers, and RMSNorm. The output projection shares weights with the embedding layer (weight tying). Accepts optional KV caches for autoregressive generation.

Architecture: `TokenEmbedding → [TransformerBlock × n_layers] → RMSNorm → output_head (tied)`

## Requirements

### Requirement: TransformerLM Module

The system MUST provide a `TransformerLM` module in `src/model/lm.py` that assembles a complete decoder-only language model.

**Interface:**
```python
class TransformerLM(nn.Module):
    def __init__(self, config: M01Config) -> None: ...
    def forward(self, token_ids: Tensor, kv_caches: Optional[list[Optional[KVCache]]] = None) -> Tensor: ...
```

**Scenarios:**

- **Given** a `TransformerLM` with default M01Config, **When** `forward(token_ids)` is called with shape `(batch, seq_len)`, **Then** output logits shape MUST be `(batch, seq_len, 32768)`.
- **Given** a `TransformerLM`, **When** `forward(token_ids)` is called with a single token `(1, 1)`, **Then** output logits shape MUST be `(1, 1, 32768)`.
- **Given** a `TransformerLM`, **When** `forward(token_ids)` is called with various batch sizes `(1, 8)`, `(4, 16)`, `(8, 32)`, **Then** output logits shape MUST be `(batch, seq_len, 32768)` for each.
- **Given** a `TransformerLM`, **When** `forward(token_ids, kv_caches=list_of_caches)` is called, **Then** output logits shape MUST be `(batch, seq_len, 32768)`.
- **Given** a `TransformerLM`, **When** `forward(token_ids)` is called without KV caches, **Then** it MUST produce valid output (defaults to None).
- **Given** a `TransformerLM` with default M01Config, **When** `sum(p.numel() for p in model.parameters())` is computed, **Then** total MUST equal `80_461_440`.
- **Given** a `TransformerLM`, **When** `forward(token_ids)` is called and `loss.backward()` is performed, **Then** backward MUST succeed.
- **Given** a `TransformerLM` after forward+backward, **When** all parameter gradients are inspected, **Then** every parameter MUST have a finite (non-NaN, non-inf) gradient.
- **Given** a `TransformerLM` with KV caches after forward+backward, **When** parameter gradients are inspected, **Then** gradients MUST flow through all parameters.

### Requirement: Architecture and Assembly

The model MUST follow this architecture:
1. `TokenEmbedding` — maps token IDs to `(batch, seq_len, d_model)` embeddings scaled by `1/sqrt(d_model)`.
2. `nn.ModuleList` of `TransformerBlock` × `n_layers` (12 layers in default config) — each applying pre-norm self-attention and feedforward.
3. `RMSNorm` — applied after all blocks, before the output head.
4. Output head — weight-tied with the embedding layer.

**Scenarios:**

- **Given** a `TransformerLM` with default M01Config, **When** `len(model.blocks)` is inspected, **Then** it MUST equal `n_layers` (12).
- **Given** a `TransformerLM`, **When** `model.norm` is inspected, **Then** it MUST be an `RMSNorm` module.
- **Given** a `TransformerLM`, **When** `model.embedding` is inspected, **Then** it MUST be a `TokenEmbedding` module.
- **Given** a `TransformerLM`, **When** `model.blocks[0]` is inspected, **Then** it MUST be a `TransformerBlock` module.

### Requirement: Weight Tying

The output projection head MUST share weights with the input embedding layer. No separate output parameter matrix exists.

**Scenarios:**

- **Given** a `TransformerLM`, **When** `output_head` is inspected, **Then** it MUST be `TokenEmbedding.output_head` — a bound method that calls `F.linear(h, embedding.weight)`.
- **Given** a `TransformerLM`, **When** `model.output_head(hidden)` is called, **Then** the weight matrix used MUST be `model.embedding.weight` (same object reference).
- **Given** a `TransformerLM` after training, **When** `embedding.weight` is inspected, **Then** gradients MUST reflect updates from both the embedding lookup path and the output projection path.
