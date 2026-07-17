# Spec: transformer-embeddings

## Overview

Tied token embedding layer — input embedding matrix shared with output projection head to reduce parameter count.

## Requirements

### Requirement: TokenEmbedding Module

The system MUST provide a `TokenEmbedding` module in `src/transformer/embeddings.py` that maps token IDs to dense embeddings and provides an output projection head using tied weights.

**Scenarios:**

- **Given** a `TokenEmbedding` with default M01Config, **When** `forward(token_ids)` is called with shape `(batch, seq_len)`, **Then** output shape MUST be `(batch, seq_len, 640)`.
- **Given** a `TokenEmbedding`, **When** `output_head(hidden)` is called with shape `(batch, seq_len, 640)`, **Then** output shape MUST be `(batch, seq_len, 32768)`.
- **Given** a `TokenEmbedding`, **When** `output_head(hidden)` is called, **Then** it MUST use `F.linear(hidden, self.embedding.weight)` — the same weight matrix as the embedding layer.
- **Given** a `TokenEmbedding` after forward + backward pass, **When** `embedding.weight.grad` is inspected, **Then** it MUST be non-null (gradient flows through tied weights).
- **Given** a `TokenEmbedding`, **When** `parameters()` is called, **Then** there MUST be exactly one parameter group (the embedding weight).
- **Given** a `TokenEmbedding`, **When** `forward(token_ids)` is called, **Then** embeddings MUST be scaled by `1 / sqrt(d_model)`.

### Requirement: Weight Tying

The output projection head MUST share weights with the input embedding layer. This saves approximately 10.5M parameters (vocab_size × d_model = 32768 × 640 × 2 bytes).

**Scenarios:**

- **Given** a `TokenEmbedding`, **When** `output_head(hidden)` computes `F.linear(hidden, self.embedding.weight)`, **Then** the weight matrix MUST be the same object as `self.embedding.weight`.
- **Given** a `TokenEmbedding` after training, **When** `embedding.weight` is inspected, **Then** the gradient MUST reflect updates from both embedding lookup and output projection.
