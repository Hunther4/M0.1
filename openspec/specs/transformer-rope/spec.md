# Spec: transformer-rope

## Overview

Rotary Position Embedding (RoPE) via explicit sin/cos frequency computation and 2D rotation of dimension pairs.

## Requirements

### Requirement: RotaryPositionalEmbedding Module

The system MUST provide a `RotaryPositionalEmbedding` module in `src/transformer/rope.py` that applies rotary positional encoding to query and key tensors.

**Scenarios:**

- **Given** a `RotaryPositionalEmbedding` with default M01Config, **When** `forward(x, offset=0)` is called, **Then** the output shape MUST equal the input shape `(batch, seq_len, d_head)`.
- **Given** a `RotaryPositionalEmbedding`, **When** `forward(x, offset=0)` is called, **Then** position 0 MUST produce an identity rotation (output ≈ input within atol=1e-5).
- **Given** a `RotaryPositionalEmbedding`, **When** `forward(x, offset=1)` is called, **Then** each dimension pair MUST be rotated by angle `m·θᵢ` where `m=1` and `θᵢ = 1 / 10000^(2i/d_head)`.
- **Given** a `RotaryPositionalEmbedding`, **When** `forward(x, offset=0)` is called with input `[a, b, c, d]` (d_head=4), **Then** output MUST be `[a, b, c, d]` (cos(0)=1, sin(0)=0).
- **Given** a `RotaryPositionalEmbedding`, **When** `forward(x, offset=1)` is called with input `[a, b, c, d]` (d_head=4), **Then** output MUST be `[a·cos(θ₁) - b·sin(θ₁), a·sin(θ₁) + b·cos(θ₁), c·cos(θ₂) - d·sin(θ₂), c·sin(θ₂) + d·cos(θ₂)]`.

### Requirement: Frequency Precomputation

Frequencies MUST be precomputed at initialization using the formula `θᵢ = 1 / rope_theta^(2i/d_head)` for `i ∈ [0, d_head/2)`.

**Scenarios:**

- **Given** default M01Config (`rope_theta=10000.0`, `d_head=64`), **When** `RotaryPositionalEmbedding` is initialized, **Then** frequency tensor shape MUST be `(d_head//2,)` = `(32,)`.
- **Given** default M01Config, **When** frequencies are precomputed, **Then** `θ₀` MUST equal `1.0 / 10000.0^(0/64) = 1.0` (not 1/10000).

### Requirement: Educational Implementation

The implementation MUST use explicit sin/cos rotation rather than the complex number trick, to show the actual rotation math for educational clarity.

**Scenarios:**

- **Given** the `RotaryPositionalEmbedding` source code, **When** inspected, **Then** it MUST use `torch.cos` and `torch.sin` explicitly rather than `torch.view_as_complex` / `torch.view_as_real`.
- **Given** the source code, **When** inspected, **Then** comments MUST document the rotation formula: `rot(x, m) = x_even·cos(mθ) - x_odd·sin(mθ), x_even·sin(mθ) + x_odd·cos(mθ)`.
