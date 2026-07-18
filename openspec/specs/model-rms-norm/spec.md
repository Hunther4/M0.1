# Spec: model-rms-norm

## Overview

RMSNorm is a normalization layer that normalizes inputs by root mean square along the last dimension, then scales by a learnable gamma parameter. No shift/bias parameter. Formula: `y = (x / sqrt(mean(x²) + eps)) * gamma`.

Reference: https://arxiv.org/abs/1910.07467

## Requirements

### Requirement: RMSNorm Module

The system MUST provide an `RMSNorm` module in `src/model/rms_norm.py` that normalizes input tensors along the last dimension.

**Interface:**
```python
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...  # (..., d_model) → (..., d_model)
```

**Scenarios:**

- **Given** an `RMSNorm` with `d_model=640`, **When** `forward(x)` is called with shape `(batch, seq_len, 640)`, **Then** output shape MUST be `(batch, seq_len, 640)`.
- **Given** an `RMSNorm` with `d_model=128`, **When** `forward(x)` is called with shape `(batch, seq_len, 128)`, **Then** output shape MUST be `(batch, seq_len, 128)`.
- **Given** an `RMSNorm`, **When** `loss.sum().backward()` is called, **Then** backward MUST succeed and `x.grad` MUST be non-null.
- **Given** an `RMSNorm` with `d_model=640`, **When** `gamma.grad` is inspected after backward, **Then** it MUST be non-null (gamma is learnable).
- **Given** an `RMSNorm`, **When** `forward(x)` is called with zero input, **Then** output MUST be zero (eps prevents division by zero NaN).
- **Given** an `RMSNorm` with `eps=1e-10`, **When** `forward(x)` is called, **Then** no NaN values MUST appear in the output.
- **Given** an `RMSNorm`, **When** instantiated, **Then** `gamma` MUST be initialized to `torch.ones(d_model)`.
- **Given** an `RMSNorm` with `d_model=640`, **When** `gamma` is inspected, **Then** `gamma.numel()` MUST equal `640`.

### Requirement: Normalization Formula

The RMSNorm computation MUST follow the formula: `y = x / sqrt(mean(x²) + eps) * gamma`, computed along the last dimension with `keepdim=True`.

**Scenarios:**

- **Given** an `RMSNorm`, **When** `forward(x)` is called, **Then** the normalized value before gamma scaling MUST have RMS ≈ 1 (within tolerance).
- **Given** an `RMSNorm`, **When** `forward(x)` is called, **Then** eps MUST be added inside the sqrt: `sqrt(mean(x²) + eps)`, NOT `sqrt(mean(x²)) + eps`.
- **Given** an `RMSNorm`, **When** `forward(x)` is called with various `d_model` values (64, 128, 640, 1024), **Then** output shape MUST match input shape for each.

### Requirement: No Shift Parameter

RMSNorm MUST NOT have a shift/bias parameter. Only the scale parameter `gamma` is learnable.

**Scenarios:**

- **Given** the `RMSNorm` source code, **When** inspected, **Then** `state_dict()` MUST contain exactly one entry: `"gamma"`.
- **Given** the `RMSNorm` source code, **When** inspected, **Then** no bias or beta parameter MUST exist in the module.
