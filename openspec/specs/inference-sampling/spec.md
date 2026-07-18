# Spec: inference-sampling

## Overview

Sampling strategies for token selection during autoregressive generation. Provides temperature scaling, top-k filtering, and top-p (nucleus) sampling as composable transformations over raw logits.

## Requirements

### Requirement: Temperature Scaling

The system MUST support temperature scaling to control the randomness of sampling. Temperature < 1.0 sharpens the distribution (more deterministic), temperature > 1.0 flattens it (more random).

**Interface:**
```python
def sample(
    logits: Tensor,             # (1, 1, V) raw logits
    temperature: float = 1.0,   # scaling factor
    top_k: int = 0,             # 0 = no filtering
    top_p: float = 1.0,         # 1.0 = no filtering
) -> int:                       # sampled token ID
```

**Scenarios:**

- **Given** `sample(logits, temperature=1.0)`, **When** called, **Then** the output MUST be a valid token ID (integer in [0, vocab_size)).
- **Given** `sample(logits, temperature=1.0)`, **When** called multiple times with the same logits, **Then** the output MAY vary (stochastic sampling).
- **Given** `sample(logits, temperature=0.1)`, **When** called, **Then** the output MUST be the argmax of the logits (near-deterministic).
- **Given** `sample(logits, temperature=2.0)`, **When** called, **Then** the output distribution MUST be flatter than temperature=1.0.

### Requirement: Top-k Filtering

The system MUST support top-k filtering that restricts sampling to the k most probable tokens. Tokens outside top-k are masked to -inf before softmax.

**Scenarios:**

- **Given** `sample(logits, top_k=10)`, **When** called, **Then** the sampled token MUST be among the top-10 highest logits.
- **Given** `sample(logits, top_k=0)`, **When** called, **Then** top-k filtering MUST be disabled (all tokens eligible).
- **Given** `sample(logits, top_k=1)`, **When** called, **Then** the output MUST be the argmax of the logits (greedy).

### Requirement: Top-p (Nucleus) Sampling

The system MUST support top-p sampling that restricts sampling to the smallest set of tokens whose cumulative probability exceeds p.

**Scenarios:**

- **Given** `sample(logits, top_p=0.9)`, **When** called, **Then** the sampled token MUST be from the nucleus (tokens whose cumulative probability ≥ 0.9).
- **Given** `sample(logits, top_p=1.0)`, **When** called, **Then** top-p filtering MUST be disabled (all tokens eligible).
- **Given** `sample(logits, top_p=0.0)`, **When** called, **Then** the output MUST be the argmax of the logits (only most probable token survives).

### Requirement: Combined Filtering

The system MUST support combining temperature, top-k, and top-p filtering. Filtering order: temperature scaling → top-k → top-p → multinomial sample.

**Scenarios:**

- **Given** `sample(logits, temperature=0.8, top_k=50, top_p=0.95)`, **When** called, **Then** all three filters MUST be applied in order.
- **Given** `sample(logits, temperature=0.8, top_k=50, top_p=0.95)`, **When** called multiple times, **Then** outputs MUST be valid token IDs.

### Requirement: Stateless Function

The `sample` function MUST be stateless — each call is independent and MUST NOT depend on previous calls.

**Scenarios:**

- **Given** `sample(logits_a) → token_a` and `sample(logits_b) → token_b`, **When** both are called, **Then** `token_b` MUST NOT depend on `token_a`.
- **Given** `sample`, **When** inspected, **Then** it MUST NOT maintain any internal state between calls.
