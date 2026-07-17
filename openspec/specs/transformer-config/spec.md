# Spec: transformer-config

## Overview

Dataclass holding all M0.1 model hyperparameters with validated defaults and computed fields.

## Requirements

### Requirement: M01Config Dataclass

The system MUST provide an `M01Config` dataclass in `src/transformer/config.py` that holds all model hyperparameters.

**Default Values:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| vocab_size | 32768 | Token vocabulary size |
| context_length | 8192 | Maximum sequence length |
| d_model | 640 | Embedding dimension |
| n_heads | 10 | Number of attention heads |
| d_ff | 1728 | Feedforward hidden dimension |
| n_layers | 12 | Number of transformer layers |
| rope_theta | 10000.0 | RoPE base frequency |
| num_experts | 1 | Number of MoE experts |
| dropout | 0.0 | Dropout rate |

**Scenarios:**

- **Given** default M01Config, **When** instantiated, **Then** all parameters MUST have the default values listed above.
- **Given** `d_model=640` and `n_heads=10`, **When** instantiated, **Then** `d_head` MUST be computed as `64`.
- **Given** `d_model=640` and `n_heads=10`, **When** instantiated, **Then** `d_model % n_heads == 0` MUST hold (validation in `__post_init__`).
- **Given** custom values for any parameter, **When** instantiated, **Then** those values MUST override defaults.
- **Given** invalid values (e.g., `d_model=640, n_heads=7`), **When** instantiated, **Then** `__post_init__` MUST raise an assertion error.
