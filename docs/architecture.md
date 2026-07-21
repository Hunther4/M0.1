# M0.1 Architecture Specification

**Core engine:** `TrainingEngineV2` (`src/engine_v2/engine.py`) — an enterprise-grade, FSM-driven
training engine. Built on a Finite State Machine: `INIT → LOAD → TRAIN → VALIDATE → SAVE → FINISHED`.

**Hardware:** Native PyTorch HIP/ROCm on AMD Radeon RX 9060 XT 16 GB (gfx1200, ROCm 7.14).

## Model Configuration (M01Config)

| Field | Value | Note |
|-------|-------|------|
| `vocab_size` | **16384** | Must match `data/tokenizers/tokenizer.json` (dataclass default corrected to 16384). |
| `d_model` | 640 | |
| `n_heads` | 10 | |
| `n_layers` | 12 | Guarded by `_assert_config_compatible` on resume. |
| `num_experts` | 4 | Routed experts (MoE). Guarded on resume. |
| `num_shared_experts` | 1 | Guarded on resume. |
| `moe_top_k` | 1 | Guarded on resume. |
| `context_length` | 8192 | |
| `use_mla` | True | Multi-head Latent Attention. |

The full model is **trained via layer-stacking** (continual resume), not in a single pass.

## Tokenizer (real, single source of truth)

- **File:** `data/tokenizers/tokenizer.json`
- **Vocab:** 16384
- **SHA-256 prefix:** `6bc3a6…`
- There is **no 32k tokenizer**. `train.py` and `evaluate.py` both load this file.

## Checkpoint Persistence

`TrainingEngineV2.save_checkpoint()` persists a canonical `checkpoint.pt` (via
`AsyncCheckpointManagerV2`, atomic write). Each checkpoint records:

- `model_state`, `optimizer_state`, `scheduler_state`
- `ema_state`, `amp_scaler_state`
- `rng_states` (Python / NumPy / Torch / ROCm RNGs)
- `metrics`, `env` (environment metadata)
- `dataset_hash` (from `data/spanish_pretrain.txt` if present)
- **`tokenizer_hash`** — SHA-256 of `data/tokenizers/tokenizer.json`
- **`model_config`** — full `M01Config` dict, used by `_assert_config_compatible()` on resume

This makes every checkpoint self-describing: the tokenizer version and architecture are baked into the
file, so stacking onto the wrong checkpoint fails loudly instead of silently corrupting weights.

See `docs/Checkpoint.md` for the resume / compatibility contract.
