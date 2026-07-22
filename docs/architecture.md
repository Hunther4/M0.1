# M0.1 Architecture Specification

M0.1 is a decoder-only Transformer using `TrainingEngineV2`, MLA by default and DeepSeek-style shared plus routed experts.

## Default model

| Field | Default | Notes |
|---|---:|---|
| `vocab_size` | 16384 | Must match the tokenizer |
| `context_length` | 8192 | Maximum cache length |
| `d_model` | 640 | Model width |
| `n_heads` | 10 | `d_head=64` |
| `n_layers` | 12 | Transformer blocks |
| `num_experts` | 4 | Routed experts |
| `num_shared_experts` | 1 | Always-active shared expert |
| `moe_top_k` | 2 | Up to two accepted routed experts per token |
| `capacity_factor` | 1.25 | Per-expert capacity after warmup |
| `use_mla` | True | MLA is the active default attention mode |
| `attention_backend` | `auto` | `auto`, `flash`, `efficient` or `math` |
| `initializer_range` | 0.02 | Base standard deviation for model initialization |
| `scale_residual_projections` | True | Scale residual projection initialization by model depth |

When both `use_mla` and `use_hybrid_attention` are true, MLA has precedence. Set `use_mla=False` to select hybrid CSA/HCA explicitly.

## MoE routing

The router keeps selected and accepted assignments separate. Capacity is applied per expert, accepted weights are renormalized per token and expert outputs are accumulated with weighted addition. Tokens rejected by every routed expert retain the shared-expert path. Routing telemetry exposes accepted assignments, dropped tokens and expert load.

## Attention and cache modes

- MHA uses the standard preallocated `KVCache`.
- Hybrid attention uses `HybridKVCache`, preserving both CSA and HCA projections so historical positions can switch to HCA outside the local window.
- MLA uses `MLAKVCache`, storing compressed `c_KV` and positional `k_R`. Cached decoding computes content scores from the latent directly and projects V after latent aggregation, avoiding historical K/V expansion.
- `AttentionDispatcher` selects SDPA backends and falls back to the math backend when an explicitly requested optimized backend is unavailable.

`build_attention_cache()` is the canonical cache factory used by generation.

Generation rejects requests where prompt tokens plus `max_gen_len` exceed `context_length`; it does not silently truncate the requested continuation.

## Numerical stability

- RMSNorm computes the RMS in FP32, then casts both normalized activations and `gamma` to the activation dtype for the final multiply. FP16/BF16 inputs therefore remain mixed precision.
- The tied embedding matrix uses the same explicit scale for input lookup and output projection. Its default remains `1.0`.
- Attention output projections and SwiGLU down projections use `initializer_range / sqrt(2 * n_layers)`. This includes every shared and routed MoE expert. Existing checkpoints are unaffected because loading replaces initialized weights.

## Engine FSM

Normal execution follows `INIT -> LOAD -> TRAIN`, with `VALIDATE` and `SAVE` as needed, then `FINISHED`. Health failures use `TRAIN -> RECOVERING -> TRAIN`; exhausted recovery enters terminal `ERROR`.

## Checkpoints

Checkpoints include model/optimizer/scheduler/EMA/AMP/RNG state, environment metadata, full model configuration and fingerprints of the actual dataset source manifest and tokenizer. Canonical and previous generations are checksum-validated. See `docs/Checkpoint.md`.
