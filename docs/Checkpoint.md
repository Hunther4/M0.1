# Canonical Checkpoint System

`TrainingEngineV2` uses `AsyncCheckpointManagerV2` from `src/engine_v2/checkpoint_v2.py`.

## Files and atomic save

Each run can contain:

- `checkpoint.pt`: current canonical checkpoint.
- `checkpoint.pt.sha256`: checksum of the canonical file.
- `checkpoint.previous.pt`: previous canonical generation.
- `checkpoint.previous.pt.sha256`: checksum of the previous generation.

A save writes `checkpoint.pt.tmp`, calculates its checksum, preserves the existing canonical file as the previous generation and atomically replaces `checkpoint.pt`. The manager serializes saves and propagates background write failures through `wait_completion()`.

## Load and recovery

`load_canonical()` waits for an active save, verifies SHA-256 and loads the first valid candidate in this order:

1. `checkpoint.pt`.
2. `checkpoint.previous.pt`.

If neither candidate has a valid sidecar/checksum, loading fails explicitly. Legacy V1/V2 key aliases are normalized at the deserialization boundary.

All loaders retain `weights_only=True`. Current checkpoints serialize `M01Config` as a plain dictionary. Legacy checkpoints that embedded an `M01Config` object are supported through a scoped allowlist and are immediately normalized to a dictionary; unrelated pickle globals remain rejected. The allowlist does not persist after the load.

## Model manipulation

- `merge_checkpoints.py` requires identical configuration, keys, shapes and dtypes. MoE parameters are rejected by default because expert order is permutation-invariant; `copy-first` or `copy-second` may copy the complete expert/router set without interpolation.
- `expand_model.py` preserves the dense/MoE boundary and scales copied attention-output and FFN-down projections by `sqrt(old_layers / new_layers)` by default. Use the explicit `none` strategy only when exact copying is intended.

## State preserved

| Key | Content |
|---|---|
| `step`, `global_tokens` | Training progress counters |
| `model_state` | Model weights |
| `optimizer_state` | Optimizer state |
| `scheduler_state` | LR scheduler state |
| `ema_state` | EMA state when enabled |
| `amp_scaler_state` | AMP state |
| `rng_states` | Python, NumPy, Torch and CUDA/ROCm RNG snapshots |
| `metrics` | Metric registry snapshot |
| `env` | Environment and Git metadata |
| `dataset_hash` | SHA-256 manifest of the source files actually exposed by the training dataset |
| `tokenizer_hash` | SHA-256 of the tokenizer used by the configured data directory |
| `model_config` | Full model configuration |

Dataset and tokenizer fingerprints are cached once per engine instance. If a custom dataset does not expose `source_files`, `dataset_hash` is recorded as `unknown` rather than attributing a different corpus.

## Resume compatibility

```python
engine.resume()
engine.resume("path/to/checkpoint.pt")
```

- No path loads the current run's canonical/backup pair.
- An explicit path loads that checkpoint directly.
- `vocab_size`, `n_layers`, `d_model`, `num_experts` and `num_shared_experts` must match.
- Compatible parameter shapes are loaded; incompatible optimizer/EMA tensors are reinitialized safely.
- `moe_top_k` is intentionally not a shape guard, allowing a deliberate Top-1 to Top-2 routing-policy migration with unchanged expert parameters.

## Bounded automatic recovery

On a failed health check, the engine:

1. Enters `RECOVERING` and waits for checkpoint I/O.
2. Restores model, optimizer, scheduler, AMP, RNG and counters.
3. Repositions the data iterator for deterministic loaders.
4. Reduces optimizer and scheduler LR anchors.
5. Retries from the restored step.

Consecutive failures are bounded by `max_recovery_attempts` (default `3`). Exhaustion transitions to terminal `ERROR` and raises `RuntimeError`; LR is never reduced indefinitely.

Exact batch replay requires a deterministic loader/sampler. Custom shuffled or distributed samplers must persist their own state if exact ordering across rollback is required.
