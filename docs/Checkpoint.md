# Canonical Single Checkpoint System

**Manager:** `AsyncCheckpointManagerV2` (`src/engine_v2/checkpoint_v2.py`). Atomic write:
`checkpoint.pt.tmp` → `os.replace()` → `checkpoint.pt`. No separate `checkpoint.previous.pt` file in
the current engine — recovery rolls back by re-loading the existing canonical `checkpoint.pt` (see
Recovery below).

## Canonical File

- `runs/<run-name>/checkpoints/checkpoint.pt` (single canonical checkpoint per run).

## State Preserved

`TrainingEngineV2.save_checkpoint()` writes exactly these keys:

| Key | Content |
|-----|---------|
| `step`, `global_tokens` | Training progress counters |
| `model_state` | Model weights (`state_dict`) |
| `optimizer_state` | AdamW optimizer state |
| `scheduler_state` | LR scheduler state |
| `ema_state` | EMA shadow weights (if EMA enabled) |
| `amp_scaler_state` | AMP GradScaler state |
| `rng_states` | Python / NumPy / Torch / ROCm RNG snapshots |
| `metrics` | MetricRegistry snapshot |
| `env` | Environment metadata (GPU, torch, ROCm) |
| `dataset_hash` | SHA-256 of `data/spanish_pretrain.txt` (if present) |
| `tokenizer_hash` | **SHA-256 of `data/tokenizers/tokenizer.json`** |
| `model_config` | **Full `M01Config` dict** (architecture fingerprint) |

## Resume API

```python
engine.resume()                 # resumes current run's canonical checkpoint.pt
engine.resume("path/to/ckpt.pt") # stacks knowledge onto an explicit checkpoint (layer-stacking)
```

- With an explicit, existing path → loads that file directly (used to **stack** knowledge in layers).
- With `None` (or missing file) → loads the current run's canonical `checkpoint.pt`.

## `_assert_config_compatible()` — Stacking Guard

Before loading, `resume()` compares the saved `model_config` against the current model. It raises
`ValueError` if any of these differ:

- `vocab_size`
- `n_layers`
- `d_model`
- `num_experts`
- `num_shared_experts`
- `moe_top_k`

This prevents silently corrupting a "stacked" model by resuming onto an architecturally incompatible
checkpoint.

## Automatic Recovery (NaN/Inf/Overflow)

On a health-check failure mid-`fit()`:
1. Flush CUDA cache.
2. `self.resume()` → restore the last clean canonical `checkpoint.pt`.
3. Halve the LR (`lr *= 0.5`) and continue.

## Graceful Shutdown

Intercepts `SIGINT`/`SIGTERM`, sets a stop flag, saves the canonical `checkpoint.pt`, flushes loggers,
and exports the profiler.
