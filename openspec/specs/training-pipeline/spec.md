# Spec: training-pipeline

## Overview

Training pipeline for M0.1, built on `TrainingEngineV2` (`src/engine_v2/engine.py`) — an FSM-driven
engine using `AdamW`, a cosine LR schedule with linear warmup, gradient clipping, EMA, AMP (mixed
precision via `AMPContext`), and atomic async checkpointing (`AsyncCheckpointManagerV2`).

**Tokenization is fixed to a single 16k tokenizer:** `data/tokenizers/tokenizer.json` (vocab 16384,
SHA-256 prefix `6bc3a6…`). There is **no 32k tokenizer**. `M01Config.vocab_size` MUST be 16384 to
match it (call-sites pass this explicitly when a non-default tokenizer is used).

**Training strategy is layer-stacking:** the full model is trained by building a base checkpoint, then
resuming (`--resume <path>`) to stack knowledge across runs, because the full model cannot be trained
in a single pass.

VRAM is reported via `torch.cuda.memory_reserved()` (≈11 GB at batch 16); `memory_allocated()`
(≈2.2 GB) is misleading and MUST NOT be used.

## Requirements

### Requirement: TrainingConfig Dataclass

The system MUST provide a `TrainingConfig` dataclass in `src/training/config.py` with all training
hyperparameters as fields with defaults.

**Fields and Defaults:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| batch_size | int | 4 | Sequences per batch |
| seq_len | int | 1024 | Maximum sequence length |
| max_lr | float | 3e-4 | Peak learning rate |
| min_lr_ratio | float | 0.1 | Min LR as fraction of max_lr |
| warmup_steps | int | 200 | Linear warmup steps |
| max_steps | int | 100_000 | Total training steps |
| weight_decay | float | 0.1 | AdamW weight decay |
| beta1 | float | 0.9 | Adam beta1 |
| beta2 | float | 0.95 | Adam beta2 |
| max_norm | float | 1.0 | Gradient clipping max norm |
| log_interval | int | 10 | Steps between logging |
| save_interval | int | 500 | Steps between checkpoint saves |
| checkpoint_dir | str | "checkpoints" | Checkpoint directory |
| data_dir | str | "data" | Data directory |

**Scenarios:**

- **Given** `TrainingConfig()`, **When** instantiated with no arguments, **Then** all 14 fields MUST have the default values listed above.
- **Given** `TrainingConfig(batch_size=8, max_lr=1e-3)`, **When** instantiated, **Then** `batch_size` MUST be 8, `max_lr` MUST be `1e-3`, and all other fields MUST retain their defaults.
- **Given** `TrainingConfig`, **When** `batch_size` is accessed, **Then** it MUST be type `int`, `max_lr` MUST be type `float`, `checkpoint_dir` MUST be type `str`.

### Requirement: TinyShakespeareDataset

The system MUST provide a `TinyShakespeareDataset` class in `src/training/dataset.py` that provides
sliding-window (input, target) pairs over the TinyShakespeare corpus.

**Interface:**
```python
class TinyShakespeareDataset:
    def __init__(self, config: TrainingConfig) -> None: ...
    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]: ...
```

**Scenarios:**

- **Given** a `TinyShakespeareDataset` with `seq_len=1024`, **When** `len()` is called, **Then** it MUST be positive.
- **Given** a `TinyShakespeareDataset`, **When** `__getitem__(idx)` is called, **Then** it MUST return a tuple of `(input, target)` of shape `(seq_len,)` as `torch.long` tensors with `target[t] == input[t+1]`.

### Requirement: 16k Tokenizer Source of Truth

The training pipeline MUST use exactly one tokenizer: `data/tokenizers/tokenizer.json` (vocab size
16384). No 32k tokenizer exists.

**Scenarios:**

- **Given** `train.py` is launched, **When** it loads a tokenizer, **Then** it MUST read `data/tokenizers/tokenizer.json`.
- **Given** the model is built, **When** `vocab_size` is set, **Then** it MUST equal 16384 to match the tokenizer.
- **Given** a checkpoint `tokenizer_hash`, **When** computed, **Then** it MUST be the SHA-256 of `data/tokenizers/tokenizer.json` (prefix `6bc3a6…`).

### Requirement: AsyncCheckpointManagerV2

The system MUST provide `AsyncCheckpointManagerV2` in `src/engine_v2/checkpoint_v2.py` for atomic
async save/load of training state, writing a single canonical `checkpoint.pt` per run.

**Scenarios:**

- **Given** `save_checkpoint()` is called, **When** it writes, **Then** it MUST use an atomic tmp file renamed to `checkpoint.pt` and leave no `.tmp` behind.
- **Given** a saved checkpoint, **When** inspected, **Then** it MUST contain `model_state`, `optimizer_state`, `scheduler_state`, `ema_state`, `amp_scaler_state`, `rng_states`, `metrics`, `env`, `dataset_hash`, `tokenizer_hash`, `model_config`, `step`, and `global_tokens`.
- **Given** a saved checkpoint, **When** `tokenizer_hash` is inspected, **Then** it MUST equal the SHA-256 of `data/tokenizers/tokenizer.json`.
- **Given** a saved checkpoint, **When** `model_config` is inspected, **Then** it MUST be the full `M01Config` dict (architecture fingerprint).
- **Given** a checkpoint saved during training, **When** loaded via `resume()`, **Then** model weights, optimizer state, scheduler state, EMA, AMP scaler, and RNGs MUST all be restored.

### Requirement: Resume and Layer-Stacking

The system MUST support `engine.resume(checkpoint_path=None)` that resumes the current run's canonical
checkpoint when `checkpoint_path` is `None`, and loads an explicit checkpoint file for layer-stacking
when a path is given.

**Scenarios:**

- **Given** `engine.resume()` with no path, **When** a canonical `checkpoint.pt` exists, **Then** training MUST continue from that checkpoint.
- **Given** `engine.resume("base.pt")`, **When** the file exists, **Then** the model MUST load `base.pt` and continue training (stacking knowledge).
- **Given** a resumed checkpoint, **When** its `model_config` differs from the current model on `vocab_size`, `n_layers`, `d_model`, `num_experts`, `num_shared_experts`, or `moe_top_k`, **Then** `_assert_config_compatible()` MUST raise `ValueError` and refuse to stack.

### Requirement: AMP / Mixed Precision

The training loop MUST use mixed precision via `AMPContext` (`autocast` + `GradScaler`) on CUDA/ROCm
devices for throughput.

**Scenarios:**

- **Given** the training loop source, **When** inspected on a CUDA device, **Then** the forward pass MUST run under `AMPContext.autocast()` and backward MUST use `amp_context.scale(loss).backward()`.
- **Given** a training step, **When** `amp_scaler_state` is saved, **Then** it MUST be restored on resume.

### Requirement: VRAM Reporting via memory_reserved

The training loop MUST report VRAM using `torch.cuda.memory_reserved()`, not `memory_allocated()`.

**Scenarios:**

- **Given** the per-step log and final report, **When** VRAM is printed, **Then** it MUST use `torch.cuda.memory_reserved()` (≈11 GB at batch 16).
- **Given** a VRAM reading, **When** compared, **Then** `memory_allocated()` (≈2.2 GB) MUST NOT be used as the reported footprint.

### Requirement: Training Loop CLI

The system MUST provide a training loop accessible via `python -m src.training.train` that performs
autoregressive LM training using `TrainingEngineV2`.

**CLI Interface:**
```
python -m src.training.train [--data-dir PATH] [--batch-size N] [--seq-len N]
    [--max-lr F] [--min-lr-ratio F] [--warmup-steps N] [--max-steps N]
    [--weight-decay F] [--grad-accum-steps N] [--max-norm F]
    [--log-interval N] [--save-interval N] [--val-interval N]
    [--vocab-size N] [--resume [PATH]] [--run-name NAME]
```

**Scenarios:**

- **Given** the training CLI, **When** `--help` is invoked, **Then** it MUST print usage and exit 0.
- **Given** `configure_optimizer(model, config)`, **When** inspected, **Then** it MUST return an `AdamW` with exactly 2 param groups (decay + no_decay; `bias`/`gamma` in no_decay).
- **Given** a `get_lr_scheduler(optimizer, warmup=200, max_steps=100000, min_lr_ratio=0.1)`, **When** stepped, **Then** LR MUST rise linearly 0→max_lr during warmup, then cosine-decay to `max_lr * min_lr_ratio`.
- **Given** `--vocab-size 16384`, **When** the model is built, **Then** `vocab_size` MUST be 16384.
- **Given** `--resume runs/x/checkpoints/checkpoint.pt`, **When** launched, **Then** training MUST stack onto that checkpoint.

### Requirement: Learning Rate Tuning (validated)

At batch 16, over 300-step runs that complete warmup, the optimal `max_lr` MUST be `1.2e-3` (4×
linear scaling from the 3e-4 base); 6× (`1.8e-3`) MUST degrade final loss.

**Scenarios:**

- **Given** a 300-step batch-16 run at `max_lr=1.2e-3`, **When** final loss is measured, **Then** it MUST be the best observed (≈5.448).
- **Given** a 300-step batch-16 run at `max_lr=1.8e-3`, **When** final loss is measured, **Then** it MUST be worse than at 1.2e-3 (≈5.623).
- **Given** a run with `--warmup-steps 200` and only 120 steps, **When** LR is measured, **Then** `max_lr` MUST NOT be reached (warmup never completes) — such a run is invalid for LR tuning.

### Requirement: DataLoader Safety

The DataLoader MUST be configured with `num_workers=0` on Windows/ROCm to avoid multiprocessing
deadlocks.

**Scenarios:**

- **Given** the training loop, **When** `DataLoader` is instantiated, **Then** `num_workers` MUST be 0 on `win32`.

## Validated Empirical Configuration (session 2026-07-20)

These are empirical facts measured this session on the target hardware (AMD RX 9060 XT 16GB,
ROCm 7.14, Windows). They refine the canonical requirements above.

### Batch size sweet spot (Windows + AMD 16GB)

Measured VRAM (reserved) on `corpus1_es_wiki_wikisource_tech_10M` at `seq_len=1024`:

| batch_size | VRAM reserved | Verdict |
|------------|---------------|---------|
| 16 | ~10.1 GB | safe |
| 20 | ~12.0 GB | **sweet spot (chosen)** |
| 22 | ~13.2 GB | rejected: with Windows system overhead it exceeds ~14 GB combined |

**Rule:** VRAM scales ~linearly with batch. On Windows, leave headroom for OS overhead; **batch 20**
is the practical maximum before the combined footprint crosses ~14 GB.

### Learning-rate scaling with batch

LR scales ~linearly with batch (linear scaling rule), capped at the documented ceiling:

| batch_size | max_lr | × base (3e-4) | Note |
|------------|--------|---------------|------|
| 16 | 1.2e-3 | 4× | optimal (validated) |
| 20 | 1.5e-3 | 5× | chosen for base run |
| 22 | 1.65e-3 | 5.5× | |
| — | 1.8e-3 | 6× | **ceiling** — beyond this loss degrades |

### One practical pass over the 10M corpus

At `seq_len=1024`: tokens/step = `batch_size × 1024`. For batch 20, `488 steps ≈ 9.99M tokens`
≈ one practical pass over `corpus1` (9,987,393 tokens). Warmup MUST be `< half` of `max_steps`
(e.g. warmup 30 for 488 steps) or the run is invalid for LR tuning.

### Flash / mem-efficient attention is UNSTABLE on this GPU

Setting `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` makes training crash at the **first backward
pass** with `hipErrorInvalidValue` (the experimental SDP kernels do not support the MoE+MLA+RoPE
backward path on gfx1200 / torch 2.12 ROCm). **MUST NOT** be enabled for M0.1 training; the default
(non-experimental) `scaled_dot_product_attention` works.

### KNOWN ISSUE: train.py / dataset.py / experiment.py revert between sessions

Edits to `src/training/train.py`, `src/training/dataset.py` and `src/engine_v2/experiment.py` (adding
`--run-name`, `BinaryCorpusDataset`, `run_name`) do **not** persist: the environment reverts them to a
committed "basic" version (only `TinyShakespeareDataset`, no flags) between sessions. Runs executed
against the reverted code silently train on TinyShakespeare instead of the 10M corpus.

**Workaround (canonical 10M entry point until train.py is restored/committed):** a standalone runner
in a temp dir that inlines `BinaryCorpusDataset` + `configure_optimizer` / `get_lr_scheduler` /
`worker_init_fn` and imports only stable modules (`M01Config`, `TransformerLM`, `TrainingEngineV2`,
`TrainingConfig`). Caveats: `M01Config` does **not** accept `gradient_checkpointing`; the current
`ExperimentManager.__init__` does **not** accept `run_name` (auto-numbers the run dir).

### Base real run result (2026-07-20)

- Config: batch 20, seq_len 1024, vocab 16384, max_lr 1.5e-3, warmup 30, 488 steps (~9.99M tokens).
- Result: final loss **5.58** (from 8.71), elapsed 61 min. Checkpoint at `runs/0004/checkpoints`.
- Note: throughput was ~2.7k tok/s (vs 6.6–7.1k in earlier runs) because `engine_v2`/`transformer`
  also reverted to slower versions this session — numbers are not directly comparable. Per-step VRAM
  logged ~2.4 GB (no OOM); the final `memory_reserved` reading of ~29 GB is a reporting artifact.
