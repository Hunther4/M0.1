# Spec: training-pipeline

## Overview

Training pipeline for M0.1 including hyperparameter configuration (TrainingConfig), sliding-window dataset (TinyShakespeareDataset), atomic checkpointing (CheckpointManager), and the full training loop CLI with AdamW optimizer, cosine LR schedule, and gradient clipping. All training runs in fp32 on a single device.

## Requirements

### Requirement: TrainingConfig Dataclass

The system MUST provide a `TrainingConfig` dataclass in `src/training/config.py` with all training hyperparameters as fields with defaults.

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
- **Given** `TrainingConfig`, **When** inspected, **Then** it MUST be a `dataclass` with `__repr__` and field equality.

### Requirement: TinyShakespeareDataset

The system MUST provide a `TinyShakespeareDataset` class in `src/training/dataset.py` that provides sliding-window (input, target) pairs over the TinyShakespeare corpus.

**Interface:**
```python
class TinyShakespeareDataset:
    def __init__(self, config: TrainingConfig) -> None: ...
    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]: ...
```

**Scenarios:**

- **Given** a `TinyShakespeareDataset` with `seq_len=1024`, **When** `len()` is called, **Then** it MUST be positive.
- **Given** a `TinyShakespeareDataset`, **When** `len()` is called, **Then** it MUST equal `total_tokens - seq_len`.
- **Given** a `TinyShakespeareDataset`, **When** `__getitem__(idx)` is called, **Then** it MUST return a tuple of `(input, target)`.
- **Given** a `TinyShakespeareDataset`, **When** `__getitem__(idx)` is called, **Then** `input` MUST be a `torch.long` tensor.
- **Given** a `TinyShakespeareDataset`, **When** `__getitem__(idx)` is called, **Then** `target` MUST be a `torch.long` tensor.
- **Given** a `TinyShakespeareDataset`, **When** `__getitem__(idx)` is called, **Then** `input` shape MUST be `(seq_len,)`.
- **Given** a `TinyShakespeareDataset`, **When** `__getitem__(idx)` is called, **Then** `target` shape MUST be `(seq_len,)`.
- **Given** a `TinyShakespeareDataset`, **When** `__getitem__(idx)` is called, **Then** `target[t]` MUST equal `input[t+1]` for all `t` (autoregressive shift).
- **Given** a `TinyShakespeareDataset`, **When** `__getitem__` is called with two different indices, **Then** the returned pairs MUST differ.
- **Given** a `TinyShakespeareDataset`, **When** `__getitem__(len-1)` is called, **Then** it MUST NOT raise IndexError.
- **Given** a `TinyShakespeareDataset` wrapped in a `DataLoader(batch_size=4, num_workers=0)`, **When** iterated, **Then** batches MUST have shape `(4, seq_len)` for both input and target.
- **Given** a `TinyShakespeareDataset` with `DataLoader(num_workers=0)`, **When** a single batch is fetched, **Then** it MUST work without deadlock or error.

### Requirement: CheckpointManager

The system MUST provide a `CheckpointManager` class in `src/training/checkpoint.py` for atomic save/load of training state.

**Interface:**
```python
class CheckpointManager:
    def __init__(self, checkpoint_dir: str) -> None: ...
    def save(self, step: int, model: nn.Module, optimizer: optim.Optimizer,
             scheduler: optim.lr_scheduler.LRScheduler, loss: float,
             config: dict, epoch: int = 0) -> None: ...
    def load(self, model: nn.Module, optimizer: optim.Optimizer,
             scheduler: optim.lr_scheduler.LRScheduler) -> dict: ...
```

**Scenarios:**

- **Given** a `CheckpointManager` with a new directory path, **When** instantiated, **Then** the directory MUST be created automatically.
- **Given** a `CheckpointManager` with an existing directory, **When** instantiated, **Then** it MUST accept the existing directory without error.
- **Given** a `CheckpointManager`, **When** `save()` is called, **Then** a `checkpoint.pt` file MUST exist in the checkpoint directory.
- **Given** a `CheckpointManager`, **When** `save()` is called, **Then** no `.checkpoint.tmp` file MUST remain (cleanup after atomic write).
- **Given** a `CheckpointManager`, **When** `save()` is called, **Then** the saved checkpoint dict MUST contain exactly these keys: `epoch`, `step`, `loss`, `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `config`.
- **Given** a `CheckpointManager`, **When** `save(step=99)` is called, **Then** the saved checkpoint's `step` MUST be 99.
- **Given** a `CheckpointManager`, **When** `save(loss=3.14159)` is called, **Then** the saved checkpoint's `loss` MUST be 3.14159.
- **Given** a `CheckpointManager`, **When** `save()` is called, **Then** it MUST use atomic write: write to `.checkpoint.tmp` first, then `os.replace()` to `checkpoint.pt`.
- **Given** a `CheckpointManager`, **When** `load()` is called after `save()`, **Then** it MUST return a dict with `epoch`, `step`, `loss`, and `config` keys.
- **Given** a `CheckpointManager`, **When** `save()` then `load()` is called, **Then** the model weights MUST match the saved weights (save → modify model → load restores original).
- **Given** a `CheckpointManager`, **When** `save()` then `load()` is called, **Then** the optimizer state MUST be restored.
- **Given** a `CheckpointManager`, **When** `save()` then `load()` is called, **Then** the scheduler state MUST be restored.
- **Given** a `CheckpointManager` with no saved checkpoint, **When** `load()` is called, **Then** it MUST raise `FileNotFoundError`.

### Requirement: Training Loop

The system MUST provide a training loop accessible via `python -m src.training.train` that performs autoregressive language model training in fp32.

**CLI Interface:**
```
python -m src.training.train [--batch-size N] [--seq-len N] [--max-lr F]
    [--min-lr-ratio F] [--warmup-steps N] [--max-steps N]
    [--weight-decay F] [--beta1 F] [--beta2 F] [--max-norm F]
    [--log-interval N] [--save-interval N]
    [--checkpoint-dir PATH] [--data-dir PATH]
```

**Scenarios:**

- **Given** the training CLI, **When** `--help` is invoked, **Then** it MUST print usage information and exit with code 0.
- **Given** a `configure_optimizer(model, config)` call, **When** the optimizer is inspected, **Then** it MUST be an `AdamW` instance with exactly 2 param groups.
- **Given** `configure_optimizer`, **When** param groups are inspected, **Then** the first group MUST have `weight_decay > 0` (decay group) and the second MUST have `weight_decay = 0.0` (no_decay group).
- **Given** a model with bias and gamma parameters, **When** `configure_optimizer` is called, **Then** parameters named `bias` or `gamma` MUST be in the `no_decay` group.
- **Given** a `get_lr_scheduler(optimizer, warmup=200, max_steps=100000, min_lr_ratio=0.1)`, **When** LR values are inspected after each `scheduler.step()`, **Then** LR MUST increase linearly from 0 to `max_lr` during warmup, then decay following a cosine curve to `max_lr * min_lr_ratio`.
- **Given** a training step, **When** `loss.backward()` is called, **Then** gradients MUST be clipped to `max_norm=1.0` via `clip_grad_norm_`.
- **Given** a loss value that is NaN or Inf, **When** detected before `optimizer.step()`, **Then** training MUST abort gracefully.

### Requirement: Integrated Training

The full training pipeline MUST work end-to-end: build model → load dataset → run optimizer → save checkpoints.

**Scenarios:**

- **Given** a `TransformerLM`, `TinyShakespeareDataset`, `TrainingConfig`, and `CheckpointManager`, **When** `train(config, model_config)` is called for 2 steps, **Then** loss after step 2 MUST be lower than initial loss (model learns).
- **Given** a training run, **When** `train(config, model_config)` is called for 1 step with a single batch, **Then** no runtime errors MUST occur.
- **Given** a checkpoint saved during training, **When** loaded and compared to a modified model, **Then** the loaded weights MUST match the saved checkpoint's state.
- **Given** a checkpoint saved after 2 training steps, **When** loaded, **Then** the restored model weights MUST produce the same loss as the original saved checkpoint.

### Requirement: No Mixed Precision

The training loop MUST NOT use mixed precision (bf16/fp16). All operations must be in fp32.

**Scenarios:**

- **Given** the training loop source code, **When** inspected, **Then** it MUST NOT call `torch.cuda.amp`, `autocast`, or `GradScaler`.
- **Given** the training loop source code, **When** inspected, **Then** forward pass MUST produce fp32 logits.

### Requirement: DataLoader Safety

The DataLoader MUST be configured with `num_workers=0` to avoid multiprocessing deadlocks on Windows/ROCm.

**Scenarios:**

- **Given** the training loop, **When** `DataLoader` is instantiated, **Then** `num_workers` MUST be 0.
- **Given** the dataset code, **When** inspected, **Then** no multiprocessing or shared memory patterns MUST be present.
