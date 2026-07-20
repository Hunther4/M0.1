# M0.1

Decoder-only transformer language model from scratch in PyTorch. Research-grade implementation featuring Multi-head Latent Attention (MLA), Mixture of Experts (MoE) with shared + routed experts, SwiGLU activations, Rotary Position Embeddings (RoPE), weight-tied embeddings, and an enterprise-grade training framework with automatic recovery, EMA, and AMP.

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![Research](https://img.shields.io/badge/purpose-research-BF4FE0?style=flat-square)
![Architecture](https://img.shields.io/badge/architecture-MLA%20%7C%20MoE%20%7C%20SwiGLU%20%7C%20RoPE-00BFFF?style=flat-square)
![Parameters](https://img.shields.io/badge/params-110M%E2%80%93180M-9cf?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-yellow?style=flat-square)

---

## Overview

M0.1 is a **from-scratch decoder-only transformer language model** built for research and experimentation with modern architecture techniques. Rather than wrapping existing libraries, every component -- from the attention mechanism to the training loop -- is implemented directly in PyTorch, providing full visibility and control for research purposes.

The project targets approximately **181M parameters** at full scale and currently operates at Stage 1 (~110M parameters) with 4 routed experts and 1 shared expert. It is trained on TinyShakespeare and custom Spanish text corpora, using a trained BPE tokenizer.

The architecture incorporates techniques from state-of-the-art models:
- **Multi-head Latent Attention (MLA)** from DeepSeek-V2/V3, reducing KV cache footprint by compressing key-value states into a low-rank latent space.
- **Mixture of Experts (MoE)** with DeepSeek-style shared + routed experts and fine-grained expert scaling.
- **SwiGLU** activation function for the feedforward networks.
- **Weight-tied embeddings**, sharing the weight matrix between input embedding and output projection.
- **Rotary Position Embeddings (RoPE)** for relative position encoding.
- **RMSNorm** for pre-normalization in every transformer block.

---

## Architecture

### Model Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `vocab_size` | 32,768 | Vocabulary size (BPE tokenizer) |
| `context_length` | 8,192 | Maximum sequence length |
| `d_model` | 640 | Embedding / hidden dimension |
| `n_heads` | 10 | Number of attention heads |
| `d_head` | 64 | Dimension per attention head |
| `d_ff` | 1,728 | Dense feedforward hidden dimension |
| `n_layers` | 12 | Total transformer layers (2 dense + 10 MoE) |
| `rope_theta` | 10,000.0 | RoPE base frequency |

The model consists of 12 transformer layers: the first 2 are dense feedforward layers, and the remaining 10 use Mixture of Experts. Each layer uses pre-norm (RMSNorm) with residual connections around both the attention and feedforward sublayers.

### Multi-head Latent Attention (MLA)

Standard multi-head attention (MHA) stores separate key and value projections for every head, resulting in a KV cache of size `2 * n_heads * d_head = 2 * d_model` per token. MLA compresses the key-value states into a low-rank latent space, dramatically reducing cache memory.

**How MLA works:**

1. The query is split into two parts: a "content" component (no RoPE) and a "position" component (with RoPE):
   - `W_q`: projects to `n_heads * d_head_no_rope` (content query)
   - `W_qr`: projects to `n_heads * d_head_rope` (position query with RoPE)

2. Key-value states are compressed through a down-projection:
   - `W_kv_down`: compresses from `d_model` to `mla_kv_c_dim` (128)
   - The latent KV is normalized via `RMSNorm` before up-projection

3. The compressed latent is up-projected to reconstruct:
   - `W_k_up`: up-projects to `n_heads * d_head_no_rope` (content key)
   - `W_v_up`: up-projects to `n_heads * d_head` (full value)

4. A separate projection handles the RoPE portion of the key:
   - `W_kr`: projects to `n_heads * d_head_rope` (position key with RoPE)

**KV cache reduction:**

| Mechanism | Cache per token | MLA saving |
|-----------|----------------|------------|
| Standard MHA | `2 * d_model = 1,280` floats | -- |
| MLA | `mla_kv_c_dim + n_heads * d_head_rope = 128 + 160 = 288` floats | **77% reduction** |

The content portion of K and V is reconstructed on the fly from the compressed latent during attention computation. Only the compressed latent (128 floats) plus the RoPE key projection (160 floats = 10 heads * 16 dim) needs to be cached per token.

**Configuration details:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `d_head` | 64 | Total head dimension |
| `d_head_rope` | 16 | RoPE dimension per head (clamped to even < d_head) |
| `d_head_no_rope` | 48 | Content dimension per head |
| `mla_kv_c_dim` | 128 | KV compression latent dimension |

**Fallback modes:** The attention module also supports standard MHA (`use_mla=False`) and Hybrid Attention (`use_hybrid_attention=True`) with Compressed Sparse Attention (CSA) and Heavily Compressed Attention (HCA), allowing comparative experimentation.

### Mixture of Experts (MoE)

The MoE implementation follows DeepSeek's architecture with two types of experts:

- **Shared experts**: Always active for every token. They capture general, commonly-needed knowledge.
- **Routed experts**: Dynamically selected per token via a learned gate. They specialize in different domains or patterns.

The output of an MoE layer is:

```
output = shared_experts(x) + sum_{i in top-k} g_i(x) * expert_i(x)
```

where `g_i(x)` is the gate probability for expert `i` (re-normalized among the top-k).

**Gate mechanism:**
- A linear projection maps the hidden state to `num_experts` logits.
- Softmax produces a probability distribution over experts.
- Top-k experts are selected by probability.
- Selected expert outputs are weighted by their (re-normalized) probabilities.

**Auxiliary losses:**
- **Load balancing loss**: `num_experts * sum(f * P)` where `f` is the fraction of tokens assigned to each expert and `P` is the average gate probability. This encourages uniform routing.
- **Router Z-loss**: `0.001 * mean(logsumexp(gate_logits)^2)` penalizes large gate logits, preventing routing collapse without compromising model quality.

**Expert scaling ladder (5 stages):**

The project defines a staged scaling plan that keeps the active parameter count per token roughly constant while increasing expert diversity. The routed expert hidden dimension (`d_ff_routed`) decreases as the number of experts increases, maintaining computational efficiency.

| Stage | Routed | Shared | Top-K | d_ff_routed | d_ff_shared | Total params | Active params/token |
|-------|--------|--------|-------|-------------|-------------|-------------|-------------------|
| 1 (current) | 4 | 1 | 1 | 640 | 1,024 | ~110M | ~540K |
| 2 | 8 | 2 | 2 | 448 | 768 | ~120M | ~580K |
| 3 | 16 | 2 | 2 | 256 | 640 | ~134M | ~520K |
| 4 | 32 | 4 | 4 | 128 | 432 | ~151M | ~500K |
| 5 | 40 | 5 | 4 | 112 | 384 | ~164.5M | ~500K |

The fixed base (embedding, attention, and 2 dense layers) accounts for approximately 41.4M parameters. The dense reference layer uses `d_ff=1728`, giving approximately 3.32M active parameters per dense layer. Each MoE stage targets roughly the same active parameter count per token as approximately 0.5x-0.6x of a dense layer.

### SwiGLU FeedForward

The feedforward network uses the SwiGLU activation function, which has been shown to outperform ReLU and GELU in transformer language models:

```
SwiGLU(x) = SiLU(gate_proj(x)) * up_proj(x)
output = down_proj(SwiGLU(x))
```

where `SiLU(x) = x * sigmoid(x)`. Each feedforward sublayer has three weight matrices:
- `gate_proj`: `d_model -> d_ff`
- `up_proj`: `d_model -> d_ff`
- `down_proj`: `d_ff -> d_model`

This is applied consistently in both dense layers, shared experts, and routed experts.

### Weight-Tied Embeddings

The embedding matrix is shared between the input embedding layer and the output projection head:

- Input: `Embedding(vocab_size, d_model)` maps token IDs to dense vectors.
- Output: A linear projection using `embedding.weight` (transposed) maps hidden states to vocabulary logits.

This saves approximately 21 million parameters (`vocab_size * d_model = 32,768 * 640 = 20.97M`) and provides a single gradient signal through the embedding matrix, which acts as a regularizer and often improves convergence.

### Rotary Position Embeddings (RoPE)

Rotary Position Embeddings encode positional information by rotating pairs of dimensions in the query and key representations:

```
rotated_even = x_even * cos(m*theta_i) - x_odd * sin(m*theta_i)
rotated_odd  = x_even * sin(m*theta_i) + x_odd * cos(m*theta_i)
```

where `theta_i = 1 / rope_theta^(2i/d_head)` and `m` is the position index. RoPE is applied to the position-specific portion of queries and keys (in MLA, this is the `d_head_rope = 16` dimension per head). The computation is done in FP32 for numerical stability, then cast back to the input dtype.

The implementation supports arbitrary position offsets for cached autoregressive generation, where precomputed sin/cos values are indexed by the current cache length.

### RMSNorm

Root Mean Square Normalization normalizes the input by its root mean square, then scales by a learnable parameter:

```
RMSNorm(x) = (x / sqrt(mean(x^2) + eps)) * gamma
```

The normalization is computed in FP32 to avoid overflow in FP16/BF16, then cast back to the original dtype before scaling. RMSNorm is applied as pre-norm before both the attention and feedforward sublayers in every transformer block.

### Router Z-Loss

The DeepSeek-style Router Z-Loss is an auxiliary loss that penalizes large gate logits to prevent routing collapse:

```
z_loss = 0.001 * mean(logsumexp(gate_logits, dim=-1)^2)
```

This loss:
- Promotes balanced logit magnitudes across experts.
- Prevents the router from becoming deterministic too early.
- Is applied additively to the total loss and does not interfere with the primary language modeling objective.

In the V2 training framework, the Z-loss is separated from the load balancing loss via the `RouterZLossTerm` in the composable `LossPipeline`.

---

## Training System

### TrainingEngineV2 (Enterprise Framework)

The primary training orchestration is built on a **Finite State Machine (FSM)** with clearly defined states:

```
INIT -> LOAD -> TRAIN -> VALIDATE -> SAVE -> EVALUATE -> EXPORT -> FINISHED
```

**Key components:**

- **StateMachine**: Defines valid transitions, preventing illegal state changes and providing a verifiable lifecycle for the training process.
- **EventBus**: Decoupled publish/subscribe communication. Plugins register for events (`STEP_START`, `BEFORE_BACKWARD`, `STEP_END`, etc.) without tight coupling to the engine.
- **LossPipeline**: Composable loss manager supporting multiple weighted loss terms. The default pipeline uses `CrossEntropyLossTerm` (language modeling), `RouterAuxLossTerm` (load balancing), and `RouterZLossTerm` (Z-loss regularization).
- **GranularProfiler**: Per-component timing for forward pass, backward pass, optimizer step, and data loading, exported as structured JSON.
- **AsyncCheckpointManagerV2**: Background checkpoint saving with environment metadata capture, RNG state preservation, and atomic save semantics.
- **HealthChecker**: Monitors gradients for NaN/Inf, triggers automatic recovery by rolling back to the last clean checkpoint and halving the learning rate.

The V2 engine supports:
- Automatic recovery on NaN/Inf loss with LR rollback.
- Graceful shutdown on SIGINT/SIGTERM with canonical checkpoint save.
- Loss breakdown tracking per step (logged to JSONL and CSV).
- Environment metadata capture (hardware, OS, Python, PyTorch version, command line, CPU cores, RAM).

### Simplified Callback System (V1 Engine)

The original training engine (`TrainingEngine`) exposes a minimal callback interface with only 3 hooks:

| Hook | When it fires | Purpose |
|------|---------------|---------|
| `on_step_end` | After each optimizer step | Logging, metric computation, early stopping checks |
| `on_validation` | After validation completes | Post-validation logging or adjustments |
| `on_save` | After checkpoint save | Post-save notifications or remote syncs |

Built-in callbacks:
- **LoggerCallback**: Console logging every `log_interval` steps with loss, LR, perplexity, gradient norm, throughput, VRAM, and MoE metrics.
- **MoEMonitorCallback**: Computes MoE routing metrics (entropy, Gini, dead experts) and detects router collapse. Stops training if collapse persists for 50+ consecutive steps.
- **EarlyStopCallback**: Stops training when loss becomes NaN or Inf.
- **CheckpointCallback**: Saves checkpoints at configurable intervals with RNG state capture.
- **JSONLLoggerCallback**: Writes step-level metrics to `runs/run_XXXX/metrics.jsonl`.

### Exponential Moving Average (EMA)

EMA maintains shadow weights using Polyak averaging:

```
shadow = decay * shadow + (1 - decay) * model_param
```

Key features:
- Shadow weights are updated after every optimizer step.
- **Validation uses EMA weights**: `apply_shadow()` swaps model weights to EMA values before validation, and `restore()` reverts them afterward. This produces lower validation loss and better perplexity in practice.
- EMA state is included in checkpoints, ensuring continuity across training resumptions.
- The update is skipped if shadow weights are currently in-place (applied to the model), preventing double-accumulation.

### Automatic Mixed Precision (AMP)

The `AMPContext` provides a unified precision context wrapper:

- Supports both FP16 and BF16 precision.
- Uses `torch.amp.autocast` and `torch.amp.GradScaler` for gradient scaling.
- Default to FP16 (`float16`), configurable to BF16 (`bfloat16`).
- Enabled automatically when CUDA is available.
- Exposes the current scaler scale for logging.

### Gradient Accumulation

Gradient accumulation allows effective batch sizes larger than what fits in GPU memory:

```
effective_batch = batch_size * gradient_accumulation_steps
```

The loss is divided by `gradient_accumulation_steps` before backpropagation, and the optimizer step (gradient clipping, scaler step, scheduler step, EMA update) only occurs on accumulation boundaries.

### Validation Pipeline

- Runs at configurable intervals (default: every 500 steps).
- Uses EMA weights when EMA is enabled, then restores original weights.
- Computes cross-entropy loss and perplexity on a held-out split of the dataset.
- Tracks the best validation loss seen during training.
- Limited to 30-50 validation steps to avoid excessive overhead.

### MoE Monitoring

The `MoEMonitorCallback` (V1) and `MetricRegistry` with forward hooks (V2) provide real-time monitoring of MoE behavior with 20+ metrics across 4 categories:

#### Distribution Metrics

| Metric | Description | Formula |
|--------|-------------|---------|
| `expert_usage_histogram` | Token count per routed expert | `bincount(topk_indices)` |
| `gini_coefficient` | Token distribution inequality (0=perfect, 1=monopoly) | `(2 * sum(i * x_i)) / (n * sum(x_i)) - (n+1)/n` |
| `imbalance_ratio` | Ratio of max to min expert usage | `max(hist) / max(min(hist>0), 1)` |
| `expert_utilization_pct` | Utilization percentage relative to capacity | `hist / capacity` |

#### Router Metrics

| Metric | Description | Formula |
|--------|-------------|---------|
| `entropy` | Mean per-token normalized entropy | `H(p) / log(n_experts)` |
| `confidence` | Mean max gate probability | `mean(max(softmax(logits)))` |
| `gate_logits_std` | Standard deviation of raw gate logits | `std(gate_logits)` |
| `gate_logits_mean` | Mean of raw gate logits | `mean(gate_logits)` |
| `top1_frequency` | Fraction of tokens where expert is top-1 choice | `bincount(top1) / N_tokens` |
| `top2_frequency` | Fraction of tokens where expert appears in top-2 | `bincount(top2_flat) / N_tokens` |

#### Health Metrics

| Metric | Description | Formula |
|--------|-------------|---------|
| `dead_expert_streak` | Consecutive steps with zero-token experts | `counter increment on (hist == 0).any()` |
| `expert_saturation` | How close each expert is to theoretical capacity | `hist / (tokens / experts * top_k)` |
| `aux_loss_ema` | Exponential moving average of auxiliary loss | `decay * prev_ema + (1-decay) * current` |

#### Quality Metrics

| Metric | Description | Formula |
|--------|-------------|---------|
| `expert_kl_divergence` | KL divergence of mean gate distribution from uniform | `sum(p * log(p / uniform))` |
| `shared_expert_usage` | Shared expert activity (always active) | Static: always `True` with count |

#### Detection Mechanisms

| Detection | Trigger | Action |
|-----------|---------|--------|
| Router collapse | All experts dead across all MoE layers | `should_stop = True` after 50 consecutive steps |
| NaN loss | Loss is NaN or Inf | `should_stop = True` immediately |
| V2 Health check | Gradients are NaN/Inf, or loss explosion | Rollback to last checkpoint, halve LR, flush cache |

### Experiment Tracking (RunManager / ExperimentManager)

Every training run creates an auto-structured experiment directory:

```
runs/
  run_0001/
    config.yaml          # Frozen copy of model + training configuration
    metrics.jsonl        # Step-by-step metrics (newline-delimited JSON)
    train.log            # Console stdout log
    checkpoint/          # Model checkpoints (checkpoint.pt, checkpoint.previous.pt)
    summary.json         # Training summary (final loss, total tokens, duration)
    plots/               # (future) Auto-generated metric plots
    environment.txt      # Hardware, OS, Python version metadata
    config.json          # Full configuration as JSON
    training_profile.json # Per-component timing breakdown (V2 profiler)
```

**Config hash**: Each run records an 8-character SHA256 hash of the configuration, enabling exact reproducibility comparisons between runs.

**JSONL logging**: Metrics are written as JSON lines, one per step (at log interval), containing loss, CE loss, aux loss, learning rate, gradient norm, throughput, and MoE metrics. This format is easily parsed with tools like `pandas.read_json(..., lines=True)`.

### LR Scheduler

Cosine decay with linear warmup:

```
warmup:     lr = step / warmup_steps * max_lr        (step < warmup_steps)
cosine:     lr = min_lr_ratio + (1-min_lr_ratio) * 0.5 * (1 + cos(pi * progress))
```

Default parameters: `max_lr=3e-4`, `min_lr_ratio=0.1`, `warmup_steps=200`.

### torch.compile Support

The training entry point supports `--compile` for `torch.compile` graph optimization, which can significantly accelerate training on compatible hardware by fusing operations and reducing Python overhead.

---

## Quick Start

### Prerequisites

- Python 3.10 or later
- PyTorch 2.0 or later (CUDA or ROCm)
- A CUDA-capable GPU with at least 8GB VRAM recommended (tested on AMD Radeon RX 9060 XT with ROCm)

### Installation

```bash
git clone https://github.com/Hunther4/M0.1.git
cd M0.1
pip install torch numpy pytest
```

### Prepare Data

Download TinyShakespeare:

```bash
python -m src.dataset.prep --download
```

Or ingest a custom Spanish text corpus:

```bash
python -m src.dataset.prep -i path/to/corpus.txt -o data/raw_text/corpus.txt
```

### Train

```bash
python -m src.training.train --batch-size 4 --max-steps 1000
```

For ROCm GPU (AMD):

```bash
.\venv_rocm\Scripts\python.exe -m src.training.train
```

### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--batch-size` | 4 | Batch size per step |
| `--seq-len` | 1024 | Sequence length |
| `--max-lr` | 3e-4 | Peak learning rate |
| `--max-steps` | 100,000 | Total training steps |
| `--warmup-steps` | 200 | LR warmup steps |
| `--grad-accum-steps` | 1 | Gradient accumulation steps |
| `--val-interval` | 500 | Steps between validation runs |
| `--log-interval` | 10 | Steps between console logs |
| `--save-interval` | 1,000 | Steps between checkpoint saves |
| `--compile` | False | Enable torch.compile |
| `--ema-decay` | 0.0 | EMA decay rate (0=disabled) |
| `--tag` | "" | Optional run tag for experiment tracking |
| `--resume` | False | Resume from latest checkpoint |

### Generate Text

```python
from src.model.lm import TransformerLM
from src.transformer.config import M01Config
from src.tokenizer.bpe import Tokenizer
from src.inference.generate import generate
import torch

config = M01Config()
model = TransformerLM(config)
model.load_state_dict(torch.load("path/to/checkpoint.pt", weights_only=True)["model_state_dict"])
model.eval().to("cuda")

tokenizer = Tokenizer()
tokenizer.load("data/tokenizer.json")

text = generate(model, tokenizer, "To be, or not to be", max_gen_len=200, temperature=0.8)
print(text)
```

---

## Project Structure

```
M0.1/
├── src/
│   ├── transformer/
│   │   ├── config.py           # M01Config dataclass (d_model, n_heads, num_experts, etc.)
│   │   ├── embeddings.py       # TokenEmbedding with weight-tied output head
│   │   ├── attention.py        # CausalSelfAttention (MLA, Hybrid, MHA modes)
│   │   ├── moe.py              # MoELayer (shared + routed experts, gate, aux loss)
│   │   ├── feedforward.py      # SwiGLU FeedForward (expert building block)
│   │   ├── rope.py             # Rotary Positional Embeddings
│   │   └── kv_cache.py         # Pre-allocated KV cache for autoregressive generation
│   ├── model/
│   │   ├── lm.py               # TransformerLM (embed -> blocks -> norm -> tied output)
│   │   ├── block.py            # TransformerBlock (pre-norm attention + FF/MoE)
│   │   └── rms_norm.py         # RMSNorm
│   ├── training/
│   │   ├── train.py            # CLI entry point (training)
│   │   ├── engine.py           # TrainingEngine (V1, callback-based)
│   │   ├── config.py           # TrainingConfig (batch_size, max_lr, warmup_steps, etc.)
│   │   ├── state.py            # TrainerState (single source of truth)
│   │   ├── callbacks.py        # Callback interface and implementations
│   │   ├── checkpoint.py       # CheckpointManager (atomic save/load)
│   │   ├── ema.py              # ModelEMA (Polyak averaging)
│   │   ├── amp.py              # AMPContext (mixed precision wrapper)
│   │   ├── moe_metrics.py      # 20+ MoE metrics (4 categories)
│   │   ├── metrics.py          # MetricRegistry (CE, perplexity, throughput, memory)
│   │   ├── run_manager.py      # RunManager (structured experiment directories)
│   │   ├── dataset.py          # TinyShakespeareDataset (sliding window)
│   │   ├── datasets.py         # Additional dataset utilities
│   │   ├── loop.py             # Training loop utilities
│   │   ├── eval.py             # Evaluation utilities
│   │   └── setup.py            # Training setup helpers
│   ├── engine_v2/
│   │   ├── engine.py           # TrainingEngineV2 (FSM-based, enterprise-grade)
│   │   ├── fsm.py              # StateMachine (state transitions with validation)
│   │   ├── bus.py              # EventBus (decoupled publish/subscribe)
│   │   ├── loss_pipeline.py    # LossPipeline (composable loss terms)
│   │   ├── experiment.py       # ExperimentManager (run directory structure)
│   │   ├── checkpoint_v2.py    # AsyncCheckpointManagerV2
│   │   ├── ema.py              # EMA (V2)
│   │   ├── amp.py              # AMPContext (V2)
│   │   ├── metrics.py          # MetricRegistry (V2)
│   │   ├── plugins.py          # Plugin system
│   │   ├── profiler.py         # GranularProfiler
│   │   ├── health.py           # HealthChecker
│   │   └── loggers.py          # ConsoleLogger, JSONLLogger, CSVLogger
│   ├── tokenizer/
│   │   ├── bpe.py              # BPE tokenizer (train, encode, decode)
│   │   └── __init__.py
│   ├── dataset/
│   │   └── prep.py             # Data preparation (download, ingest)
│   ├── inference/
│   │   ├── generate.py         # Autoregressive text generation
│   │   ├── sampling.py         # Sampling strategies (temperature, top-k, top-p)
│   │   ├── profiling.py        # Inference profiling
│   │   └── cli.py              # Inference CLI
│   ├── eval/
│   │   ├── evaluate.py         # Evaluation pipeline
│   │   ├── metrics.py          # Evaluation metrics
│   │   ├── qa.py               # Question answering evaluation
│   │   └── utils.py            # Evaluation utilities
│   └── tools/
│       └── counter.py          # Parameter counting and model analysis
├── tests/
│   ├── test_attention.py       # Attention module tests
│   ├── test_moe.py             # MoE layer tests
│   ├── test_block.py           # Transformer block tests
│   ├── test_lm.py              # Language model tests
│   ├── test_rope.py            # RoPE tests
│   ├── test_embeddings.py      # Embedding tests
│   ├── test_feedforward.py     # Feedforward tests
│   ├── test_rms_norm.py        # RMSNorm tests
│   ├── test_kv_cache.py        # KV cache tests
│   ├── test_config.py          # Config validation tests
│   ├── test_tokenizer.py       # Tokenizer tests
│   ├── test_training.py        # Training loop tests
│   ├── test_moe_metrics.py     # MoE metrics tests
│   ├── test_inference.py       # Inference tests
│   ├── test_gpu_sanity.py      # GPU sanity checks
│   ├── test_engine_v2_hardened.py  # V2 engine hardened tests
│   ├── test_eval_*.py          # Evaluation tests
│   ├── test_scripts_*.py       # Script integration tests
│   └── ... (34 test files total)
├── data/                       # Training data (TinyShakespeare, Spanish corpus)
├── runs/                       # Experiment runs (run_0001/, run_0002/, etc.)
├── docs/                       # Architecture and design documentation
├── scripts/                    # Training and evaluation scripts
├── artifacts/                  # Generated artifacts
├── moe_calc.py                 # MoE scaling ladder calculator
├── requirements.txt            # Python dependencies
├── pytest.ini                  # Pytest configuration
└── README.md                   # This file
```

---

## Testing

The project has 34 test files covering all components:

```bash
python -m pytest tests/ -q
```

Run a specific test file:

```bash
python -m pytest tests/test_moe.py -v
```

Run GPU-specific tests (requires CUDA/ROCm):

```bash
python -m pytest tests/ -m gpu
```

The test suite covers:
- **Model components**: attention (MLA, MHA, Hybrid), MoE (routing, shared experts, aux loss), feedforward (SwiGLU), embeddings (weight tying), RMSNorm, RoPE, KV cache.
- **Model assembly**: transformer block, language model, parameter initialization.
- **Training**: engine V1 and V2, callbacks, state management, checkpoint save/load, EMA, AMP, gradient flow.
- **MoE metrics**: all 20+ metric functions, router collapse detection.
- **Inference**: autoregressive generation, KV cache prefill, sampling strategies.
- **Tokenizer**: BPE training, encoding, decoding.
- **Cross-cutting**: model initialization, script integration, GPU sanity.
- **Evaluation**: metric computation, QA evaluation.

---

## Roadmap

### MoE Scaling Stages

The expert scaling ladder is the primary development axis:

- **Stage 1 (current):** 4+1 routed/shared experts, top-1 routing, d_ff_routed=640. Validation of basic MoE training dynamics, Gini coefficient measurement, router entropy monitoring. Approximately 110M parameters.
- **Stage 2:** 8+2 experts, top-2 routing, d_ff_routed=448. Balance testing with multiple active experts per token. Measurement of expert specialization patterns.
- **Stage 3:** 16+2 experts, top-2 routing, d_ff_routed=256. Entropy and coefficient of variation measurement at moderate scale.
- **Stage 4:** 32+4 experts, top-4 routing, d_ff_routed=128. DeepSeek-lite configuration with fine-grained experts.
- **Stage 5:** 40+5 experts, top-4 routing, d_ff_routed=112, d_ff_shared=384. Full-scale configuration targeting approximately 164.5M total parameters (~181M with embedding tying accounted).

### Training Infrastructure

- Validation on larger datasets (The Pile, C4 subsets, or Spanish corpora).
- Distributed training (DDP / FSDP) for larger model scales.
- Extended inference optimization (speculative decoding, KV cache quantization).
- WandB or MLflow integration for remote experiment tracking.
- Hyperparameter optimization sweeps for LR, warmup, weight decay, and MoE-specific parameters.
- Automatic MoE expert analysis: per-expert token clustering to identify learned specializations.

### Research Directions

- Expert merging and pruning experiments to reduce inference cost.
- Comparative analysis of attention variants (MLA vs MHA vs Hybrid) at identical parameter counts.
- Capacity factor implementation for bounded expert load in MoE layers.
- Joint optimization of Z-loss weight and load balancing loss coefficients.
- Analysis of the relationship between router entropy, Gini coefficient, and model quality (validation loss).

---

## License

MIT

---

## References

- DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model (2024)
- DeepSeek-V3 Technical Report (2024)
- Llama 2: Open Foundation and Fine-Tuned Chat Models (2023)
- SwiGLU: A Gated Linear Unit for Feedforward Networks (2020)
- RoFormer: Enhanced Transformer with Rotary Position Embedding (2021)
- RMSNorm: Root Mean Square Layer Normalization (2019)
- TinyShakespeare: Andrej Karpathy's char-rnn dataset
