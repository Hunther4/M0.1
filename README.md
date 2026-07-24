# M0.1

Decoder-only transformer language model built from scratch in PyTorch. ~99.7M parameters, trained on Spanish text corpora. Research-grade implementation focused on modern architecture experimentation — Multi-head Latent Attention (MLA), Mixture of Experts (MoE), and an enterprise-grade training framework.

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![Research](https://img.shields.io/badge/purpose-research-BF4FE0?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-yellow?style=flat-square)

---

## Overview

M0.1 is a **from-scratch decoder-only transformer language model** — every component from the attention mechanism to the training loop is implemented directly in PyTorch rather than wrapping existing libraries. The goal is full visibility and control for research and experimentation with modern LLM architecture techniques.

The model currently operates with 12 transformer layers (2 dense feedforward + 10 MoE), 4 routed experts with 1 shared expert and top-2 routing, Multi-head Latent Attention (MLA), and a 16K BPE tokenizer. Training is done on custom Spanish corpora using the V2 training engine.

**What this project is:**
- A research vehicle for experimenting with MLA, MoE routing dynamics, and training stability
- A from-scratch implementation you can read, modify, and extend without library abstractions
- A training framework with automatic recovery, experiment tracking, and composable loss pipelines

**What this project is not:**
- A production-ready or deployment-optimized model
- A state-of-the-art benchmark competitor (the model is ~100M params and trained on limited data)
- A wrapper around existing transformer libraries (HuggingFace, etc.)

---

## Architecture

### Model Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `vocab_size` | 16,384 | Vocabulary size (BPE tokenizer) |
| `context_length` | 8,192 | Maximum sequence length |
| `d_model` | 640 | Embedding / hidden dimension |
| `n_heads` | 10 | Number of attention heads |
| `d_head` | 64 | Dimension per attention head |
| `d_ff` | 1,728 | Dense feedforward hidden dimension |
| `d_ff_shared` | 448 | Shared expert FF hidden dimension |
| `d_ff_routed` | 784 | Routed expert FF hidden dimension |
| `n_layers` | 12 | Total transformer layers |
| `n_dense_layers` | 2 | Layers with dense FF (first 2) |
| `num_experts` | 4 | Number of routed experts |
| `num_shared_experts` | 1 | Number of always-active shared experts |
| `moe_top_k` | 2 | Top-k experts selected per token |
| `rope_theta` | 10,000.0 | RoPE base frequency |

The model has approximately **99.7M total parameters**. The 12 layers break down as: the first 2 use a standard dense feedforward (SwiGLU), and the remaining 10 use Mixture of Experts (1 shared + 4 routed, top-2). Every layer uses pre-normalization (RMSNorm) with residual connections around both the attention and feedforward sublayers.

For the external-model research notes that informed the current optimization direction, see [docs/vendor_landscape_and_optimization_plan.md](./docs/vendor_landscape_and_optimization_plan.md).
For the phased execution plan and improvement targets, see [docs/execution_plan_and_targets.md](./docs/execution_plan_and_targets.md).

### Prompt Prefix Caching

Inference can reuse the prefilled KV state of repeated or shared prompt prefixes
through `PromptPrefixCache`. The cache supports MLA, hybrid attention, and
standard MHA, uses bounded LRU eviction, and invalidates entries after model or
configuration changes.

```python
from src.inference.generate import generate
from src.inference.prompt_cache import PromptPrefixCache

prompt_cache = PromptPrefixCache(max_entries=8, max_bytes=256 * 1024 * 1024)
output = generate(model, tokenizer, prompt, prompt_cache=prompt_cache)
print(prompt_cache.stats)
```

Measure the effect on a real checkpoint instead of assuming a fixed speedup:

```bash
python -m scripts.evaluation.benchmark_inference \
  --checkpoint checkpoints/checkpoint.pt \
  --tokenizer data/tokenizers/tokenizer.json \
  --prompt "Shared system prompt: first task" \
  --prompt "Shared system prompt: second task" \
  --mode compare \
  --output artifacts/evals/prompt_cache_benchmark.json
```

### Multi-head Latent Attention (MLA)

MLA is the attention mechanism introduced in DeepSeek-V2 that reduces the KV cache footprint by compressing key-value states into a low-rank latent space. Instead of caching separate K and V projections for every head (`2 * n_heads * d_head = 2 * d_model` floats per token), MLA caches a compressed latent plus a small RoPE projection.

**How it works:**

1. **Query split**: The query is projected into two parts — a content component (no position encoding, 48 dims per head) and a position component (with RoPE, 16 dims per head):
   - `W_q`: `d_model → n_heads * d_head_no_rope` (content query)
   - `W_qr`: `d_model → n_heads * d_head_rope` (position query with RoPE)

2. **KV compression**: Key-value states are compressed through a down-projection, normalized, and stored as the latent representation:
   - `W_kv_down`: `d_model → mla_kv_c_dim` (128 dims) — this is what gets cached
   - `norm_kv`: RMSNorm applied to the compressed latent

3. **On-the-fly reconstruction**: During attention computation, the latent is up-projected to reconstruct content key and full value:
   - `W_k_up`: `mla_kv_c_dim → n_heads * d_head_no_rope` (content key)
   - `W_v_up`: `mla_kv_c_dim → n_heads * d_head` (full value)

4. **Position encoding**: A separate projection handles the RoPE portion of the key:
   - `W_kr`: `d_model → n_heads * d_head_rope` (position key with RoPE)

**KV cache comparison:**

| Mechanism | Cache per token | Savings |
|-----------|----------------|---------|
| Standard MHA | `2 × d_model = 1,280` floats | — |
| MLA | `mla_kv_c_dim + n_heads × d_head_rope = 128 + 160 = 288` floats | ~77% less |

The cached MLA forward path (`_forward_mla_cached`) computes attention scores without materializing historical K/V tensors — it runs content scores through a three-term einsum (`q_c @ k_up @ latent`) and rope scores directly, then applies causal masking and softmax in FP32 to prevent numerical instability.

The attention module also supports standard MHA (`use_mla=False`) and Hybrid Attention (`use_hybrid_attention=True`) with Compressed Sparse Attention (CSA) and Heavily Compressed Attention (HCA) for comparative experimentation, though MLA is the primary mode.

### Mixture of Experts (MoE)

The MoE layer follows DeepSeek's architecture with two expert types:

- **Shared experts** (1): Always active for every token. Capture general, commonly-needed knowledge with a smaller FF dimension (d_ff_shared=448).
- **Routed experts** (4): Dynamically selected per token via a learned gate. Specialize in different patterns. Each has a larger FF dimension (d_ff_routed=784).

The output of an MoE layer:

```
output = shared_experts(x) + Σ(g_i(x) × expert_i(x))  for i in top-k
```

where `g_i(x)` is the gate probability re-normalized among the selected top-k experts.

**Gate mechanism:**
- A linear projection maps the hidden state to `num_experts` logits
- During training, Gaussian noise (std = 0.1 × softplus(logits)) is added to the logits for exploration, then cleaned probabilities are used for loss computation
- Softmax produces a probability distribution over experts
- Top-2 experts are selected by probability, and their weights are re-normalized

**Capacity filtering:** Each expert has a capacity limit (`capacity = floor(tokens × k / num_experts × capacity_factor)`) to prevent collapse. Tokens above capacity are dropped (their assignments rejected) and do not contribute to the routed output for that rank. The capacity factor warms up linearly from `capacity_factor_warmup_start` to the target during early training steps.

**Auxiliary losses:**
- **Load balancing loss**: `num_experts × Σ(f_i × P_i)` where `f_i` is the fraction of tokens assigned to expert `i` (after capacity), and `P_i` is the average clean gate probability for expert `i`. This encourages uniform routing across experts.
- **Z-loss**: `0.001 × mean(logsumexp(gate_logits)²)` — penalizes large gate logit magnitudes, preventing routing collapse early in training.

Both losses are computed separately and combined in the LossPipeline, keeping the gradient paths independent.

### SwiGLU FeedForward

The feedforward network uses SwiGLU, which empirically outperforms ReLU and GELU in transformer LMs:

```
SwiGLU(x) = SiLU(gate_proj(x)) × up_proj(x)
output = down_proj(SwiGLU(x))
```

where `SiLU(x) = x × sigmoid(x)`. Each FF sublayer has three weight matrices (`gate_proj`, `up_proj`, `down_proj`), all projecting between `d_model` and the FF hidden dimension. This is applied consistently in dense layers, shared experts, and routed experts, each with their own configured FF dimension.

### Weight-Tied Embeddings

The embedding matrix is shared between the input embedding layer and the output projection head (lm_head). The output projection uses the same weight matrix transposed instead of a separate learned projection. This saves ~10.5M parameters (`vocab_size × d_model = 16,384 × 640 = 10,485,760`) and provides a single gradient signal through the embedding matrix, which acts as a regularizer.

### Rotary Position Embeddings (RoPE)

RoPE encodes positional information by rotating pairs of dimensions in queries and keys:

```
rotated_even = x_even × cos(m × θ_i) — x_odd × sin(m × θ_i)
rotated_odd  = x_even × sin(m × θ_i) + x_odd × cos(m × θ_i)
```

where `θ_i = 1 / rope_theta^(2i/d_head)` and `m` is the position index. In MLA mode, RoPE is applied only to the position-specific portion (16 dims per head). The RoPE computation is done in FP32 for numerical stability and cast back to the original dtype. Precomputed sin/cos tables support arbitrary position offsets for cached autoregressive generation.

### Parameter Breakdown

| Component | Parameters | % of Total |
|-----------|-----------|-----------|
| Token embeddings (tied) | 10,485,760 | 10.5% |
| Position embeddings (RoPE, fixed) | 0 | 0% |
| Attention (MLA, 12 layers) | 12,887,040 | 12.9% |
| Dense FF layers (2 layers) | 6,643,712 | 6.7% |
| MoE shared expert (10 layers) | 2,867,200 | 2.9% |
| MoE routed experts (10 layers × 4 experts) | 62,720,000 | 62.9% |
| RMSNorm (params) | 15,360 | <0.1% |
| Output norm + lm_head (tied) | 640 + 0 | <0.1% |
| **Total** | **~99.7M** | **100%** |

The tied embedding matrix accounts for 10.5M but is counted once (shared between input and output). The routed experts dominate at ~63% of total parameters, which is typical for MoE architectures — each token only activates 2 of 4 experts (plus the shared expert), so the effective inference compute is much lower than the total parameter count.

### RMSNorm

Root Mean Square Normalization is used as pre-norm in every transformer block:

```
RMSNorm(x) = (x / sqrt(mean(x²) + ε)) × γ
```

The normalization is computed in FP32 to avoid underflow/overflow in FP16/BF16. Both the normalized activation and the learnable scale parameter `γ` are cast to the activation dtype for the final multiply, preventing FP32 parameters from silently promoting mixed-precision activations.

---

## Training System

### TrainingEngineV2

The primary training orchestration (`src.training.train`) is built on a Finite State Machine with validated state transitions. The entry point uses a LION (Last-In, One-shot) resume pattern — it only needs the latest checkpoint to reconstruct the full training state, including optimizer, scheduler, RNG, and step count.

**Default training hyperparameters:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `batch_size` | 4 | Fits ~8GB GPU VRAM at seq_len=1024 |
| `seq_len` | 1024 | Balanced throughput vs context coverage |
| `max_lr` | 3e-4 | Standard for AdamW with this model scale |
| `min_lr_ratio` | 0.1 | Cosine decay floor (LR doesn't go to zero) |
| `warmup_steps` | 200 | Prevents early training instability |
| `weight_decay` | 0.1 | AdamW default for transformer LMs |
| `max_norm` | 1.0 | Gradient clipping threshold |

**LR Scheduler:**

Cosine decay with linear warmup:

```
warmup:     lr = step / warmup_steps × max_lr                (step < warmup_steps)
cosine:     lr = min_lr + 0.5 × (max_lr — min_lr) × (1 + cos(π × progress))
```

where `progress = (step — warmup_steps) / (max_steps — warmup_steps)`. The scheduler is applied through PyTorch's `LambdaLR`.

**FSM states and transitions:**

```
INIT → LOAD → TRAIN ⇄ VALIDATE → SAVE → EVALUATE → EXPORT → FINISHED
                                              ↓
                                        RECOVERING → LOAD
                                              ↓
                                           ERROR
```

**FSM validation**: The StateMachine defines legal transitions explicitly — attempting an illegal transition raises a `StateError`. This provides a verifiable lifecycle for the training process and prevents silent corruption from out-of-order operations.

**Key components:**

- **Finite StateMachine**: Validates every state transition. Provides `current_state`, `previous_state`, and transition history for debugging.
- **EventBus**: Decoupled publish/subscribe communication. Plugins register for lifecycle events (`STEP_START`, `BEFORE_BACKWARD`, `AFTER_OPTIMIZER`, `STEP_END`, `VALIDATION_START`, `VALIDATION_END`, `CHECKPOINT_SAVED`, etc.) without tight coupling to the engine.
- **LossPipeline**: Composable loss manager that accepts a list of weighted loss terms. The default pipeline uses:
  - `CrossEntropyLossTerm`: standard language modeling objective (weight 1.0)
  - `RouterAuxLossTerm`: load balancing loss (weight 0.01)
  - `RouterZLossTerm`: Z-loss regularization (weight 0.001)
  Each term receives the model output, targets, and model reference, and returns a named loss tensor. The pipeline sums weighted terms and stores each component for logging.
- **GranularProfiler**: Per-component timing (forward, backward, optimizer, data loading) exported as structured JSON to `training_profile.json`. Non-sampling — records every step.
- **AsyncCheckpointManagerV2**: Background checkpoint saving with SHA256 verification. On save, tensors are detached and moved to CPU synchronously before the background thread starts, preventing CUDA D2H races. Includes backup/rollback: `checkpoint.pt` + `checkpoint.previous.pt` with matching `.sha256` sidecars.
- **HealthChecker**: Monitors gradients for NaN/Inf and loss health. On NaN loss detection, triggers automatic recovery: rolls back to the last clean checkpoint and halves the learning rate.

### EMA (Exponential Moving Average)

Maintains shadow weights using Polyak averaging updated in-place with `lerp_()`:

```
shadow ← decay × shadow + (1 — decay) × model_param
```

During validation, `apply_shadow()` swaps model weights to EMA values. After validation, `restore()` reverts them. EMA state is included in checkpoints for continuity across resumptions.

### AMP (Automatic Mixed Precision)

The `AMPContext` wraps FP16 (default) or BF16 precision with `torch.amp.autocast` and `GradScaler`. Enabled automatically when CUDA is available. The scaler scale is exposed for logging.

### Gradient Accumulation

Effective batch size = `batch_size × gradient_accumulation_steps`. The loss is divided by `gradient_accumulation_steps` before backpropagation, and the optimizer step (gradient clipping, scaler update, scheduler step, EMA update) only executes on accumulation boundaries.

### Training Recovery

The engine handles two recovery paths:

**1. NaN/Inf loss recovery:** When `HealthChecker.check_loss()` detects NaN or Inf after the forward pass, the optimizer step is skipped entirely. The engine rolls back model weights to the last clean checkpoint and halves the learning rate via the scheduler. Gradient history is discarded to prevent corrupt gradients from contaminating future steps.

**2. Graceful shutdown:** On SIGINT/SIGTERM, the engine intercepts the signal and saves a canonical checkpoint before exiting. This ensures training can be resumed without losing progress even when killed mid-step.

### Parameter Initialization

All linear layers use `nn.Linear` default init (uniform Kaiming for weights, zeros for biases where applicable). Embedding layers use the default uniform init. RMSNorm scale parameters (`gamma`) are initialized to ones. The model does not use special initialization schemes (like DeepSeek's or Llama's scaled init) — the current scheme relies on warmup and gradient clipping to stabilize early training.

### Optimizer

AdamW with `betas=(0.9, 0.95)`, `eps=1e-8`. Parameters are split into two groups: those with `weight_decay` applied (all weight matrices except biases and RMSNorm gammas) and those without (biases, norm scales). This follows the standard decoupled weight decay pattern used in most modern LM training code.

### MoE Runtime Monitoring

TrainingEngineV2 exposes 20+ MoE metrics via the `MetricRegistry` with forward hooks on the MoE layer:

| Category | Metrics |
|----------|---------|
| Distribution | `expert_usage_histogram`, `gini_coefficient`, `imbalance_ratio`, `expert_utilization_pct` |
| Router | `entropy`, `confidence`, `gate_logits_std`, `gate_logits_mean`, `top1_frequency`, `top2_frequency` |
| Health | `dead_expert_streak`, `expert_saturation`, `aux_loss_ema` |
| Quality | `expert_kl_divergence`, `shared_expert_usage` |

Collapse detection stops training if all experts across all MoE layers are dead for 50+ consecutive steps.

### Experiment Tracking

Every run creates a structured directory under `runs/`:

```
runs/run_XXXX/
  ├── config.yaml          # Frozen model + training config
  ├── environment.txt      # Hardware, OS, Python version
  ├── metrics.jsonl        # Step-by-step metrics (JSONL)
  ├── training_profile.json # Per-component timing
  ├── summary.json         # Final training summary
  └── checkpoints/         # checkpoint.pt, checkpoint.previous.pt
```

Config hash (8-char SHA256) enables exact reproducibility comparisons between runs. Metrics are logged at `log_interval` steps to both console and JSONL.

---

## Data

The model trains on pre-tokenized binary shard corpora. Each corpus is a directory of `shard_XXXXXX.bin` files (uint16 big-endian, 1M tokens per shard).

| Corpus | Tokens | Shards | Sources |
|--------|--------|--------|---------|
| corpus1 | ~10M | 10 | Wikipedia ES, Wikisource ES, curated tech texts |
| corpus2 | ~16.6M | 17 | Wikipedia ES, Project Gutenberg ES (zero overlap with corpus1) |

The `BinaryCorpusDataset` (`src/training/dataset.py`) reads all shards from a corpus directory, concatenates token IDs, and provides sliding-window `(input, target)` pairs for autoregressive training. The `build_training_dataset()` function in `train.py` resolves the corpus path, defaulting to corpus2 when no `--corpus-dir` is specified.

To build a new corpus: use `src/data/prep.py` to tokenize raw text and produce binary shards.

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/Hunther4/M0.1.git
cd M0.1
pip install torch numpy pytest

# Train with default corpus (corpus2)
python -m src.training.train --batch-size 4 --max-steps 1000

# Train with a specific corpus
python -m src.training.train --corpus-dir data/corpus/corpus1_es_wiki_wikisource_tech_10M

# Resume from checkpoint
python -m src.training.train --resume

# ROCm (AMD GPU)
.\venv_rocm\Scripts\python.exe -m src.training.train
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-dir` | `data` | Root data directory |
| `--corpus-dir` | — | Explicit path to binary shard corpus (overrides default) |
| `--batch-size` | 4 | Sequences per optimizer step |
| `--seq-len` | 1024 | Token sequence length |
| `--max-lr` | 3e-4 | Peak learning rate |
| `--min-lr-ratio` | 0.1 | Minimum LR as fraction of max_lr |
| `--warmup-steps` | 200 | Linear LR warmup steps |
| `--max-steps` | 100,000 | Total training steps |
| `--weight-decay` | 0.1 | AdamW weight decay |
| `--grad-accum-steps` | 1 | Gradient accumulation steps |
| `--max-norm` | 1.0 | Gradient clipping max norm |
| `--log-interval` | 10 | Steps between console logging |
| `--save-interval` | 1000 | Steps between checkpoint saves |
| `--val-interval` | 500 | Steps between validation runs |
| `--resume` | — | Resume from canonical checkpoint (no arg) or specific path |
| `--vocab-size` | 16384 | Vocabulary size (must match tokenizer) |
| `--run-name` | — | Pin run directory name instead of auto-numbering |

---

## Project Structure

```
M0.1/
├── src/
│   ├── transformer/         # Core model components
│   │   ├── config.py        # M01Config — all model hyperparameters
│   │   ├── embeddings.py    # TokenEmbedding with weight tying
│   │   ├── attention.py     # CausalSelfAttention (MLA, Hybrid, MHA)
│   │   ├── attention_backend.py  # SDPA dispatch with fallback
│   │   ├── moe.py           # MoELayer (shared + routed, gate, aux loss, Z-loss)
│   │   ├── feedforward.py   # SwiGLU FeedForward
│   │   ├── rope.py          # Rotary Position Embeddings (FP32-safe)
│   │   └── kv_cache.py      # Pre-allocated KV caches (MLA, Hybrid, standard)
│   ├── model/
│   │   ├── lm.py            # TransformerLM — full model assembly
│   │   ├── block.py         # TransformerBlock (pre-norm attn + FF/MoE)
│   │   └── rms_norm.py      # RMSNorm (FP32-safe)
│   ├── training/
│   │   ├── train.py         # CLI entry point (V2 engine)
│   │   ├── config.py        # TrainingConfig dataclass
│   │   ├── dataset.py       # TinyShakespeareDataset, BinaryCorpusDataset
│   │   ├── checkpoint.py    # CheckpointManager (V1 compat)
│   │   ├── moe_metrics.py   # MoE runtime metrics (20+ functions)
│   │   └── loop.py, eval.py, setup.py, datasets.py  # V1 legacy (kept for compat)
│   ├── engine_v2/
│   │   ├── __init__.py      # Public API exports
│   │   ├── engine.py        # TrainingEngineV2 — FSM-based orchestration
│   │   ├── fsm.py           # StateMachine with validated transitions
│   │   ├── bus.py           # EventBus — decoupled pub/sub
│   │   ├── loss_pipeline.py # Composable loss terms (CE, aux, Z-loss)
│   │   ├── experiment.py    # ExperimentManager — run directory structure
│   │   ├── checkpoint_v2.py # AsyncCheckpointManagerV2
│   │   ├── ema.py           # EMA shadow weights
│   │   ├── amp.py           # AMPContext (FP16/BF16)
│   │   ├── metrics.py       # MetricRegistry with forward hooks
│   │   ├── plugins.py       # Plugin system
│   │   ├── profiler.py      # Per-component timing
│   │   ├── health.py        # HealthChecker — NaN/Inf recovery
│   │   └── loggers.py       # ConsoleLogger, JSONLLogger, CSVLogger
│   ├── tokenizer/
│   │   └── bpe.py           # BPE tokenizer (train, encode, decode)
│   ├── data/
│   │   └── prep.py          # Corpus builder — raw text → binary shards
│   ├── inference/
│   │   ├── generate.py      # Autoregressive generation
│   │   ├── sampling.py      # Temperature, top-k, top-p sampling
│   │   ├── profiling.py     # Inference profiling utilities
│   │   └── cli.py           # Inference REPL
│   ├── eval/
│   │   ├── evaluate.py      # Evaluation pipeline
│   │   ├── metrics.py       # Perplexity, accuracy, F1
│   │   ├── qa.py            # QA-style evaluation
│   │   └── utils.py         # Evaluation helpers
│   └── tools/
│       └── counter.py       # Parameter counting
├── tests/                   # ~79 pytest test files
├── scripts/
│   ├── training/train.py    # Training entry point
│   ├── evaluation/compare.py, generate_report.py
│   └── model_manipulation/ expand_model.py, merge_checkpoints.py, README.md
├── data/
│   ├── corpus/              # Binary shard corpora
│   └── tokenizers/          # BPE tokenizer files
├── docs/                    # Architecture documentation
├── .gitignore
└── README.md
```

---

## Testing

```bash
# Run full suite
python -m pytest tests/ -q

# Run specific module tests
python -m pytest tests/test_moe.py -v
python -m pytest tests/test_attention.py -v
python -m pytest tests/test_engine_v2_hardened.py -v
```

The test suite (~79 tests) covers:
- **Attention**: MLA forward/cached/FP32-safe paths, MHA, Hybrid, KV cache integration
- **MoE**: Routing, capacity filtering, shared experts, aux loss, Z-loss, entropy, Gini
- **Transformer blocks**: Block assembly, forward/backward, residual shapes
- **Model**: Full LM forward, parameter initialization, loss computation
- **Training engine V2**: FSM transitions, EventBus, LossPipeline, EMA, AMP, checkpoint save/load, gradient monitoring
- **Tokenizer**: BPE training, encode, decode, vocabulary management
- **Inference**: Generation loop, KV cache prefill, sampling strategies
- **Evaluation**: Metric computation, QA pipeline
- **Cross-cutting**: Config validation, checkpoint integrity, model manipulation

---

## Inference

```python
from src.model.lm import TransformerLM
from src.transformer.config import M01Config
from src.tokenizer.bpe import Tokenizer
from src.inference.generate import generate

config = M01Config(vocab_size=16384)
model = TransformerLM(config)
model.load_state_dict(torch.load("checkpoint.pt", weights_only=True)["model_state_dict"])
model.eval().to("cuda")

tokenizer = Tokenizer()
tokenizer.load("data/tokenizers/tokenizer.json")

text = generate(
    model, tokenizer,
    prompt="To be, or not to be",
    max_gen_len=200,
    temperature=0.8,
    top_k=40,
    top_p=0.9
)
print(text)
```

The inference pipeline supports:
- Autoregressive generation with KV cache (MLA, Hybrid, or standard)
- Configurable sampling: temperature, top-k filtering, top-p (nucleus) sampling
- Context validation — warns if prompt exceeds model's context length
- Coherence evaluation mode (checks repetition, topic drift, early ending)

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
