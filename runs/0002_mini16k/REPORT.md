# M0.1 Mini Training Experiment — New 16k Byte-Level BPE Tokenizer

**Run:** `E:\M0.1\runs\0002_mini16k`
**Base checkpoint:** `E:\M0.1\checkpoints\base_v16k\base_checkpoint.pt`
**Interpreter:** `E:\M0.1\venv_rocm\Scripts\python.exe` (Python 3.12, torch 2.12.0+rocm7.14.0, ROCm, AMD Radeon RX 9060 XT)
**Date:** 2026-07-20
**Status:** MINI test (~300 steps). `runs/0001` and its checkpoint were NOT touched.

---

## 1. Training Philosophy (how M0.1 trains)

Extracted from `src/training/config.py`, `src/training/train.py`, and the
`engine_v2/*` modules.

| Aspect | Setting (from code) |
|---|---|
| **Optimizer** | `AdamW`, parameter-group weight decay. `bias` and RMSNorm `gamma` → `weight_decay=0`; all other params → `weight_decay=0.1`. `betas=(0.9, 0.95)`, `eps=1e-8`, `lr=max_lr`. |
| **Learning rate** | `max_lr = 3e-4`, cosine decay with floor `min_lr_ratio=0.1` → `min_lr = 3e-5`. |
| **Schedule** | `LambdaLR`: **linear warmup** for `warmup_steps` (200) from 0 → `max_lr`, then cosine decay to the `min_lr` floor. |
| **Weight decay** | `0.1` (decay-group params only). |
| **AMP / mixed precision** | `AMPContext` wraps `torch.amp.autocast` + `GradScaler` (enabled on `cuda`). Forward/backward under autocast; loss scaled; gradients unscaled before clipping; `scaler.step` + `scaler.update` each step. |
| **EMA** | `EMA(model, decay=0.9999)`. Shadow weights used for validation (`apply_shadow`/`restore`) and saved in checkpoint. |
| **Gradient clipping** | `clip_grad_norm_(max_norm=1.0)` after unscaling, on each gradient-accumulation boundary. |
| **Batch / seq** | `batch_size=4`, `seq_len=1024` (config defaults; mini run kept batch 4). |
| **Loss** | Composable `LossPipeline`: `CrossEntropyLoss(vocab_size)` + `RouterAuxLoss(0.02)` + `RouterZLoss(0.001)`. Vocab size is read from `model.config.vocab_size`. |
| **Checkpoint format** | `AsyncCheckpointManagerV2` single canonical `checkpoint.pt` with **SHA256 integrity** + rolling `checkpoint.previous.pt` backup. State = `{step, global_tokens, model_state, optimizer_state, scheduler_state, ema_state, amp_scaler_state, rng_states, metrics, env, dataset_hash, tokenizer_hash}`. Async write + atomic rename. Engine saves a baseline at step 0, then every 1000 steps and at the end. |
| **Health / recovery** | NaN/Inf health check → rollback to last clean checkpoint, halve LR, continue. |
| **Tokenizer integration (exact point)** | The engine **does not tokenize text**. The dataset (`src/training/dataset.py → TinyShakespeareDataset`) loads the tokenizer via `src.tokenizer.bpe.Tokenizer` and encodes text at init. The engine's only tokenizer touchpoint is `save_checkpoint`, which computes `tokenizer_hash` from `data_dir/tokenizer.json` (fallback `tokenizer_final_8k.json`) and records it in the checkpoint for traceability. Loss pipeline vocab comes from `model.config.vocab_size`. |

**FSM / architecture:** `TrainingEngineV2` is a model-agnostic FSM core (states
INIT→LOAD→TRAIN→SAVE→VALIDATE→FINISHED) with an event bus, multichannel
loggers (console/JSONL/CSV), graceful SIGINT/SIGTERM handling, and a granular
profiler.

---

## 2. Tokenizer Integration — IMPORTANT FINDING

**The new tokenizer file is NOT in HuggingFace `tokenizers` format.** Despite the
task stating "load it with the `tokenizers` library (`Tokenizer.from_file`)",
`E:\M0.1\data\tokenizers\tokenizer.json` is the **project's own `src/tokenizer/bpe.py`
save format**: `{"vocab": {id_str: [byte,...] | special_name}, "merges": [[a,b],...]}`.
`tokenizers.Tokenizer.from_file` fails with a JSON parse error on it.

**Resolution (minimal change, matches the harness):** load it with the project's
own `src.tokenizer.bpe.Tokenizer` — the exact loader the training harness already
uses in `TinyShakespeareDataset`. Verified:
- vocab size = **16384** (incl. specials `<|endoftext|>=256`, `<|pad|>=257`),
  16126 merges.
- Roundtrip encode/decode is lossless on Spanish text.
- Max token id observed < 16384 → fits in **uint16**.

So the integration is: `bpe.Tokenizer.load(tokenizer.json)` → encode corpus →
write uint16 `.bin` shards → `BinTokenDataset` reads them → model `vocab_size`
overridden to 16384. A copy of `tokenizer.json` is placed in the run's `data/`
dir so the engine's checkpoint `tokenizer_hash` is correct.

---

## 3. Dataset Build

Script: `E:\M0.1\runs\0002_mini16k\build_dataset.py`

- **Sources:** `data/raw_text/spanish_pretrain.txt`, `data/raw_text/combinado.txt`,
  and `E:\LLM_DATASETS\FINAL/*.parquet` (`text` column; the parquet `tokens`
  column belongs to an older tokenizer and was NOT reused).
- **Encoding:** `bpe.Tokenizer.encode` per document, with `<|endoftext|>` (id 256)
  inserted between documents as a boundary.
- **Output:** 7 × `shard_*.bin` uint16 files under `runs/0002_mini16k/data/`,
  plus a copy of `tokenizer.json` and `build_info.txt`.
- **Subsampling:** the full corpus exceeded 6M tokens, so it was **subsampled at a
  6,000,000-token cap** (reached at doc 68,900 of 86,705). This is a MINI test;
  a full run should encode the entire corpus.

| Metric | Value |
|---|---|
| Total tokens (this mini build) | 6,000,155 |
| Documents | 86,705 (68,900 encoded before cap) |
| Shards | 7 × uint16 (1,000,000 tokens each) |
| Vocab | 16384 (uint16-compatible) |

---

## 4. Mini Run Configuration

- `M01Config(vocab_size=16384)` → `TransformerLM` (**99.7M** params).
- `batch_size=4`, `seq_len=1024`, `max_steps=300`, `warmup_steps=200`,
  `max_lr=3e-4`, `min_lr_ratio=0.1`, `weight_decay=0.1`, `max_norm=1.0`.
- Device `cuda` (AMD Radeon RX 9060 XT). AMP + EMA enabled (unchanged philosophy).
- Run dir pinned to `runs/0002_mini16k` via a `FixedExperimentManager` (no
  auto-numbered subdir created). Logs to `runs/0002_mini16k/metrics.jsonl`.

---

## 5. Results (~300 steps)

| Metric | Value |
|---|---|
| **Final loss** | **6.5770** (CrossEntropy 6.37 + RouterAux 0.20 + RouterZ 0.003) |
| Start loss (step 0) | ~10.06 (≈ ln(16384)=9.70 → correct random-init sanity) |
| **Throughput** | **~6,771 tok/s avg** (steady-state end ≈ 6,947 tok/s) |
| **ms/step** | **~605 ms** (incl. step-0 + final checkpoint writes; steady ≈ 590 ms) |
| **Peak GPU memory** | **7,130 MB (6.96 GB)** peak / ~1.9 GB allocated steady-state |
| Total tokens seen | 1,228,800 (= 300 × 4 × 1024) |
| Elapsed | 181.5 s (3.02 min) |

Loss descended smoothly from ~10.0 to ~6.58 — healthy convergence signal on a
fresh 16k-vocab model over 300 steps. The `tokenizer_hash` recorded in the run
checkpoint is the SHA256 of the canonical 16k `tokenizer.json`.

---

## 6. Throughput & Time Estimates

Assumption: same 99.7M model, same RX 9060 XT, measured **~6,800 tok/s** at
`batch_size=4`. The run used only ~1.9 GB allocated (7.1 GB peak) of 16 GB, so a
real run can raise `batch_size` substantially (better GPU utilization → higher
tok/s). Two scenarios below:

| Target | Conservative (batch=4, 6.8k tok/s) | Scaled (batch≈16–24, ~4–6× tok/s) |
|---|---|---|
| **runs/0001 equivalent (14.7M tokens)** | ~36 min | ~6–9 min |
| **1B tokens** | ~41 h | ~7–10 h |
| **10B tokens** (typical small-LLM pretrain) | ~17 days | ~3–4 days |

At `batch_size=4` we are memory-light; the dominant cost is under-utilized GPU.
For a real pretrain, increase batch size to fill ~14 GB before paying for longer
wall-clock. The 300-step mini confirms the pipeline, tokenizer, and engine are
correct end-to-end; it is not a convergence result.

---

## 7. Base Checkpoint Strategy (created)

**Created:** `E:\M0.1\checkpoints\base_v16k\base_checkpoint.pt`

Script: `E:\M0.1\runs\0002_mini16k\create_base_checkpoint.py`

**What it is:** a fresh, fully-resumable step-0 checkpoint with
`M01Config(vocab_size=16384)`, built with the **same `AsyncCheckpointManagerV2`
API and the same state-dict schema the engine uses**, so `TrainingEngineV2.resume()`
loads it directly. It records `tokenizer_hash` (SHA256 of the 16k tokenizer).

**Why a new base (not reusing runs/0001 or its step-0):**
1. **Vocab incompatibility** — the previous 512-vocab model cannot be lifted onto
   a 16384-vocab embedding/output head; a clean init is mandatory.
2. **Reproducibility** — a single canonical, tokenizer-pinned starting point makes
   every future run traceable to a known weight/init + tokenizer.
3. **Resume/fork** — future runs init from this base and branch independently;
   the base itself is never overwritten by training (engine writes to run-local
   `checkpoint.pt`, not here).

Verified: reload sanity passes; embedding weight shape is `16384 × d_model`.

---

## 8. Artifacts

```
E:\M0.1\runs\0002_mini16k\
  build_dataset.py          # tokenizer -> uint16 .bin shards
  run_mini16k.py            # training entry (BinTokenDataset + FixedExperimentManager)
  create_base_checkpoint.py # canonical base checkpoint
  data/                     # 7 shard_*.bin + tokenizer.json + build_info.txt
  metrics.jsonl / metrics.csv
  run_summary.json
  summary.json
  training_profile.json
  environment.txt
  checkpoints/
    checkpoint.pt           # final step-300 (run-local; 1.6 GB)
    checkpoint.previous.pt  # baseline step-0 (run-local)

E:\M0.1\checkpoints\base_v16k\
  base_checkpoint.pt        # CANONICAL base, vocab=16384, step 0
```

---

## 9. Blockers / Issues Found

1. **Tokenizer format mismatch (handled):** `tokenizer.json` is `bpe.py` format,
   not HuggingFace `tokenizers` format — `tokenizers.Tokenizer.from_file` cannot
   load it. Used the project's `src.tokenizer.bpe.Tokenizer` instead (correct and
   minimal). The task's "use the tokenizers library" instruction was based on a
   wrong format assumption.
2. **`src/training/train.py` is currently broken (external edit, out of scope):**
   a `--vocab-size` argument was added at line 90 but lines 89–90 are
   over-indented (8 spaces) and the `return parser.parse_args(argv)` line was
   dropped, causing an `IndentationError` on import. This happened *after* the
   mini run completed (the run imported `train.py` successfully earlier). I did
   **not** modify `train.py` (constraint: don't touch the existing harness). The
   base-checkpoint script re-implements the two small helper functions inline to
   avoid the broken import. **Recommend:** repair `train.py` lines 89–90
   (4-space indent, restore `return`) before any future run that imports it.
3. `torch.from_file` did not handle `uint16` on this setup (returned empty
   tensors); switched the shard reader to `numpy.fromfile(..., uint16)`.
