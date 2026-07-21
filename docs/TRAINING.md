# M0.1 Training Guide

**Status:** Documentation of the *validated* training pipeline (engine_v2). Corpús is available but **training is deferred** — no runs are executed today (see [Training Status](#training-status-deferred)).

## Quick Path (smoke test)

```bash
# From repo root, ROCm venv:
.\venv_rocm\Scripts\python.exe -m src.training.train `
  --run-name run_test `
  --batch-size 16 `
  --vocab-size 16384 `
  --warmup-steps 40 `
  --max-steps 250 `
  --max-lr 1.2e-3

# Layer-stacking: continue ("stack") knowledge on a base checkpoint:
.\venv_rocm\Scripts\python.exe -m src.training.train `
  --run-name run_test2 `
  --resume runs/run_test/checkpoints/checkpoint.pt `
  --vocab-size 16384 `
  --max-steps 300
```

Verification: the run prints a `RUN COMPLETION REPORT` with `GPU VRAM` (~11 GB at batch 16) and a `Final Loss`.

## The 16k Tokenizer Reality

> ⚠️ **There is exactly ONE tokenizer: `data/tokenizers/tokenizer.json`.**
> - Vocab size: **16384**
> - SHA-256 prefix: **`6bc3a6…`**
> - **There is NO 32k tokenizer.** Any config assuming vocab 32768 is wrong for this tokenizer.

Training (`train.py`) and evaluation (`evaluate.py`) both load `data/tokenizers/tokenizer.json`. The model **must** use `vocab_size=16384` to match it.

> ✅ **FIXED:** `M01Config.vocab_size` now defaults to `16384` (`src/transformer/config.py`),
> matching `data/tokenizers/tokenizer.json`. Call-sites still pass it explicitly for clarity.

## `train.py` CLI Flags

| Flag | Default | Notes |
|------|---------|-------|
| `--vocab-size` | `16384` | **Must match the 16k tokenizer.** |
| `--resume [PATH]` | `None` | See [Layer-Stacking](#layer-stacking-resume-workflow). `None` = canonical of this run; `PATH` = stack on that checkpoint. |
| `--run-name` | `None` | Pin the `runs/<name>` directory (use `run_test` for smoke tests). |
| `--batch-size` | `4` | **16** was used for the LR tuning below. |
| `--seq-len` | `1024` | |
| `--max-lr` | `3e-4` | Base; optimal = `1.2e-3` at batch 16 (see [LR](#learning-rate-tuning)). |
| `--min-lr-ratio` | `0.1` | Min LR = `max_lr * 0.1`. |
| `--warmup-steps` | `200` | ⚠️ Confound source — see [Warmup Confound](#warmup--short-run-confound). |
| `--max-steps` | `100000` | Use 200–300 for smoke tests. |
| `--weight-decay` | `0.1` | |
| `--grad-accum-steps` | `1` | |
| `--max-norm` | `1.0` | Gradient clip norm. |
| `--log-interval` | `10` | |
| `--save-interval` | `1000` | Canonical checkpoint cadence. |
| `--val-interval` | `500` | Validation cadence (EMA weights). |
| `--data-dir` | `data` | Tokenizer read from `<data-dir>/tokenizers/tokenizer.json`. |

Final report uses `torch.cuda.memory_reserved()` for VRAM (see [VRAM](#vram-reporting)).

## Layer-Stacking Resume Workflow

Because the **full model cannot be trained at once**, knowledge is stacked across runs:

1. **Base run** — train a base checkpoint with `--run-name run_test` (200–300 steps, `--warmup-steps 40`).
2. **Stack run** — resume *onto an explicit checkpoint path*: `--resume runs/run_test/checkpoints/checkpoint.pt`.
   This loads that checkpoint's weights/optimizer/scheduler/RNGs and continues training ("stacks" knowledge into the layers).
3. Repeat stacking as needed. Each `--resume PATH` continues from that exact file.

`engine.resume(checkpoint_path)`:
- `None` → loads the **current run's canonical** `checkpoint.pt`.
- explicit path → loads that file (layer-stacking).

> 🔒 **`_assert_config_compatible()` guards stacking.** Resume raises `ValueError` if the saved
> checkpoint's `vocab_size`, `n_layers`, `d_model`, `num_experts`, `num_shared_experts`, or
> `moe_top_k` differ from the current model. This prevents silently corrupting a stacked model with an
> incompatible architecture. (See `docs/Checkpoint.md`.)

## Learning Rate Tuning

Derived from **300-step runs** (long enough to *complete* warmup):

| Batch | max_lr | Scaling | Final loss |
|-------|--------|---------|------------|
| 16 | `3e-4` | base (1×) | — |
| 16 | **`1.2e-3`** | **4× linear** | **5.448 (best)** |
| 16 | `1.8e-3` | 6× | 5.623 (degraded) |

**Conclusion:** at batch 16, optimal `max_lr = 1.2e-3` (4× linear scaling from the 3e-4 base). Pushing to 6× (1.8e-3) *degrades* loss — do not exceed 4×.

## VRAM Reporting

> ⚠️ **Always report `torch.cuda.memory_reserved()`, never `memory_allocated()`.**
> - `memory_reserved()` ≈ **11 GB** at batch 16 (this is the real GPU footprint).
> - `memory_allocated()` ≈ 2.2 GB is **misleading** (only live tensors, ignores caching allocator).

Both the per-step log and the final `RUN COMPLETION REPORT` use `memory_reserved()`.

## Warmup / Short-Run Confound

> ⚠️ **A run whose `--warmup-steps` ≥ half of `--max-steps` never reaches `max_lr`.** It spends the
> entire run in the linear warmup ramp, so it is **invalid for LR tuning** (you are measuring a
> sub-peak LR). The CLI default is now `--warmup-steps 40`, so short smoke-tests are valid by default.

Valid short smoke-tests must avoid this:
- use `--warmup-steps 40` (short warmup), **or**
- run **200–300 steps** so warmup completes and `max_lr` is actually reached.

## `run_test` Naming Convention

> All training smoke-tests are named **`run_test`** (via `--run-name run_test`) and use **200–300 steps**.
> This makes logs and `runs/` directories predictable and avoids accidental collisions with real runs.

## Optimizations Applied (validated pipeline)

These are wired into the code (not just docs):

- **Corpus loader** — `BinaryCorpusDataset` auto-loads `data/corpus/corpus1_es_wiki_wikisource_tech_10M` (uint16 big-endian shards, ~9.99M tokens) when present; falls back to the raw-text loader otherwise. No more training on the wrong corpus.
- **Mixed precision = bfloat16** — AMP autocast uses `torch.bfloat16` on ROCm (numerically safer than fp16) and drops `GradScaler`. (`src/engine_v2/amp.py`)
- **z-loss single source** — `RouterZLossTerm` owns z-loss; the MoE aux term no longer double-counts it.
- **Gradient accumulation** — already wired (`--grad-accum-steps`); loss is scaled and the optimizer steps every N micro-batches.
- **EMA in-place** — shadow update no longer clones 110M params per step.
- **Checkpoint hash cache** — dataset/tokenizer hashes computed once per run, not re-hashed every save; now hashes the real corpus instead of hardcoding `spanish_pretrain.txt`.
- **ROCm allocator** — `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` set by default (less fragmentation, more usable VRAM).
- **torch.compile (opt-in)** — set `M01_TORCH_COMPILE=1` to enable `torch.compile` for throughput. **OFF by default**: the inductor backend is unstable on this AMD gfx1200 / torch 2.12 ROCm, so validate in a run before enabling.
- **Defaults aligned** — `--batch-size 16`, `--max-lr 1.2e-3`, `--seq-len 2048`, `--warmup-steps 40`.

## `.gitignore` Rules (relevant to training)

The following are **excluded from git** — do not expect them tracked:
- `data/` (entire tree: corpora, tokenizers, raw text)
- `venv/`, `venv_rocm/`
- `runs/**/checkpoints/`, `*.pt`, `*.bin`, `runs/**/*.bin`
- `.qwen/`, `Local_temporal/`, `scratch/`, `results/`, `artifacts/`, `Openspecs_AI/`
- `openspec/changes/` (so the change doc below is documentation-only, not committed)

Checkpoints and runs live locally / in external storage, never in git.

## Training Status (Deferred)

- **Corpus available & now active:** `data/corpus/corpus1_es_wiki_wikisource_tech_10M` (~9.99M tokens, uint16-BE shards per `build_info.txt`) is auto-loaded by `BinaryCorpusDataset` — the previous bug (training on `spanish_pretrain.txt` instead) is fixed.
- **Second corpus available (separate):** `data/corpus/corpus2_es_wiki_gutenberg_17M` (~16.6M tokens, Wikipedia ES + Gutenberg ES, ZERO overlap with corpus 1). Not yet wired into training — point the corpus config to this folder to use it.
- **`runs/` is deleted** — start from a clean state (no inherited checkpoints).
- **No training runs execute today.** This document records the *validated* pipeline and findings so the next session can train immediately without re-deriving them.

## Next Step

When training resumes: start with the [Quick Path](#quick-path-smoke-test) `run_test` (warmup 40, 250 steps, `max_lr=1.2e-3`, batch 16), confirm ~11 GB VRAM, then layer-stack from its checkpoint.
