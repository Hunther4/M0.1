# M0.1

Decoder-only transformer language model from scratch in PyTorch. ~99.7M params, trained on Spanish corpora.

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![Research](https://img.shields.io/badge/purpose-research-BF4FE0?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-yellow?style=flat-square)

---

## Architecture

| Component | Detail |
|-----------|--------|
| Layers | 12 (2 dense FF + 10 MoE) |
| d_model | 640, 10 heads, d_head=64 |
| Attention | Multi-head Latent Attention (MLA) — KV cache compressed via low-rank latent (128 dim) + RoPE (16 dim/head). 77% less cache than MHA. |
| MoE | 4 routed + 1 shared expert, top-2 routing, DeepSeek-style load balancing + Z-loss |
| FF | SwiGLU (gate/up/down projections) |
| Position | Rotary Position Embeddings (RoPE) |
| Norm | RMSNorm pre-norm |
| Tie | Weight-tied embeddings (shared input/output matrix) |
| Vocab | 16,384 BPE tokenizer |

## Training System

Entry point: `python -m src.training.train [--corpus-dir path]`

**TrainingEngineV2** — FSM-based engine with:
- Finite State Machine (`INIT → LOAD → TRAIN → VALIDATE → SAVE → EVALUATE → EXPORT → FINISHED`, plus `RECOVERING`/`ERROR`)
- EventBus for decoupled plugin communication
- LossPipeline with composable terms (CE, aux loss, Z-loss)
- AsyncCheckpointManagerV2 with background save + SHA256 integrity
- HealthChecker (NaN/Inf detection, auto rollback + LR halve)
- EMA shadow weights, AMP (FP16/BF16), gradient accumulation
- ExperimentManager with structured run directories (`runs/run_XXXX/`)

## Data

Binary shard corpora under `data/corpus/` (uint16 BE, built with `src/data/prep.py`):

| Corpus | Tokens | Sources |
|--------|--------|---------|
| corpus1 | ~10M | Wikipedia ES, Wikisource, tech texts |
| corpus2 | ~16.6M | Wikipedia ES, Gutenberg (zero overlap) |

Switch corpus at runtime: `--corpus-dir data/corpus/corpus2_es_wiki_gutenberg_17M`

## Quick Start

```bash
git clone https://github.com/Hunther4/M0.1.git && cd M0.1
pip install torch numpy pytest
python -m src.training.train --batch-size 4 --max-steps 1000
```

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--batch-size` | 4 | Batch size per step |
| `--seq-len` | 1024 | Sequence length |
| `--max-lr` | 3e-4 | Peak learning rate |
| `--max-steps` | 100,000 | Total training steps |
| `--warmup-steps` | 200 | LR warmup steps |
| `--grad-accum-steps` | 1 | Gradient accumulation |
| `--corpus-dir` | — | Override corpus path |
| `--resume` | — | Resume from checkpoint |

## Project Structure

```
M0.1/
├── src/
│   ├── transformer/   config, attention (MLA), moe, feedforward, rope, kv_cache
│   ├── model/         lm, block, rms_norm
│   ├── training/      train.py, dataset, config, checkpoint (V1 legacy kept for compat)
│   ├── engine_v2/     TrainingEngineV2 (fsm, bus, loss_pipeline, ema, amp, ...)
│   ├── tokenizer/     bpe.py
│   ├── data/          prep.py (corpus builder)
│   ├── inference/     generate.py, cli.py
│   └── eval/          evaluation pipeline, metrics, QA
├── tests/             ~79 tests (pytest)
├── scripts/           train.py, compare.py, generate_report.py, expand_model.py, merge_checkpoints.py
└── docs/              architecture docs
```

## Testing

```bash
python -m pytest tests/ -q
```

Covers: attention (MLA/MHA/Hybrid), MoE (routing, capacity, metrics), blocks, embeddings, RoPE, RMSNorm, KV cache, training engine V2, tokenizer, inference, evaluation, checkpoint integrity.

## Inference

```python
from src.model.lm import TransformerLM; from src.transformer.config import M01Config
from src.tokenizer.bpe import Tokenizer; from src.inference.generate import generate

model = TransformerLM(M01Config(vocab_size=16384))
model.load_state_dict(torch.load("checkpoint.pt", weights_only=True)["model_state_dict"])
model.eval().to("cuda")
tok = Tokenizer(); tok.load("data/tokenizers/tokenizer.json")
print(generate(model, tok, "To be, or not to be", max_gen_len=200, temperature=0.8))
```

## License

MIT

## References

- DeepSeek-V2/V3, Llama 2, SwiGLU, RoFormer, RMSNorm
