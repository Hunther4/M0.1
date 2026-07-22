# Evaluation Suite

Evaluation tools for the M0.1 model.

## Tokenizer & Vocab (important)

`evaluate.py` reconstructs the model from the checkpoint's embedded `config` dict, then loads the
**real tokenizer**:

- **Tokenizer path:** `data/tokenizers/tokenizer.json`.
- **Vocab size:** taken from the checkpoint's `config.vocab_size`; if absent, **defaults to `16384`**.
- There is **no 32k tokenizer** — the only tokenizer is the 16k one used in training.

> ⚠️ The eval model is built with `M01Config(vocab_size=config_dict.get("vocab_size", 16384), …)`.
> If a checkpoint was trained with `vocab_size=16384` (the validated default), evaluation must use the
> same 16k tokenizer, or token IDs will not align.

## Quick Start

```bash
# Basic perplexity evaluation
python src/eval/evaluate.py --checkpoint runs/run_test/checkpoints/checkpoint.pt

# With coherence and NIAH benchmarks
python src/eval/evaluate.py --checkpoint runs/run_test/checkpoints/checkpoint.pt --coherence --niah

# Verbose
python src/eval/evaluate.py --checkpoint runs/run_test/checkpoints/checkpoint.pt --verbose
```

## CLI Options

| Option | Description |
|--------|-------------|
| `--checkpoint PATH` | Path to model checkpoint `.pt` (required) |
| `--dataset PATH` | Validation dataset path (default: `data/tiny_shakespeare_val.txt`) |
| `--coherence` | Run coherence benchmark |
| `--niah` | Run Needle-in-a-Haystack benchmark |
| `--device DEVICE` | Device to use: cuda/cpu (default: auto-detect) |
| `-v, --verbose` | Enable verbose logging |

## Output Format

Evaluation results are saved to `artifacts/evals/results_<timestamp>.json`:

```json
{
  "timestamp": "2026-07-18T10:00:00Z",
  "checkpoint": "runs/run_test/checkpoints/checkpoint.pt",
  "metrics": {
    "perplexity": 15.234
  },
  "qa": {
    "coherence": {
      "average_coherence": 15.2,
      "interval": 128,
      "interval_perplexities": [...]
    },
    "niah": {
      "needle": "42",
      "accuracy": 0.95,
      "context_length": 512
    }
  }
}
```

## Reporting Scripts

### Generate Report

```bash
python scripts/generate_report.py --eval artifacts/evals/results_xxx.json
python scripts/generate_report.py --eval results.json --output report.md
```

### Compare Checkpoints

```bash
python scripts/compare.py --json1 eval1.json --json2 eval2.json
python scripts/compare.py --json1 eval1.json --json2 eval2.json --output comparison.md
```

## Metrics

### Perplexity

Measures how well the model predicts the validation data. Lower is better.

### Coherence

Runs one causal forward over the complete bounded prompt, computes token-level negative log-likelihood and aggregates it into reporting intervals. Intervals never remove the preceding context.

### NIAH (Needle in a Haystack)

Inserts the needle once into non-repetitive filler at a configurable depth between 10% and 90%. A retrieval query is appended after the complete context and the benchmark scores teacher-forced reproduction of the needle as the answer. Reported metadata includes requested/actual depth, answer position and filler token diversity.

NIAH accuracy is exact token argmax accuracy; `avg_probability` reports mean probability assigned to the expected answer tokens. This prevents a repeated local phrase or a fixed probability threshold from counting as retrieval.

## Module Structure

```
src/eval/
├── __init__.py      # Package exports
├── utils.py         # Logging, JSON saving
├── metrics.py       # Perplexity, loss calculation
├── qa.py            # Coherence, NIAH benchmarks
└── evaluate.py      # CLI entry point
```
