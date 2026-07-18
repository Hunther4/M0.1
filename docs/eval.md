# Evaluation Suite

Evaluation tools for the M0.1 model.

## Overview

The evaluation suite provides quantitative metrics and qualitative benchmarks for model checkpoints.

## Quick Start

```bash
# Run basic perplexity evaluation
python src/eval/evaluate.py --checkpoint checkpoints/model.pt

# Run with coherence and NIAH benchmarks
python src/eval/evaluate.py --checkpoint checkpoints/model.pt --coherence --niah

# Verbose output
python src/eval/evaluate.py --checkpoint checkpoints/model.pt --verbose
```

## CLI Options

| Option | Description |
|--------|-------------|
| `--checkpoint PATH` | Path to model checkpoint (required) |
| `--dataset PATH` | Validation dataset path (default: data/tiny_shakespeare_val.txt) |
| `--coherence` | Run coherence benchmark |
| `--niah` | Run Needle-in-a-Haystack benchmark |
| `--device DEVICE` | Device to use: cuda/cpu (default: auto-detect) |
| `-v, --verbose` | Enable verbose logging |

## Output Format

Evaluation results are saved to `artifacts/evals/results_<timestamp>.json`:

```json
{
  "timestamp": "2026-07-18T10:00:00Z",
  "checkpoint": "checkpoints/model.pt",
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

Convert evaluation JSON to Markdown:

```bash
python scripts/generate_report.py --eval artifacts/evals/results_xxx.json
python scripts/generate_report.py --eval results.json --output report.md
```

### Compare Checkpoints

Compare two evaluation runs:

```bash
python scripts/compare.py --json1 eval1.json --json2 eval2.json
python scripts/compare.py --json1 eval1.json --json2 eval2.json --output comparison.md
```

## Metrics

### Perplexity

Measures how well the model predicts the validation data. Lower is better.

### Coherence

Measures local perplexity at 128-token intervals to assess long-range coherence.

### NIAH (Needle in a Haystack)

Tests the model's ability to retrieve a specific piece of information from a larger context.

## Module Structure

```
src/eval/
├── __init__.py      # Package exports
├── utils.py         # Logging, JSON saving
├── metrics.py       # Perplexity, loss calculation
├── qa.py           # Coherence, NIAH benchmarks
└── evaluate.py     # CLI entry point
```