# Spec: inference-cli

## Overview

Command-line interface for text generation from trained checkpoints. Wraps the generate function with argparse-based parameter handling and checkpoint loading.

## Requirements

### Requirement: CLI Entry Point

The system MUST provide a CLI entry point accessible via `python -m src.inference.cli`.

**Interface:**
```bash
python -m src.inference.cli \
  --prompt "To be or not to be" \
  --max-length 256 \
  --temperature 1.0 \
  --top-k 0 \
  --top-p 1.0 \
  --device auto \
  --checkpoint checkpoints/checkpoint.pt
```

**Scenarios:**

- **Given** `python -m src.inference.cli --help`, **When** executed, **Then** usage information MUST be printed and the process MUST exit with code 0.
- **Given** `python -m src.inference.cli --prompt "Hello"`, **When** executed, **Then** generated text MUST be printed to stdout.

### Requirement: Required Arguments

The CLI MUST require at least the `--prompt` argument. All other arguments MUST have sensible defaults.

**Scenarios:**

- **Given** `python -m src.inference.cli` (no arguments), **When** executed, **Then** an error MUST be displayed indicating `--prompt` is required.
- **Given** `python -m src.inference.cli --prompt "test"`, **When** executed with no other args, **Then** defaults MUST be used: max_length=256, temperature=1.0, top_k=0, top_p=1.0, device=auto.

### Requirement: Checkpoint Loading

The CLI MUST load a trained checkpoint via CheckpointManager and instantiate a TransformerLM with the saved configuration.

**Scenarios:**

- **Given** `--checkpoint checkpoints/checkpoint.pt`, **When** the file exists, **Then** the model weights MUST be loaded and used for generation.
- **Given** `--checkpoint nonexistent.pt`, **When** the file does not exist, **Then** a FileNotFoundError MUST be raised with a clear message.
- **Given** no `--checkpoint` argument, **When** executed, **Then** the default path `checkpoints/checkpoint.pt` MUST be used.

### Requirement: Device Selection

The CLI MUST support `--device` with values `auto`, `cuda`, or `cpu`.

**Scenarios:**

- **Given** `--device auto`, **When** executed, **Then** CUDA MUST be used if available, else CPU.
- **Given** `--device cpu`, **When** executed, **Then** CPU MUST be used regardless of CUDA availability.
- **Given** `--device cuda`, **When** executed and CUDA is available, **Then** CUDA MUST be used.

### Requirement: Output Format

The CLI MUST print the generated text to stdout, preceded by a clear label.

**Scenarios:**

- **Given** a successful generation, **When** output is printed, **Then** it MUST include a label like `Generated:` followed by the text.
- **Given** a generation error, **When** the error occurs, **Then** a clear error message MUST be printed to stderr.
