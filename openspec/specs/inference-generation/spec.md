# Spec: inference-generation

## Overview

Autoregressive text generation with KV cache for efficient sequential decoding. The generate function tokenizes a prompt, runs the model forward, samples tokens iteratively, and returns the full generated text.

## Requirements

### Requirement: Autoregressive Generation Loop

The system MUST provide a `generate` function that produces text by iteratively sampling one token at a time, appending it to the sequence, and feeding the updated sequence back into the model.

**Interface:**
```python
def generate(
    model: TransformerLM,
    tokenizer: Any,
    prompt: str,
    max_length: int = 256,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    device: str = "auto",
) -> str:
```

**Scenarios:**

- **Given** a trained model and tokenizer, **When** `generate(model, tokenizer, "To be")` is called, **Then** the output MUST be a non-empty string.
- **Given** a trained model and tokenizer, **When** `generate(model, tokenizer, prompt, max_length=50)` is called, **Then** the total generated length MUST NOT exceed `max_length` tokens.
- **Given** a trained model and tokenizer, **When** `generate` is called, **Then** the prompt MUST be included at the start of the output.

### Requirement: KV Cache Usage

The generate function MUST use KV cache to avoid recomputing key-value pairs for previously processed tokens. After the initial prefill of the prompt, each subsequent forward pass MUST process only the last token.

**Scenarios:**

- **Given** a model with KV cache, **When** `generate` is called, **Then** the initial prompt MUST be processed in a single forward pass (prefill).
- **Given** a model with KV cache, **When** generation proceeds, **Then** each subsequent step MUST forward only the last token (not the full sequence).
- **Given** a model with and without KV cache, **When** the same prompt and parameters are used, **Then** the output text MUST be identical (deterministic equivalence).

### Requirement: Device Auto-Detection

The generate function MUST support automatic device selection. When `device="auto"`, it MUST use CUDA if available, otherwise CPU. Explicit `"cuda"` or `"cpu"` values MUST be respected.

**Scenarios:**

- **Given** `device="auto"` and CUDA is available, **When** `generate` is called, **Then** inference MUST run on CUDA.
- **Given** `device="auto"` and CUDA is not available, **When** `generate` is called, **Then** inference MUST run on CPU.
- **Given** `device="cpu"`, **When** `generate` is called, **Then** inference MUST run on CPU regardless of CUDA availability.
- **Given** `device="cuda"` and CUDA is available, **When** `generate` is called, **Then** inference MUST run on CUDA.

### Requirement: Sampling Parameters Delegation

The generate function MUST pass `temperature`, `top_k`, and `top_p` to the `sample` function at each step.

**Scenarios:**

- **Given** `generate(model, tokenizer, prompt, temperature=0.5)`, **When** called, **Then** each sampling step MUST use `temperature=0.5`.
- **Given** `generate(model, tokenizer, prompt, top_k=20, top_p=0.9)`, **When** called, **Then** each sampling step MUST use both `top_k=20` and `top_p=0.9`.

### Requirement: Config Access

The generate function MUST access model architecture parameters (vocab_size, n_layers, etc.) via `model.config`. The TransformerLM MUST store the config during initialization.

**Scenarios:**

- **Given** a `TransformerLM` instance, **When** `model.config` is accessed, **Then** it MUST return the `M01Config` used during initialization.
- **Given** `generate` is called, **When** the model is loaded, **Then** `model.config` MUST be used to determine vocabulary size and layer count.
