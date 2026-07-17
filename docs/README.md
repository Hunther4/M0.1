# M0.1 Project Overview & Foundations

This repository contains the foundations for M0.1, establishing the baseline environment, dataset ingestion framework, a custom Byte-level BPE tokenizer, and a visual token counter CLI.

## Setup Instructions

### Prerequisites
- Python 3.10 or higher
- `pip` (Python package manager)

### Installation
1. Clone or download the repository.
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows (cmd):
   venv\Scripts\activate.bat
   # On Windows (powershell):
   .\venv\Scripts\Activate.ps1
   # On Linux/macOS:
   source venv/bin/activate
   ```
3. Install the dependencies listed in `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

## Roadmap

* **Fase 0: Foundations (Completada)**
  * Inicialización del entorno de desarrollo.
  * Ingesta de datos (`prep.py`) para TinyShakespeare o texto local validado en UTF-8.
  * Tokenizador Byte-level BPE (`bpe.py`) con 256 bytes base + `<|endoftext|>` (256) y `<|pad|>` (257), entrenado hasta vocabulario de 32768.
  * CLI de conteo y visualización de tokens (`counter.py`) con colores ANSI y métricas.
  * Suite de testing (`pytest`) completa.
* **Fase 1: Model Architecture (Completada)**
  * M01Config dataclass with validated defaults.
  * Capa de Embeddings (tied) — `TokenEmbedding` with weight sharing.
  * Posicionales: RoPE (Rotary Position Embedding) — explicit sin/cos implementation.
  * Mecanismo de Attention (Causal Self-Attention, KV-Cache).
  * FeedForward (SwiGLU) and MoE placeholder.
  * 75 tests passing (33 new transformer tests + 42 Fase 0 tests).
* **Fase 2: Bloque Transformer y Entrenamiento**
  * Bloque Transformer completo (RMSNorm, SwiGLU, Residual Connection).
  * Arquitectura GPT Decoder-only (12 capas, 10 heads, d_model=640, d_ff=1728).
  * Bucle de entrenamiento desde cero (sin HuggingFace Trainer).
* **Fase 3: Inferencia y Generación**
  * Script de inferencia autorregresivo.
  * Muestreo (Sampling): Temperature, Top-K, Top-P.
* **Fase 4: Evaluación**
  * Cálculo de Loss y Perplexity.
* **Evolución Futura**
  * **M0.1R**: Refactorización y optimización.
  * **MoE**: Evolución a Mixture of Experts.
  * **M0.2**: Rediseño.
  * **Mamba**: Evolución de arquitectura hacia State Space Models (SSM).
  * **M1**: Modelo a gran escala.

## Execution Instructions

### Dataset Ingestion
To download TinyShakespeare:
```bash
python -m src.dataset.prep --download
```
To ingest a local text file:
```bash
python -m src.dataset.prep --input path/to/input.txt --output path/to/output.txt
```

### BPE Tokenizer Training
To train the tokenizer or use it programmatically:
```python
from src.tokenizer.bpe import Tokenizer

tokenizer = Tokenizer()
tokenizer.train("your text data here", vocab_size=32768)
tokenizer.save("tokenizer.json")
```

### Token Counter CLI
To inspect tokenization and count tokens in a string or file with terminal-colored highlighting:
```bash
python -m src.tools.counter --tokenizer tokenizer.json --text "Hello world, this is M0.1 tokenizer visualization!"
```
Or for a file:
```bash
python -m src.tools.counter --tokenizer tokenizer.json --file path/to/input.txt
```

### Running Tests
Run the test suite using `pytest`:
```bash
pytest
```

## Configuration Settings
The repository's global coding conventions and behaviors are driven by `openspec/config.yaml`.
* **Python version**: Python 3.10+
* **Testing framework**: `pytest`
* **Tokenizer Vocabulary Limit**: up to `32,768` (base 256 bytes + special tokens `<|endoftext|>` (256) and `<|pad|>` (257) + trained merges)
* **Pre-tokenization Regex pattern**: GPT-2 standard regex:
  `r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"`

## Coding Rules
* **Code Style**: Adhere to PEP 8. Use Ruff/Black format rules.
* **Typing**: Use static typing with `mypy` style annotations where appropriate (e.g. `List`, `Dict`, `Tuple`, `Set`, `Union` from `typing`).
* **Testing**: Write pytest tests under the `tests/` directory. Target unit coverage for data preprocessing, tokenizer core APIs (`train`, `encode`, `decode`, `save`, `load`), and visual counter CLI input configurations.
* **Docstrings**: Maintain documentation integrity. Write clean docstrings for all modules, classes, and public methods.
