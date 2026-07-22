# OpenSpec: Organización de Datos y Ficheros del Proyecto (M0.1)

## Estado: Consolidado y Limpio
## Fecha: 2026-07-20

---

## 1. Estructura de Datos de M0.1 (`data/`)
Hemos reorganizado la carpeta `data/` del proyecto para clasificar de manera óptima las fuentes del modelo:
*   **`data/raw_text/`**: Corpus originales de literatura en texto plano.
    *   `quijote.txt`, `becquer.txt`, `chilean_corpus.txt`, `tinyshakespeare.txt`
*   **`data/raw_text/code_projects/`**: Datasets de código fuente real de tus proyectos `Mark-XXXIX-OR`, `Peak` y `proyec Anti`.
*   **`data/tokenizers/`**: El único tokenizer BPE canónico de 16K (`tokenizer.json`).
*   **`data/distillation/`**: Respuestas de destilación del profesor Qwen3.5-9B generadas a través de la API local de LM Studio (`distill_qwen9b_20260718_1803_clean.jsonl`).

---

## 2. Organización de Scripts (`scripts/`)
Para mantener limpia la raíz de herramientas, agrupamos los scripts en subdirectorios temáticos:
*   **`scripts/training/`**: Todos los procesos de entrenamiento y SFT (`train_identity_resilience.py`, `train_mixed_gpu.py`, etc.).
*   **`scripts/evaluation/`**: Evaluación de checkpoints y pruebas de robustez/gaslighting (`evaluate_checkpoints.py`, `run_gaslight_test.py`, `compare.py`).
*   **`scripts/utils/`**: Demos de chat, reportes y utilidades binarias (`chat_demo.py`, `read_gguf_metadata.py`, `generate_report.py`).

---

## 3. Organización de Checkpoints (`checkpoints/`)
*   **Raíz**: Solo conservamos los checkpoints finales listos para inferencia y alineaciones críticas (`m01_hardened_final.pt`, `m01_resilient.pt`, `m01_uncensored.pt`, `final_combined_8k.pt`).
*   **`checkpoints/archive/`**: Archivamos todos los checkpoints intermedios de fases previas y entrenamientos parciales para liberar espacio y no contaminar la raíz.

---

## 4. Reorganización de src/ (2026-07-20)

Para unificar la preparación de datos y clarificar la arquitectura:

| Cambio | Descripción |
|--------|-------------|
| `src/data/` (nuevo) | Canonical location for data preparation — unifies `src/dataset/` + `src/training/dataset.py` |
| `src/dataset/` | Backward-compat shim → `src.data` (existing imports keep working) |
| `src/engine_v2/__init__.py` | Added proper public API exports |
| `src/training/__init__.py` | Updated — only exports active modules (V1 legacy removed from public API) |

**V1 legacy modules** (`loop.py`, `eval.py`, `setup.py`, `datasets.py`) remain for `scripts/training/*.py` compatibility but are no longer exported from `src.training.__init__`.
