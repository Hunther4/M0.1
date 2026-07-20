# 🧠 M0.1 — Guía de Entrenamiento y Estructura del Pipeline por Fases

Esta es la documentación oficial consolidada del pipeline de entrenamiento del modelo **M0.1** (Sparse Mixture of Experts de ~181M parámetros), diseñado por **Hunther4** para ser entrenado en su GPU **AMD Radeon RX 9060XT** en Windows mediante PyTorch ROCm nativo.

---

## 1. Estructura de Módulos de Entrenamiento (`src/training/`)

La lógica de entrenamiento está modularizada bajo `src/training/` de la siguiente manera:

* **`setup.py`**: Configuración de dispositivo (`cuda` / `cpu`), optimización de memoria y configuración de consola.
* **`checkpoint.py`**: Administración de checkpoints con un contrato de serialización robusto.
* **`config.py`**: Parámetros de entrenamiento (learning rate, weight decay, epochs, etc.).
* **`dataset.py` & `datasets.py`**: Cargadores de datos y conjuntos de datos especializados (`TinyShakespeareDataset`, `JsonlDataset` para shards sintéticos, y `AmplifiedDialogueDataset` para quemado de identidad).
* **`loop.py`**: Bucle de entrenamiento y evaluación paso a paso por batches.
* **`eval.py`**: Evaluación cuantitativa de pérdida y Perplejidad (PPL).
* **`moe_metrics.py`**: Métricas de balanceo de expertos, tasa de enrutamiento y entropía de decisiones del Gating Network.

---

## 2. Scripts de Entrenamiento de Fase por Fase (`scripts/training/`)

El modelo se entrena en etapas progresivas. A continuación se listan los scripts premium del pipeline en `scripts/training/`:

### 🚀 Fase 1: Pre-Entrenamiento del Modelo Base
* **`train_phase1.py`**: Inicialización del modelo base y entrenamiento preliminar con TinyShakespeare (contexto corto a 256 tokens).
* **`train_vocab_8k_gpu.py`**: Entrenamiento masivo utilizando un tokenizador BPE entrenado a 8192 de vocabulario. Incluye la suite de validación de 5 pruebas:
  1. *Perplejidad (PPL)* en set de validación.
  2. *Sintaxis JSON*: Verificación de formato para llamadas de herramientas (`<|tool_call|>`).
  3. *Precisión de Ruteo de Herramientas*: Derivación del Gating a la herramienta correcta.
  4. *Needle in a Haystack (NIAH)*: Recuperación de información exacta en el contexto.
  5. *Ortografía y Ortometría*: Tasa de palabras reconocibles en generación creativa.
* **`train_final_combined.py`**: Script de entrenamiento unificado con el corpus clásico consolidado en español (Don Quijote, Bécquer Leyendas, etc.).

### 🎭 Fase 2: Alineación de Identidad e Instrucción (SFT)
* **`train_identity_sft.py`**: Inyección profunda de la personalidad e identidad del modelo (M0.1, creado por Hunther4, GPU RX 9060XT) mediante el conjunto de diálogos ampliados y un factor de amplificación de $30\times$.
* **`train_identity_hardened.py` / `train_identity_resilience.py` / `train_identity_correction.py`**: Ajuste fino de la red de Gating y capas del MoE para hacer la identidad resiliente y evitar jailbreaks de prompt o colapsos de comportamiento.
* **`train_slang_alignment.py`**: Alineación lingüística regional para asegurar la naturalidad del español conversacional.

### 🛡️ Fase 3: Instrucción Avanzada y Robustez
* **`train_cot_sft.py`**: Alineación de razonamiento paso a paso (Chain of Thought).
* **`train_adversarial_defence.py`**: Entrenamiento contra ataques adversariales y prompts maliciosos diseñados para forzar desalineación.

---

## 3. Configuración y Parámetros Clave de VRAM

Debido al target físico (RX 9060XT de 16GB de VRAM), la configuración nativa del contexto y los batches se optimiza de la siguiente manera:

* **Longitud de Secuencia Máxima**: **1024 tokens** (para evitar consumos excesivos de memoria en la atención cuadrática de MLA).
* **Tamaño de Batch**: 8 (para pre-entrenamiento base) y 4 (para fine-tuning de diálogos y SFT).
* **Precisión**: AMP (Automatic Mixed Precision) habilitada con escalador de gradientes (`GradScaler`) en scripts avanzados para acelerar el procesamiento y reducir la huella de memoria.
