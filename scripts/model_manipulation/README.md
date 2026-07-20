# 🧠 M0.1: Model Manipulation & Continual Training Guide

This directory contains tools and workflows to perform **Model Merging**, **Progressive Depth Expansion (Model Growth)**, and **Session-Based Continual Training**.

---

## 🛠️ Provided Scripts

### 1. `init_new_180m_model.py` (Inicializador de 180M)
Inicializa desde cero el checkpoint base para el nuevo modelo expandido de **179.8M de parámetros**. Esta arquitectura está optimizada para rolplay y texto plano, con 14 capas, 5 expertos compartidos fijos y 40 expertos ruteados finamente granulados (Top-4).

* **Uso:**
  ```bash
  venv_rocm\Scripts\python.exe scripts/model_manipulation/init_new_180m_model.py
  ```

### 2. `expand_model.py` (Depth Up-Scaling)
Agrandar la capacidad de un modelo existente agregando bloques/capas sin perder el conocimiento aprendido. Mapea progresivamente los pesos de las capas previas entrenadas a la nueva arquitectura.

* **Uso para expandir de 10 a 14 capas:**
  ```bash
  venv_rocm\Scripts\python.exe scripts/model_manipulation/expand_model.py --checkpoint checkpoints/m01_hardened_final.pt --output checkpoints/m01_extended_14layers.pt --layers 14
  ```

### 3. `merge_checkpoints.py` (Model Merging)
Fusionar dos checkpoints que compartan **exactamente** la misma arquitectura mediante interpolación de pesos. Muy útil para combinar modelos adaptados a distintas tareas o datasets (ej. un modelo alineado a modismos y otro sin censura).

* **Uso:**
  ```bash
  venv_rocm\Scripts\python.exe scripts/model_manipulation/merge_checkpoints.py
  ```

### 4. `train_session_continual.py` (Entrenamiento Acumulativo de 30 Minutos)
Ejecutar ráfagas de entrenamiento con control de tiempo para ir sumando conocimiento de forma estable, conservando el estado del modelo, del optimizador y del GradScaler.

* **Uso estándar (ejemplo apuntando al modelo de 180M recién creado):**
  ```bash
  venv_rocm\Scripts\python.exe scripts/training/train_session_continual.py --checkpoint checkpoints/m01_extended_180m_base.pt --output checkpoints/m01_extended_180m_base.pt --duration_min 30 --lr 5e-4
  ```

---

## ⚠️ Qué HACER y qué NO HACER

### 🟢 Qué HACER
* **Usar tasas de aprendizaje (`--lr`) muy bajas para entrenamiento continuo:** Se recomienda usar entre `1e-5` y `3e-5` al continuar el entrenamiento de modelos pre-entrenados. Para modelos inicializados de cero (como el de 180M), podés arrancar con una tasa de `3e-4` a `5e-4` y luego ir bajándola.
* **Conservar el estado del optimizador:** Asegurarse de sobreescribir el mismo checkpoint (o arrastrar el estado) para que AdamW mantenga sus momentos y velocidades. `train_session_continual.py` ya se encarga de esto de forma nativa.
* **Hacer respaldos preventivos:** Antes de iniciar una sesión de entrenamiento sobreescribiendo un checkpoint, haz una copia de seguridad en la carpeta `checkpoints/archive/`.
* **Mantener un dataset mixto (Replay Buffer):** Al entrenar con datos nuevos, incluye al menos un 10-20% de datos lingüísticos generales para evitar que el modelo sufra degradación del lenguaje.

### 🔴 Qué NO HACER
* **❌ NO intentar fusionar modelos de arquitecturas distintas:** Intentar fusionar un checkpoint de 12.5M con uno de 55.1M o el nuevo de 180M fallará debido a la diferencia de formas en los tensores de pesos (capas, dimensiones y número de expertos).
* **❌ NO entrenar sin cargar el estado del optimizador en ráfagas cortas:** Si entrenás 30 minutos reiniciando el optimizador a cero en cada sesión, causarás picos de pérdida devastadores (*loss spikes*) en cada inicio.
* **❌ NO cambiar el vocabulario del tokenizador a mitad de camino:** Si cambias el tokenizador, la matriz de embeddings (`embedding.embedding.weight`) dejará de corresponderse con los tokens y el modelo escupirá basura.
