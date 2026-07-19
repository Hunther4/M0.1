# Información de Modelos de Frontera: DeepSeek, GLM y Qwen

Este archivo resume las especificaciones técnicas, arquitecturas, pesos (parámetros) y detalles de entrenamiento recopilados para los modelos de inteligencia artificial de frontera de las familias DeepSeek, Qwen y GLM.

---

## 1. Familia DeepSeek (V3, V3.2 y V4 Pro/Flash)

### DeepSeek-V3 y V3.2
*   **Parámetros Totales**: 671 Mil Millones (671B).
*   **Parámetros Activos**: 37 Mil Millones (37B) por token.
*   **Ventana de Contexto**: 128K tokens.
*   **Arquitectura de Atención**: **MLA (Multi-head Latent Attention)**. Comprime las representaciones de Key (K) y Value (V) en un vector latente de baja dimensión para reducir drásticamente el uso de memoria del KV cache (hasta 57 veces en inferencia).
*   **Mejoras en V3.2**: Integra **DSA (DeepSeek Sparse Attention)**, que aplica una selección de top-k más fina sobre la atención dispersa, y técnicas de alineación por Refuerzo con Recompensas Verificables (RLVR).

### DeepSeek-V4 (Pro y Flash) - *Lanzado en Abril 2026*
*   **Variante Pro**: 1.6 Billones (1.6T) de parámetros totales, **49B activos** (tasa de activación de ~3.1%). Diseñado para tareas complejas agénticas y de razonamiento.
*   **Variante Flash**: 284 Mil Millones (284B) de parámetros totales, **13B activos**. Optimizado para alta velocidad e inferencia económica.
*   **Atención Híbrida (CSA + HCA)**:
    *   **CSA (Compressed Sparse Attention)**: Comprime ventanas locales de 4 tokens en una sola entrada utilizando ponderación aprendida y un indexador en precisión FP4 (Lightning Indexer) que recupera los bloques más relevantes.
    *   **HCA (Heavily Compressed Attention)**: Comprime ventanas globales de 128 tokens no solapados para capturar la historia lejana del contexto.
*   **Estabilización**: Utiliza **mHC (Manifold-Constrained Hyper-Connections)** para asegurar la estabilidad del gradiente en capas ultra-profundas y el optimizador **Muon** para una convergencia más rápida durante el preentrenamiento.
*   **Entrenamiento**: Dos etapas post-entrenamiento: cultivo independiente de expertos por dominio y consolidación unificada mediante destilación guiada por políticas (on-policy distillation).

---

## 2. Familia GLM (Zhipu AI / Z.ai)

### GLM-5 y 5.1
*   **Parámetros**: 744B de parámetros totales, **40B activos** en inferencia.
*   **Detalles**: Utiliza DeepSeek Sparse Attention (DSA) para escalar su capacidad manteniendo costos controlados.

### GLM-5.2 - *Lanzado en Junio 2026*
*   **Parámetros**: 753 Mil Millones (753B) totales.
*   **Ventana de Contexto**: 1 Millón (1M) de tokens sin pérdida (lossless).
*   **Licencia**: Licencia **MIT** de código abierto y libre uso comercial.
*   **Arquitectura IndexShare**: Reutiliza el mismo indexador de tokens de atención dispersa a lo largo de un bloque de 4 capas consecutivas. Esto amortiza el costo de la selección de tokens y resulta en una reducción de **2.9 veces en los FLOPs por token** en contextos largos de 1M de tokens.

---

## 3. Familia Qwen (Alibaba)

### Detalles Generales
*   **Pesos Abiertos**: Las series Qwen (como Qwen-2.5 y modelos superiores) se publican bajo licencias comerciales amigables (ej., Apache 2.0).
*   **Entrenamiento**: Los conjuntos de datos de preentrenamiento originales son privados y superan los 15-20 Billones (Trillions) de tokens multilenguaje con una gran proporción de código y matemáticas.
*   **Alineamiento por Refuerzo**: Introducen algoritmos DPO (Direct Preference Optimization) y PPO avanzados alineados con el razonamiento sistemático del usuario, sirviendo de base para destilar sets de instrucciones masivos utilizados por la comunidad.

---

## 4. Resumen de Pesos y Parámetros

| Modelo | Parámetros Totales | Parámetros Activos | Ventana de Contexto | Innovación Principal |
| :--- | :--- | :--- | :--- | :--- |
| **DeepSeek-V3** | 671B | 37B | 128K | MLA + Multi-Token Prediction (MTP) |
| **DeepSeek-V3.2** | 671B | 37B | 128K+ | DeepSeek Sparse Attention (DSA) |
| **DeepSeek-V4-Flash** | 284B | 13B | 1M | CSA + HCA (Bajo costo de KV Cache) |
| **DeepSeek-V4-Pro** | 1.6T | 49B | 1M | mHC + Gating FP4 (Lightning Indexer) |
| **GLM-5.2** | 753B | ~40B | 1M | IndexShare (Ahorro de 2.9x FLOPs) |
