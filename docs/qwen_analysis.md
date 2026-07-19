# Análisis Arquitectónico y Ejecución de Qwen (Docs Proyecto)

Este documento detalla el análisis de ingeniería inversa realizado a los binarios GGUF de los modelos Qwen locales y las pautas para su ejecución e integración en el ecosistema M0.

---

## 1. Descubrimientos de Arquitectura (Auditoría GGUF)

A través del parser binario [read_gguf_metadata.py](file:///D:/Proyectos/M0.1/scripts/read_gguf_metadata.py), analizamos las cabeceras de los modelos de Qwen locales:

### Qwen3.5-9B-Uncensored (Dense)
*   **Capas**: 32 bloques de atención pre-normalizados con RMSNorm.
*   **Contexto**: 262K tokens de ventana máxima.
*   **GQA (Grouped-Query Attention)**: Relación de cabezas de **16:4 (ratio 4x)**. Las cabezas de clave y valor se agrupan cada 4 cabezas de consulta, reduciendo el cuello de botella del VRAM bandwidth en inferencia.

### Carnice-Qwen3.6-MoE-35B (APEX MoE)
*   **Capas**: 41 bloques.
*   **Contexto**: 262K tokens.
*   **GQA**: Relación de cabezas de **16:2 (ratio 8x)**. Compresión extrema para mitigar el costo del KV cache en contextos masivos.
*   **Mixture of Experts**: **256 expertos ruteados**. La división fina de expertos (Fine-grained MoE) permite que cada token acceda a nichos de conocimiento muy específicos sin aumentar la carga computacional activa.

---

## 2. Pautas de Ejecución e Integración Local

Para correr estos modelos GGUF de forma óptima aprovechando la GPU **AMD Radeon RX 9060XT**:

1.  **Ejecución en LM Studio / llama.cpp**:
    *   Cargar el modelo GGUF y configurar el **GPU Offload** al máximo de capas posibles (32 capas en el 9B caben completas en VRAM; el MoE 35B puede requerir offload parcial según la capacidad de memoria).
    *   Activar el soporte de **HIP/ROCm** en la configuración de LM Studio para habilitar el cómputo acelerado en los Tensor Cores de AMD.
2.  **Destilación en el Proyecto M0**:
    *   **Tokenizer**: Reutilizar el formato de tokenizador BBPE con vocabularios de amplio espectro (151K tokens) para evitar el deletreo por caracteres.
    *   **SwiGLU**: Integrar las proyecciones SwiGLU en las capas de FFN de M0.2 y M0.3.
    *   **GQA**: Mapear la lógica de Grouped-Query Attention en el bloque de atención causal para optimizar el KV Cache.
    *   **APEX Routing**: Mapear el ruteo Top-k con expertos especializados de 256 divisiones para el modelo MoE final.
