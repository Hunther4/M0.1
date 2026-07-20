# M0.2-Hybrid: Reporte Unificado de Arquitectura y SFT

Este es el reporte oficial unificado de la evolución, diseño y resultados del modelo **M0.2-Hybrid**, entrenado por **Hunther4** en su GPU **AMD Radeon RX 9060XT**.

---

## 1. Ficha Técnica Completa
*   **Modelo**: M0.2-Hybrid (Variante Híbrida de Mezcla de Expertos con MLA)
*   **Parámetros Totales**: 181.46 Millones
*   **Profundidad (Capas)**: 16 Capas Transformer Decoder (Capas 0 y 1 son FFN densas para estabilizar patrones; capas 2 a 15 son MoE).
*   **Ancho de Vector ($d_{model}$)**: 320
*   **Cabezas de Atención ($n_{heads}$)**: 8 (Dimensión de 40 por cabeza)
*   **Dimensión FFN ($d_{ff}$)**: 512
*   **Longitud de Contexto**: 3072 tokens nativos
*   **Vocabulario**: 8192 tokens (BPE optimizado para español y rolplay sin censura)
*   **Posicionamiento**: RoPE (Rotary Positional Embeddings) calculado en FP32 para evitar derivas de precisión.
*   **Normalización**: RMSNorm calculada en FP32 para evitar desbordes en FP16.

---

## 2. Configuración de la Mezcla de Expertos (MoE)
*   **Expertos Totales**: 45 expertos
*   **Expertos Compartidos**: 5 (Siempre activos para procesar la base estructural común del castellano)
*   **Expertos Enrutados**: 40 (Especialistas semánticamente en rolplay, narrativa, lógica, etc.)
*   **Ruteo Dinámico (Top-4 Gating)**: Por cada token, el router selecciona activamente a los **4 mejores expertos ruteados** ($moe\_top\_k = 4$).
*   **Balanceo de Carga**: Loss Auxiliar de 0.1 * Aux_Loss para evitar colapsar el router durante ráfagas de entrenamiento continuo.

---

## 3. Multi-head Latent Attention (MLA)
*   **Compresión de Baja Dimensión**: Comprime la clave (Key) y el valor (Value) de los tokens en un vector latente de baja dimensión ($d_c = 128$), reduciendo drásticamente la huella de memoria del KV cache.
*   **Proyección Separada de Posición**: Extrae 16 dimensiones por cabeza para ser rotadas dinámicamente usando RoPE en FP32, permitiendo un escalamiento limpio hasta 3072 tokens nativos de contexto sin pérdida de relación de posición.

---

## 4. Filosofía del Entrenamiento Acumulativo
M0.2-Hybrid se entrena mediante sesiones secuenciales y acotadas en tiempo de 28-30 minutos utilizando el script `train_session_continual.py`. Cada sesión:
1. Retoma el entrenamiento cargando `m01_180m_latest.pt` (pesos + optimizador AdamW + programadores de escala).
2. Entrena en ráfagas acumulando conocimiento sobre la misma base estructural.
3. Guarda snapshots permanentes de seguridad (`m01_180m_milestone_XXXXXX.pt`) cada 5000 pasos.
