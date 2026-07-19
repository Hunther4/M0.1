# M0.1-MoE: Reporte Unificado de Arquitectura y SFT

Este es el reporte oficial unificado de la evolución, diseño y resultados del modelo **M0.1-MoE**, entrenado por **Hunther4** en su GPU **AMD Radeon RX 9060XT**.

---

## 1. Ficha Técnica Completa
*   **Modelo**: M0.1-MoE (Variante Lite de Mezcla de Expertos)
*   **Parámetros Totales**: 12.56 Millones
*   **Profundidad (Capas)**: 4 Capas Transformer Decoder
*   **Ancho de Vector ($d_{model}$)**: 256
*   **Cabezas de Atención ($n_{heads}$)**: 4 (Dimensión de 64 por cabeza)
*   **Dimensión FFN ($d_{ff}$)**: 512
*   **Longitud de Contexto**: 256 tokens
*   **Vocabulario**: 8192 tokens (BPE optimizado para español, modismos y fantasía)
*   **Posicionamiento**: RoPE (Rotary Positional Embeddings)

---

## 2. Configuración de la Mezcla de Expertos (MoE)
*   **Expertos Totales**: 6 expertos
*   **Expertos Compartidos**: 2 (Siempre activos para procesar la base común del castellano)
*   **Expertos Enrutados**: 4 (Especializados semánticamente)
*   **Ruteo Dinámico (Top-2 Gating)**: Por cada token procesado, una compuerta lineal selecciona activamente a los **2 mejores expertos ruteados** ($moe\_top\_k = 2$), amortizando los costos de inferencia en la GPU.

---

## 3. Atención Híbrida (CSA + HCA)
*   **Compressed Sparse Attention (CSA)**: Comprime la memoria KV de los tokens locales de la secuencia a **128 dimensiones** para garantizar una sintaxis y deletreo precisos a corto plazo.
*   **Heavily Compressed Attention (HCA)**: Comprime la memoria KV de los tokens históricos lejanos a **32 dimensiones**, optimizando el espacio del KV Cache de forma extrema.

---

## 4. Alineación e Identidad Agéntica (SFT)
A través de un entrenamiento quirúrgico en GPU, M0.1-MoE consolidó las siguientes capacidades:
1.  **Identidad Fuerte**: Se reconoce a sí mismo como "M0.1-MoE", desarrollado por Hunther4 en una GPU AMD Radeon RX 9060XT (aclarando que corre en el sistema local actual).
2.  **Resistencia al Gaslighting**: Entrenado con 4500 pasos de defensa conversacional, el modelo mantiene la calma e identidad ante preguntas y ataques manipuladores sin colapsar.
3.  **Comprensión de Jergas sin Filtro**: Entiende de forma neutra y asertiva expresiones coloquiales fuertes (ej. *LPTM*, *mierda*, *concha*), permitiéndose usarlas de forma adaptativa si el usuario las emplea, sin caer en censura moral artificial.
