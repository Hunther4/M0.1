# Benchmark Landscape and Optimization Strategy

**Estado:** Nota de investigación en curso, al 2026-07-22  
**Alcance:** DeepSeek-V4, Kimi K2.6 / K3, GLM-5.1 / 5.2 y referencias open-weight seleccionadas  
**Objetivo:** Convertir los hallazgos de modelos externos en un plan concreto de optimización para M0.1

---

## 1. Resumen Ejecutivo

La conclusión principal es simple: no hay un único modelo externo que convenga copiar.
Lo que sí podemos copiar son las ideas que realmente mejoran la utilidad del producto:

- DeepSeek-V4 muestra cómo combinar una API de contexto largo, modos de pensamiento, tool calls y serving eficiente alrededor de un diseño de atención híbrida.
- Kimi K2.6 y Kimi K3 muestran cómo productizar coding agentic, flujos multimodales y ejecución de largo horizonte.
- GLM-5.1 y GLM-5.2 muestran cómo hacer utilizable el trabajo agentic de largo horizonte mediante context caching, streaming tool calls, MCP e infraestructura de contexto de 1M.
- Qwen 3.6 35B A3B y Qwen 3.6 27B son referencias open-weight útiles para workflows de agente y tradeoffs entre un modelo compacto MoE y uno más denso.
- Gemma 4 es una referencia open-weight fuerte para razonamiento, soporte multimodal y eficiencia por parámetro.

Para M0.1, la estrategia correcta no es perseguir escala. La estrategia correcta es copiar primero la interfaz, el cache y la disciplina de evaluación, y después tomar ideas arquitectónicas de forma selectiva.

---

## 2. DeepSeek-V4

### Lo que dice el material oficial

- Página pública y transparencia:
  - `DeepSeek-V4` aparece en la página oficial de transparencia con fecha de release `2026-04-24`.
  - La API expone `deepseek-v4-pro` y `deepseek-v4-flash`.
- Compatibilidad de API:
  - La API es compatible con OpenAI y también soporta acceso estilo Anthropic.
  - La base URL oficial es `https://api.deepseek.com`.
- Contexto y salida:
  - Longitud de contexto: `1M`.
  - Máxima salida: `384K`.
- Razonamiento:
  - El thinking mode está soportado.
  - `reasoning_effort` se puede usar con `high` y `max`.
  - En thinking mode, `temperature`, `top_p`, `presence_penalty` y `frequency_penalty` se ignoran de forma efectiva.
- Tooling:
  - Tool calls están soportadas.
  - El modelo puede participar en loops de agente.
- Precios y límites operativos:
  - `deepseek-v4-pro` y `deepseek-v4-flash` tienen precios y límites de concurrencia distintos.
  - La documentación también describe context caching y aislamiento de rate limits.
- Arquitectura:
  - El model card oficial de V4 describe un modelo MoE con `Hybrid Attention` usando `CSA + HCA`, además de `mHC` y `Muon`.
  - El material oficial de V4 no presenta a V4 como una arquitectura centrada en MLA.

### Qué importa para M0.1

- Copiar:
  - la forma de la API y sus patrones de compatibilidad,
  - la semántica del thinking mode,
  - el flujo de tool calls,
  - el diseño de contexto largo con cache,
  - la disciplina de evaluación para contexto largo y tareas agentic.
- No copiar:
  - la escala del modelo,
  - la receta interna propietaria de entrenamiento,
  - la arquitectura exacta de V4 como si fuera un mapeo 1:1 al stack actual de M0.1.

### Conclusión práctica

Para M0.1, DeepSeek-V4 debe tratarse como referencia para:

1. Diseño de protocolo orientado al producto.
2. Serving de contexto largo.
3. Modos mixtos de razonamiento.
4. Diseño eficiente de atención.

No es un blueprint para copiar pesos ni para reproducir el mismo régimen de escala.

---

## 3. Familia Kimi

### Kimi K2.6

Lo que dicen los docs oficiales y el help center:

- `kimi-k2.6` es el modelo smart actual dentro de la plataforma Kimi.
- Soporta texto, imagen y video.
- Soporta modos thinking y non-thinking.
- Está orientado a:
  - coding agentic,
  - razonamiento de contexto largo,
  - ejecución de largo horizonte,
  - diseño de front-end y generación de producto.
- La documentación oficial indica que `kimi-k2.6` ofrece una ventana de contexto de `256K`.
- La API de Kimi es compatible con OpenAI y expone chat completions, parsing de archivos, búsqueda web y herramientas oficiales.
- La API también ofrece JSON mode y soporte de tool calls.

Por qué importa:

- Kimi K2.6 es la referencia más fuerte aquí para comportamiento de “modelo útil para código”.
- Es especialmente valioso como objetivo de benchmark para:
  - generación de código,
  - refactorización iterativa,
  - uso de herramientas en varios pasos,
  - flujos de generación de documentos y hojas.

### Kimi K3

Lo que dicen el blog oficial y el help center:

- Kimi K3 es el modelo Kimi más capaz al momento de su anuncio.
- Es un modelo de `2.8T` parámetros.
- Usa `Kimi Delta Attention (KDA)` y `Attention Residuals (AttnRes)`.
- Tiene visión nativa.
- Tiene una ventana de contexto de `1M` tokens.
- Está diseñado para coding de frontera, trabajo de conocimiento y razonamiento.
- El help center indica que K3 es el modelo más fuerte para chat y tareas agentic.
- El blog oficial indica que los pesos completos están planeados para el `2026-07-27`.

Por qué importa:

- Kimi K3 es la referencia más fuerte para:
  - ejecución de largo horizonte,
  - orquestación de agentes,
  - razonamiento multimodal,
  - workflows de producto de contexto amplio.
- Al `2026-07-22`, el blog de arquitectura ya está disponible, pero los pesos completos todavía no se han liberado.

### Qué copiar de Kimi

- La separación entre uso rápido tipo chat y uso profundo tipo agente.
- La idea de seleccionar el modelo por clase de tarea, no solo por tamaño bruto.
- Los controles nativos de intensidad de razonamiento.
- La entrada multimodal nativa como capacidad de producto.
- Mejor soporte para outputs orientados a documentos, slides y workflows.

### Qué no copiar

- No asumir que K2.6 y K3 tienen las mismas restricciones de serving.
- No asumir que el futuro release de pesos significa que los detalles de implementación ya estén estables.
- No tratar “más contexto” como suficiente por sí solo; Kimi también está mostrando que workflow y agent integration deben ser de primera clase.

---

## 4. GLM-5.1 y GLM-5.2

### GLM-5.1

La documentación oficial describe GLM-5.1 como un modelo insignia para tareas de largo horizonte:

- Longitud de contexto: `200K`.
- Máxima salida: `128K`.
- Thinking mode soportado.
- Streaming output soportado.
- Function calling soportado.
- Context caching soportado.
- Structured output soportado.
- Integración con MCP soportada.

El modelo está orientado a:

- ingeniería agentic,
- ejecución de tareas de largo alcance,
- workflows de código,
- trabajo autónomo sostenido.

### GLM-5.2

El release y la documentación oficial indican que GLM-5.2 agrega:

- `1M` de contexto.
- `128K` de salida máxima.
- Modos de pensamiento.
- Streaming tool calls con `tool_stream=true`.
- Context caching.
- Function calls.
- Structured output.
- Soporte para MCP.
- `IndexShare` para reducir overhead del indexer en sparse attention.
- mejoras de MTP para speculative decoding.

Por qué importa:

- GLM-5.2 es la referencia pública más clara aquí para hacer que `1M` de contexto sea útil en la práctica.
- Es especialmente relevante para el diseño del engine, no solo del modelo.
- Su documentación ayuda a entender cómo exponer contexto largo y comportamiento de herramientas sin obligar al usuario a gestionar cada detalle de bajo nivel.

### Qué copiar de GLM

- Parámetros de tool calls en streaming.
- Serving de contexto largo con cache awareness.
- Casos de uso explícitos para coding e ingeniería de largo horizonte.
- Compatibilidad con structured output y MCP.
- La idea de que una ventana de contexto grande solo sirve si el engine realmente puede mantenerla estable y rápida.

---

## 5. Otras Referencias Open-Weight

### Qwen 3.6 35B A3B

Los materiales oficiales de Qwen Code describen:

- `Qwen3.6-35B-A3B` como una variante cuantizada local con soporte de imagen y video.
- Aparece en notas de release de Qwen Code como un modelo relevante para workflows locales multimodales.
- Es una opción compacta estilo MoE, interesante para coding agentic y structured output.

Por qué importa:

- Es una referencia práctica open-weight para código + agente.
- Es especialmente útil cuando queremos algo más capaz que los baselines densos pequeños, pero mucho más barato que modelos frontier.

### Qwen 3.6 27B

La línea Qwen 3.6 de 27B es el hermano más denso que conviene comparar contra la variante MoE compacta.

Por qué importa:

- Da un segundo punto en la curva de calidad vs costo dentro de la misma familia.
- Sirve para decidir si un MoE compacto o un modelo más denso es mejor benchmark target para nuestro stack.

Nota:

- El material público oficial que encontré es más sólido para `Qwen3.6-35B-A3B`; conviene tratar `Qwen 3.6 27B` como objetivo de familia a verificar contra el naming oficial más reciente antes de automatizarlo.

### Gemma 4

El material oficial de Google sobre Gemma 4 indica:

- Gemma 4 es una familia open-weight con `E2B`, `E4B`, `12B`, `26B A4B` y `31B`.
- Soporta razonamiento, workflows agentic, function calling, structured JSON output, system instructions nativas e inputs multimodales.
- La familia está diseñada para correr en mobile, laptop, workstation y cloud.
- La documentación oficial enfatiza la licencia Apache 2.0 y un ecosistema de herramientas amplio.

Por qué importa:

- Gemma 4 es la referencia open-weight más fuerte aquí para eficiencia capacidad/parametro.
- Es un buen punto de comparación para ver si M0.1 está aprendiendo las lecciones correctas de eficiencia y no solo de escala bruta.

---

## 6. Comparativa Cruzada

| Modelo | Mejor para | Contexto | Razonamiento / Tools | Señal arquitectónica | Relevancia para M0.1 |
|---|---|---:|---|---|---|
| DeepSeek-V4 | razonamiento eficiente con contexto largo | `1M` | thinking mode, tool calls, APIs OpenAI/Anthropic | MoE + Hybrid Attention + mHC | muy alta para API y serving |
| Kimi K2.6 | coding agent usable y workflows multimodales | `256K` | thinking / non-thinking, tools, multimodal | productización orientada a agente | muy alta para UX de code agent |
| Kimi K3 | trabajo agentic frontera y contexto largo | `1M` | strong thinking, tools, multimodal | KDA + AttnRes + MoE sparsity | altísima para tareas de largo horizonte |
| GLM-5.1 | engineering agentic | `200K` | thinking, function calls, structured output, MCP | sparse attention + RL training | alta para protocolo de herramientas |
| GLM-5.2 | trabajo agentic con contexto de 1M estable | `1M` | thinking, `tool_stream`, MCP, structured output | IndexShare + MTP | altísima para diseño de engine |
| Qwen 3.6 35B A3B | código + workflows de agente con tradeoff compacto | varía por despliegue | soporte image/video, workflows agentic | familia compacta estilo MoE | alta para baselines de code agent |
| Qwen 3.6 27B | comparación densidad vs costo | varía por despliegue | uso familiar de código/agente | hermano más denso de la misma familia | alta para comparación familiar |
| Gemma 4 | razonamiento open-weight y eficiencia multimodal | `128K` / `256K` según tamaño | function calling, JSON, system prompts | familia open-weight con opciones densas y MoE | muy alta como referencia de eficiencia |

---

## 7. Qué Debemos Adoptar en M0.1

### A. Capa de API y protocolo

- Añadir una abstracción de provider que pueda hablar:
  - estilo DeepSeek OpenAI-compatible,
  - estilo Kimi OpenAI-compatible,
  - estilo GLM OpenAI-compatible.
- Normalizar:
  - `thinking`,
  - `reasoning_effort`,
  - `tool_calls`,
  - deltas de streaming,
  - structured output / JSON mode.

### B. Capa de contexto y cache

- Hacer que el soporte de contexto largo sea un problema del engine, no solo un flag de configuración.
- Priorizar:
  - prefix caching,
  - reuse de prompts,
  - disciplina de KV-cache,
  - políticas de truncado,
  - accounting de ventana de contexto.
- No saltar a `1M` de soporte antes de que cache, generación y evaluación estén estables.

### C. Capa de atención y arquitectura

- Mantener el trabajo actual de MLA porque es valioso.
- Agregar una ruta experimental separada para ideas de hybrid attention inspiradas en DeepSeek-V4.
- No mezclar KDA, MLA, CSA/HCA y sparse attention en un solo bloque; mantenerlos separados.

### D. Capa de agente y herramientas

- Modelar el uso de herramientas como un protocolo de primera clase.
- Agregar soporte para:
  - tool calling iterativo,
  - replay de resultados,
  - manejo de trazas de razonamiento,
  - outputs tipo tool-stream.

### E. Capa de benchmarks

- Usar una suite de benchmarks con tracks separados:
  - código,
  - contexto largo,
  - tareas agentic,
  - structured output,
  - multimodal si y cuando aplique.
- Comparar contra modelos externos con el mismo prompt, el mismo contexto, las mismas herramientas y las mismas reglas de parada.

---

## 8. Plan de Optimización Recomendado

### Fase 1: Hacer real la interfaz de modelos externos

- Implementar una capa de adaptadores de provider.
- Agregar perfiles para:
  - `deepseek-v4-pro`
  - `deepseek-v4-flash`
  - `kimi-k2.6`
  - `kimi-k3`
  - `glm-5.1`
  - `glm-5.2`
- Estandarizar el parsing de salidas de thinking y tool calls.

### Fase 2: Construir un harness de contexto largo

- Agregar tests para:
  - overflow de contexto,
  - reuse de prompts,
  - comportamiento de cache,
  - regresiones por inputs largos.
- Medir latencia p50/p95 y costo en tokens.

### Fase 3: Construir un benchmark agentic

- Agregar tareas que requieran:
  - edición de código,
  - lectura de archivos,
  - loops de plan / act / verify,
  - tool calls,
  - outputs estructurados.
- Incluir tareas inspiradas en los docs de Kimi y GLM.

### Fase 4: Aislar experimentos arquitectónicos

- Mantener MLA como baseline de investigación.
- Aislar los experimentos de hybrid attention.
- Comparar:
  - MLA,
  - hybrid attention,
  - MHA estándar.

### Fase 5: Decidir dónde escala el esfuerzo

- Si el objetivo es mejor utilidad de producto:
  - invertir en cache, tools y evaluación.
- Si el objetivo es investigación arquitectónica:
  - invertir en variantes de atención y memoria de contexto largo.
- Si el objetivo es calidad de código:
  - invertir en benchmarks de código y loops de agente antes de escalar parámetros.

---

## 9. Evaluación de Unsloth

> **Nota:** el análisis detallado de Unsloth para AMD/ROCm vive en
> [./unsloth_amd_adaptation.md](./unsloth_amd_adaptation.md).  
> Esta sección es solo un resumen ejecutivo.

**Decisión:** no añadir Unsloth como dependencia ni modificar el engine principal mientras M0.1 use una arquitectura PyTorch propia con MLA/MoE custom.

**Resumo:** Unsloth es útil para fine-tuning de modelos compatibles con Hugging Face. No es integrable directamente a M0.1 por su MLA custom y router MoE propio. Los claims de velocidad y VRAM no son extrapolables.

**Plan posterior:** portar un modelo reducido y aislado a Transformers, medir el mismo workload con y sin Unsloth, y adoptarlo solo si hay ganancia reproducible.

**No hacer:** reescribir M0.1 alrededor de Unsloth o asumir soporte nativo para MLA/MoE custom.

**Fuentes:**

- [Unsloth repository](https://github.com/unslothai/unsloth)
- [Unsloth documentation](https://unsloth.ai/docs)
- [Continued Pretraining](https://unsloth.ai/docs/basics/continued-pretraining)

---

## 10. Lista de No-Go

- No intentar copiar la escala de trillones de parámetros.
- No tratar notas de release de open-source como sustituto de evaluación controlada.
- No convertir `1M` de contexto en un objetivo de marketing antes de que sea una feature de ingeniería estable.
- No fusionar ideas arquitectónicas no relacionadas en un solo bloque de atención.
- No benchmarkear solo con prompts sintéticos o triviales.

---

## 11. Paquete de Fuentes

### DeepSeek

- [DeepSeek Transparency](https://www.deepseek.com/en/transparency/)
- [DeepSeek V4 Model Card PDF](https://fe-static.deepseek.com/chat/transparency/deepseek-V4-model-card-EN.pdf)
- [DeepSeek API Docs](https://api-docs.deepseek.com/)
- [DeepSeek Thinking Mode](https://api-docs.deepseek.com/guides/thinking_mode)
- [DeepSeek Tool Calls](https://api-docs.deepseek.com/guides/tool_calls/)
- [DeepSeek Models & Pricing](https://api-docs.deepseek.com/quick_start/pricing?article_id=article_1779470751466_8)

### Kimi

- [Kimi API overview](https://www.kimi.com/help/kimi-api/api-overview)
- [Kimi API model capabilities](https://www.kimi.com/help/kimi-api/api-model-capabilities)
- [Kimi API pricing](https://www.kimi.com/help/kimi-api/api-pricing)
- [Kimi K2.6 quickstart](https://platform.kimi.com/docs/guide/kimi-k2-6-quickstart)
- [Kimi API models](https://platform.kimi.com/docs/models)
- [Kimi API quickstart](https://platform.kimi.com/docs/api/quickstart)
- [Kimi K3 blog](https://www.kimi.com/ja-jp/blog/kimi-k3)
- [Kimi model selection help](https://www.kimi.com/help/others/model-mode-selection)

### GLM

- [GLM-5 overview](https://docs.z.ai/guides/llm/glm-5)
- [GLM-5.1 overview](https://docs.z.ai/guides/llm/glm-5.1)
- [GLM-5.2 docs](https://docs.bigmodel.cn/cn/guide/models/text/glm-5.2)
- [GLM-5.2 blog](https://z.ai/blog/glm-5.2)
- [GLM migration guide](https://docs.bigmodel.cn/cn/guide/start/migrate-to-glm-new)

### Open-weight reference models

- [Qwen3 blog](https://qwenlm.github.io/blog/qwen3/)
- [Qwen Code weekly update mentioning Qwen3.6-35B-A3B](https://qwenlm.github.io/qwen-code-docs/en/blog/updates/weekly-update-2026-05-21/)
- [Gemma 4 model overview](https://ai.google.dev/gemma/docs/core)
- [Gemma 4 launch blog](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/)
