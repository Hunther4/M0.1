# Execution Roadmap and Quantitative Targets

**Estado:** En ejecución, actualizado al 2026-07-24  
**Propósito:** Convertir la comparativa de modelos externos en un plan de ejecución por fases con metas medibles

---

## 1. Línea Base

Estado base conocido de M0.1:

- Parámetros: aproximadamente `99.7M`
- Longitud de contexto: `8,192`
- Atención: MLA por defecto, con atención híbrida disponible
- MoE: `4` expertos ruteados + `1` experto compartido
- Weight tying: habilitado
- Escalado de proyecciones residuales: habilitado
- Estado de pruebas: la suite completa fue reportada como aprobada previamente

Esto significa que debemos optimizar para:

1. Mejor utilidad por parámetro.
2. Mejor utilidad por token de contexto.
3. Mejor utilidad por watt / latencia.
4. Mejor tooling y mejor fidelidad de benchmark.

---

## 2. Ganancias Exactas y Calculables

Estas son mejoras que podemos calcular directamente desde el diseño actual.

| Palanca | Línea base | Nuevo estado | Mejora exacta |
|---|---|---|---|
| Weight tying en embeddings / output head | `lm_head` separado | Matriz de embeddings compartida | Ahorra `10,485,760` parámetros, cerca de `10.5%` de un modelo de `99.7M` |
| Cache MLA vs cache MHA estándar | `2 * d_model = 1,280` floats por token | `mla_kv_c_dim + n_heads * d_head_rope = 128 + 160 = 288` floats por token | `77.5%` menos footprint de KV-cache |
| Escalado de std en proyecciones residuales | `initializer_range = 0.02` | `0.02 / sqrt(2 * 12)` | `79.6%` menor std de inicialización para proyecciones residuales |
| Salto de contexto, si el objetivo pasa de `8K` a `32K` | `8,192` | `32,768` | `300%` de incremento |
| Salto de contexto, si el objetivo pasa de `8K` a `128K` | `8,192` | `131,072` | `1,500%` de incremento |
| Salto de contexto, si el objetivo pasa de `8K` a `1M` | `8,192` | `1,000,000` | `12,107%` de incremento |

Notas:

- Las filas de contexto no son una afirmación de preparación actual. Son referencias de escala para planificación.
- La fila de cache es la mejora concreta más importante que ya está presente en la base de código.

---

## 3. Progreso Real

Lo siguiente está verificado contra el código en `src/`.

### Implementado

| Componente | Archivo | Detalle |
|---|---|---|
| `PromptPrefixCache` (LRU) | `src/inference/prompt_cache.py` | Cache LRU de prefijos para MLA, híbrida y MHA |
| Prefix reuse | `src/inference/prompt_cache.py` | Por prefijo común, no solo prompt idéntico |
| Invalidación automática | `src/inference/prompt_cache.py` | Al cambiar pesos, dispositivo, dtype o configuración |
| Límites por entrada y bytes | `src/inference/prompt_cache.py` | Con telemetría de hits, misses, evictions, bytes y tokens |
| Cache benchmark harness | `src/eval/inference_benchmark.py` | Latencia p50/p95, throughput, memoria pico, comparación cached/uncached |
| Test de cache | `tests/test_inference_benchmark.py` | `test_benchmark_reports_cache_telemetry` |
| CLI de benchmark | `scripts/evaluation/benchmark_inference.py` | `compare_prompt_cache` expuesta |

### NO implementado (pendiente)

| Componente | Prioridad | Bloqueador |
|---|---|---|
| Activation checkpointing | P0 | El engine no lo expone aún |
| Context overflow checks | P0 | No hay límite duro de ventana en el forward |
| Provider abstraction (DeepSeek/Kimi/GLM) | Fase 1 plan | Esperando decisión de scope |
| Tool call replay y trazas | Fase 3 plan | Sin runtime agentic consumidor |
| Agentic benchmark tracks | Fase 3 plan | Sin harness de tareas agentic |
| MLA vs híbrida bajo harness | Fase 4 plan | Esperando Fase 1 y 2 completas |

---

## 4. Fases de Ejecución

### Fase 1: Medición y hardening del protocolo

**Estado:** Plan

Objetivo:

- Hacer medible la interfaz del modelo y del agente antes de tocar arquitectura otra vez.

Acciones:

- Estandarizar wrappers de provider para llamadas estilo DeepSeek, Kimi y GLM.
- Normalizar `thinking`, `reasoning_effort`, `tool_calls` y structured output.
- Agregar logging de benchmark para:
  - tokens de entrada,
  - tokens de salida,
  - latencia p50/p95,
  - uso de memoria,
  - éxito / fallo por tarea.

Impacto esperado:

- `0%` de mejora de calidad solo por instrumentación.
- `100%` de mejora en observabilidad para comparaciones futuras.

---

### Fase 2: Disciplina de contexto largo

**Estado:** Parcialmente implementado (ver sección 3)

Objetivo:

- Hacer que el contexto largo sea una preocupación de primer orden del engine.

Lo que ya existe (ver sección 3):

- Caché LRU de prefijos para MLA, atención híbrida y MHA estándar.
- Reutilización por prefijo común, no sólo por prompt idéntico.
- Invalidación automática al cambiar pesos, dispositivo, dtype o configuración.
- Límite explícito por cantidad de entradas y bytes, con telemetría de hits,
  misses, evictions, bytes y tokens reutilizados.
- Validación de equivalencia cached/uncached en las tres variantes de atención.
- Harness JSON con latencia p50/p95, throughput, memoria pico, hashes de salida
  y comparación directa cached/uncached.

Lo que falta:

- Agregar accounting de ventana de contexto y checks duros de overflow.
- Agregar tests de estrés para contexto largo.
- Agregar tests de regresión para profundidad de recuperación, reuse de prompts y overflow.

Impacto esperado:

- `0%` de mejora bruta de calidad por sí sola.
- `100%` de reducción en comportamientos de overflow silencioso.
- `>50%` de reducción en ambigüedad de fallos de contexto largo una vez que el harness exista.

---

### Fase 3: Flujo Agentic y herramientas

**Estado:** Plan

Objetivo:

- Hacer que el uso de herramientas sea lo suficientemente fiable como para compararlo con los workflows agentic de Kimi y GLM.

Acciones:

- Agregar replay de tool calls y trazas estructuradas.
- Agregar loops de leer / editar / planear / verificar.
- Agregar validación de JSON mode y de esquemas.
- Agregar un track de benchmark para edición de código y workflows de documentos.

Impacto esperado:

- `+5%` a `+15%` de uplift esperado en tareas agentic versus un baseline sin estructura, una vez que el harness esté estable.
- `100%` de outputs de herramientas deberían ser parseables por máquina cuando exista un esquema.

Dependencias: Fase 1 completada.

---

### Fase 4: Experimentos arquitectónicos

**Estado:** Plan

Objetivo:

- Comparar MLA, atención híbrida y MHA estándar bajo el mismo harness.

Acciones:

- Mantener MLA como baseline principal de investigación.
- Aislar experimentos de atención híbrida.
- Comparar uso de cache, calidad y latencia.
- Medir si la atención híbrida mejora la robustez de contexto largo más que MLA en esta base de código.

Impacto esperado:

- `0%` de uplift garantizado antes de testear.
- El punto de decisión debe ser empírico, no ideológico.

Dependencias: Fase 1 y Fase 2 completadas.

---

### Fase 5: Eficiencia y decisiones de escala

**Estado:** Plan

Objetivo:

- Decidir dónde realmente paga el esfuerzo adicional.

Acciones:

- Ejecutar los mismos benchmarks sobre:
  - M0.1 actual,
  - variante MLA,
  - variante de atención híbrida,
  - modelos open-weight de referencia donde aplique.
- Comparar:
  - calidad por parámetro,
  - calidad por token,
  - calidad por watt,
  - latencia por tarea,
  - estabilidad con contexto largo.

Impacto esperado:

- Mejorar la calidad de decisión, no solo la calidad del modelo.
- Reducir trabajo desperdiciado en caminos que se ven impresionantes pero no mueven métricas.

Dependencias: Fase 4 completada.

---

## 5. Métricas Sugeridas

### Métricas núcleo

- `pass@1` en tareas de código
- `tasa de éxito` en tareas agentic
- `exact match` en tareas de extracción
- `latencia p50/p95`
- `tokens in / tokens out`
- `peak memory`
- `context overflow rate`

### Métricas de eficiencia

- `score por millón de parámetros`
- `score por segundo`
- `score por watt` si hay telemetría de hardware
- `score por dólar` para modelos API

### Métricas de contexto largo

- accuracy de recuperación a distintas profundidades
- slope de degradación versus longitud de contexto
- cache hit rate
- ahorro por reuse de prompts

---

## 6. Metas Recomendadas

| Área | Estado actual | Estado objetivo | Meta |
|---|---|---:|---|
| Cobertura de benchmark | ad hoc / parcial | mínimo 4 tracks | `+100%` de expansión de cobertura frente a pruebas ad hoc |
| Robustez del protocolo de herramientas | mixta | validación por esquema | `0` outputs de herramientas sin parseo cuando exista esquema |
| Seguridad de contexto largo | overflow checks no implementados | overflow checks explícitos | `100%` de overflows rechazados antes de ejecutar |
| Eficiencia de cache | MLA ya comprimido | mantener o mejorar | preservar la ganancia de `77.5%` y validarla en decoding |
| Calidad en tareas agentic | baseline local | benchmark target | `+5%` a `+15%` de uplift esperado tras hardening del protocolo |
| Stretch de contexto | baseline `8K` | `32K` primero, luego `128K` | `+300%` y luego `+1,500%` sobre la línea base actual |

---

## 7. Qué No Hacer

- No usar comparaciones contra modelos de trillones como métrica principal de éxito.
- No saltar a `1M` de contexto antes de que el cache y el harness estén estables.
- No mezclar variantes de atención sin aislar la medición.
- No declarar mejoras de calidad sin evidencia de benchmark.
- No dejar que la compatibilidad de API reemplace la evaluación real del modelo.
- No tratar activation checkpointing como trivial — requiere validación cuidadosa en el engine.

---

## 8. Próximos Pasos Inmediatos

1. Ejecutar el benchmark de caché sobre el checkpoint y hardware objetivo.
2. Definir el primer dataset de benchmark para tareas de código y agente.
3. **Agregar context overflow checks** — el único gap concretos de Fase 2.
4. Agregar validación de tool calls cuando exista un runtime agentic consumidor.
5. Comparar MLA versus atención híbrida bajo condiciones idénticas.
6. Mantener Unsloth como experimento aislado: la integración directa no conviene
   mientras M0.1 siga siendo una arquitectura PyTorch propia no compatible con
   Hugging Face Transformers.
7. Consultar [Unsloth en AMD: adaptacion pragmatica para M0.1](./unsloth_amd_adaptation.md)
   como plan de medición y backlog técnico antes de tocar kernels o checkpointing.