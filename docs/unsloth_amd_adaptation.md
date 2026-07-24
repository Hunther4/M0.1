# Unsloth en AMD: adaptacion pragmatica para M0.1

**Estado:** propuesta de benchmark — sin resultados ejecutados aún, 2026-07-24  
**Objetivo:** extraer mejoras reales de entrenamiento para AMD/ROCm sin reescribir M0.1 alrededor de Unsloth

---

## 1. Tesis

Unsloth no encaja hoy como reemplazo del engine principal de M0.1, porque el modelo usa
MLA, MoE custom y un loop propio en PyTorch. El valor util esta en adaptar ideas de
eficiencia y medir si producen ganancia local:

- menos uso de VRAM,
- mejor throughput por paso,
- secuencias mas largas con el mismo hardware,
- menor overhead del routing MoE,
- mejor estabilidad numerica en AMD.

La pregunta correcta no es "usar Unsloth completo?", sino:

1. que tecnica concreta mejora nuestro entrenamiento en ROCm,
2. cuanto mejora en medicion controlada,
3. cual es el costo de integrarla en M0.1.

---

## 2. Puntos de partida del stack actual

M0.1 ya tiene varias bases utiles:

- soporte ROCm documentado en el entrypoint de training,
- `bf16` en AMD como precision preferida,
- grad accumulation,
- EMA,
- checkpoint async con checksum,
- profiler granular,
- `torch.compile` opt-in,
- `attention_backend` con fallback.

Eso significa que las mejoras mas valiosas probablemente no esten en "robustez general",
sino en:

- activations,
- routing MoE,
- kernels de atencion,
- uso real de memoria.

---

## 3. Plan de benchmark AMD

### Hardware y condiciones

Corremos todo en el mismo equipo AMD/ROCm, con el mismo checkpoint, mismo dataset y
mismos seeds cuando sea posible.

### Metricas a registrar

- `tokens/s`
- `step time` promedio y p95
- `peak reserved VRAM`
- `loss final` a igual presupuesto de tokens
- `NaN/Inf rate`
- `router drop rate`
- `throughput por secuencia`

### Corridas minimas

#### Corrida A: baseline real

Config base actual de M0.1.

Propósito:

- fijar la linea base antes de tocar nada,
- verificar que los numeros de memoria y tiempo sean reproducibles.

#### Corrida B: eficiencia barata

Probar primero cambios de bajo riesgo:

- `bf16` en ROCm,
- `torch.compile` solo si el backend se comporta bien,
- ajustar `seq_len` y `grad_accum_steps` para comparar presupuesto total equivalente,
- verificar si el backend de atencion elegido rinde mejor en AMD.

Propósito:

- medir mejoras de throughput sin alterar la semantica del modelo.

#### Corrida C: ahorro estructural

Solo despues de tener baseline:

- activation checkpointing,
- reduccion de materializacion intermedia en MoE,
- batching o fusion de pasos por experto,
- caminos mas directos en routing y agregacion.

Propósito:

- medir si el ahorro de VRAM habilita un batch efectivo mayor o secuencia mas larga.

### Regla de decision

- Si una mejora no gana al menos una de estas tres cosas:
  - VRAM,
  - throughput,
  - estabilidad,
  entonces no se adopta.

---

## 4. Backlog tecnico priorizado

### P0

- `activation checkpointing` en bloques de transformer y, si es viable, en subpaths MoE.
- Medicion limpia de `bf16` vs cualquier alternativa.
- Validacion de que el profiler capture el costo real por paso con y sin cambios.

### P1

- Reducir overhead Python en routing MoE.
- Agrupar calculos de expertos para evitar loops costosos.
- Explorar fusion de kernels o rutas mas compactas para la agregacion final.

### P2

- Ajustes finos del dataloader si el cuello es input pipeline y no GPU.
- Afinar `torch.compile` solo si el backend AMD es estable en esta version concreta.
- Revisar si el scheduler y el warmup actual estan alineados con el nuevo batch efectivo.

### No prioritario

- Reescribir el engine alrededor de Unsloth.
- Adoptar `4-bit QLoRA` como camino principal para este MoE.
- Migrar el modelo completo a Transformers solo para "poder usar Unsloth".

---

## 5. Riesgos

- Una mejora de kernel puede depender demasiado de una combinacion especifica de
  PyTorch, ROCm y GPU.
- El costo de adaptar MLA custom puede superar el beneficio esperado si el objetivo
  es solo ahorrar tiempo de entrenamiento.
- Algunas mejoras de marketing de Unsloth no se trasladan a un modelo de 100M parametros
  con arquitectura propia.

---

## 6. Fuentes primarias

- [Unsloth documentation](https://unsloth.ai/docs)
- [Unsloth repository](https://github.com/unslothai/unsloth)
- [Requirements](https://unsloth.ai/docs/get-started/beginner-start-here/unsloth-requirements)
- [Continued Pretraining](https://unsloth.ai/docs/basics/continued-pretraining)
- [MoE docs](https://unsloth.ai/docs/new)

