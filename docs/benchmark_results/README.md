# Benchmark: Prompt Cache on AMD RX 9060 XT

**Fecha:** 2026-07-24  
**Hardware:** AMD Radeon RX 9060 XT (ROCm 7.14)  
**Checkpoint:** `runs/run_corpus3/checkpoints/checkpoint.pt` (20,000 pasos, 99.7M parámetros)  
**Tokenizador:** `data/tokenizers/tokenizer.json`  
**Modo:** compare (cached vs uncached)  
**Max gen len:** 32 tokens  
**Repeticiones:** 3  

---

## Resultados

| Métrica | Sin cache | Con cache | Delta |
|---|---|---|---|
| tokens/s | 11.66 | 16.15 | **+38.5%** |
| Prefill p50 | 0.078s | 0.000007s | **-99.99%** |
| Decode p50 | 2.01s | 1.94s | -3.5% |
| Peak memory | 235 MB | 156 MB | **-33.6%** |
| Request p50 | 2.09s | 1.97s | **-5.7%** |

## Comparación directa

| Indicador | Valor |
|---|---|
| **Speedup** | **1.39x** |
| **Reducción de latencia** | **27.8%** |
| **Cache hit rate** | **77.8%** (7 hits / 9 requests) |
| **Tokens reutilizados** | 47 / 78 prompt tokens |
| **Outputs match** | ✅ (correctness preservada) |

## Notas

- El speedup principal viene del prefill: cuando hay cache hit, el prefill es ~10,000x más rápido.
- Decode time no mejora significativamente con cache (es compute-bound, no memory-bound).
- El pico de memoria cae 33.6% porque no se recalcula el KV del prefijo.
- Cache hit rate de 77.8% es alto para un benchmark corto con prompts muy distintos.
- Este benchmark confirma que el prompt cache ya funciona en AMD/ROCm.

## Siguiente paso

Ejecutar Corrida B del plan AMD: medir `torch.compile`, bf16 en ROCm, y backend de atención
contra el mismo checkpoint y seeds.