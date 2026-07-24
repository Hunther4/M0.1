# Benchmark Corrida B: Precisión y Attention Backends (AMD RX 9060 XT)

**Fecha:** 2026-07-24  
**Hardware:** AMD Radeon RX 9060 XT (ROCm 7.14)  
**Checkpoint:** `runs/run_corpus3/checkpoints/checkpoint.pt` (20,000 pasos)  
**Tokenizador:** `data/tokenizers/tokenizer.json`  
**Prompts:** 3 prompts distintos, 3 repeticiones, max_gen_len=32  

---

## Resultados por Precisión

| Precisión | tokens/s | Peak Memory | Prefill p50 | Request p50 | vs FP32 speed |
|---|---|---|---|---|---|
| FP32 | 9.35 | 235.6 MB | 0.103s | 2.58s | baseline |
| **BF16** | **14.06** | **78.8 MB** | **0.091s** | **2.20s** | **+50.3%** |
| FP16 | 13.78 | 79.0 MB | 0.084s | 2.13s | +47.4% |

## Notas

- BF16 y FP16 reducen la memoria ~66% (235MB → 79MB)
- BF16 es marginalmente más rápido que FP16 con mejor estabilidad numérica en AMD/ROCm
- Prefill p95 mejora dramáticamente con BF16: de 4.87s a 0.63s (7.8x más rápido en tail)
- Request p95 cae de 7.46s a 2.78s con BF16

## Comparación con Corrida A (Prompt Cache)

| Config | tokens/s | Peak Memory |
|---|---|---|
| FP32 uncached | 11.66 | 235 MB |
| FP32 cached | 16.15 | 156 MB |
| **BF16 uncached** | **14.06** | **79 MB** |

**Conclusión:** Prompt cache y BF16 son ortogonales y complementarios:
- Cache: ayuda cuando hay prompts compartidos (prefill ~0)
- BF16: reduce memoria 66% y mejora throughput general 50%

## Próximo paso

Probar BF16 + prompt cache combinados, y validar loss/NaN rate en training.