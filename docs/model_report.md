# M0.1 — Architecture Report

**Status:** This directory holds M0.1 project documentation only.

The former `model_report.md` described **M0.2-Hybrid** (181M params, d_model=320, 45 experts, vocab 8192) — a separate project. It has been moved to `Local_temporal/archived_model_report_M0.2-Hybrid.md` for reference.

---

## M0.1 Actual Configuration

See `src/transformer/config.py` for the authoritative source.

| Parameter | Value |
|---|---|
| Parameters | ~99.7M |
| d_model | 640 |
| n_layers | 12 |
| n_heads | 10 |
| d_head | 64 |
| vocab_size | **16384** (single tokenizer: `data/tokenizers/tokenizer.json`) |
| num_experts | 4 (routed) + 1 (shared), top-2 routing |
| d_ff | 1728 |
| context_length | 8192 |
| Attention | MLA (Multi-head Latent Attention) with RoPE |
| Activation | SwiGLU |
| Precision | BF16 AMP |

**Hardware:** AMD Radeon RX 9060 XT 16GB (ROCm 7.14)
