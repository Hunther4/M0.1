# Architecture Specifications: M0.1

## Overview
M0.1 is an autoregressive Decoder-only Transformer model designed for educational and optimization purposes.

## Global Parameters
* **Target Parameters**: ~80M (actual parameters: ~80.46M)
* **Context Length**: 8192
* **Vocabulary Size**: 32768
* **Layers**: 12
* **Heads**: 10
* **Head Dimension**: 64
* **Embedding Dimension ($d_{model}$)**: 640
* **Intermediate Dimension ($d_{ff}$)**: 1728
* **Weight Tying**: Enabled (Embedding and output head share weights)
* **Normalization**: RMSNorm
* **Positional Embeddings**: RoPE (Rotary Position Embedding)
* **Activation**: SwiGLU

## Parameter Calculation Breakdown
* **Embeddings (Tied)**:
  $$32768 \times 640 = 20,971,520 \text{ parameters}$$
* **Transformer Block (Per layer)**:
  * **RMSNorm (Input & Post-Attention)**: $2 \times 640 = 1,280$
  * **Attention Projections (Q, K, V, O)**: $4 \times 640 \times 640 = 1,638,400$
  * **SwiGLU MLP (Gate, Up, Down)**: $3 \times 640 \times 1728 = 3,317,760$
  * **Total per block**: $4,957,440$
* **12 Layers Total**:
  $$12 \times 4,957,440 = 59,489,280 \text{ parameters}$$
* **Final RMSNorm**: $640$
* **Grand Total**:
  $$20,971,520 + 59,489,280 + 640 = 80,461,440 \text{ parameters (~80.5M)}$$

## Implementation Status
* **Fase 0** (✅ Complete): BPE Tokenizer, Dataset Preparation, Token Counter CLI
* **Fase 1** (✅ Complete): Model Architecture — Config, Embeddings, RoPE, CausalSelfAttention, KV-Cache, FeedForward, MoE
* **Fase 2** (✅ Complete): Transformer Assembly — RMSNorm, TransformerBlock (pre-norm), TransformerLM (12-layer, 80.46M params), Training Pipeline — TrainingConfig, TinyShakespeareDataset, CheckpointManager (atomic save), AdamW optimizer, cosine LR schedule, gradient clipping, fp32 training loop. **172 tests passing.**

## Design Constraints
* **No HuggingFace Trainer**: Built from scratch using native PyTorch.
* **Modular Codebase**: Every module has single responsibility and small file sizes.
* **Byte-level BPE**: Avoids Out-of-Vocabulary (OOV) errors by initializing vocabulary with 256 base bytes.
