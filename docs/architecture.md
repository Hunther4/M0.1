# Architecture Specifications: M0.2-Hybrid (Ultimate Version)

## Overview
`M0.2-Hybrid` is an advanced decoder-only Mixture of Experts (MoE) transformer model. It incorporates initial dense layers to stabilize early sequence representations, Multi-head Latent Attention (MLA) to reduce key-value cache memory footprints, and 45 MoE experts (5 shared, 40 routed with Top-4 gating) for specialized linguistic domain capacity.

## Global Parameters
* **Target Parameters**: ~181M (actual parameters: 181,464,384)
* **Context Length**: 3072
* **Vocabulary Size**: 8192
* **Layers**: 16 (First 2 layers are Dense FFN, remaining 14 layers are MoE)
* **Heads**: 8
* **Head Dimension**: 40
* **Embedding Dimension ($d_{model}$)**: 320
* **Attention Mechanism**: Multi-head Latent Attention (MLA)
  * **Latent KV compression dimension ($d_c$)**: 128
  * **RoPE dimension per head ($d_{head\_rope}$)**: 16
  * **Latent non-RoPE dimension per head ($d_{head\_no\_rope}$)**: 24
* **Weight Tying**: Enabled (Embedding and output head share weights)
* **Normalization**: RMSNorm (FP32 precision stable calculation)
* **Positional Embeddings**: RoPE (Rotary Position Embedding) computed dynamically
* **Activation**: SwiGLU (Gate, Up, Down projections)

## Mixture of Experts (MoE) Structure
* **Shared Experts**: 5 (always active for general syntax)
  * Shared FFN dimension: 512
* **Routed Experts**: 40 (Top-4 gated for domain specialization)
  * Routed FFN dimension: 256
* **Routing Loss**: Auxiliary Load Balancing Loss ($E \sum f_i P_i$) integrated with a coefficient of 0.1 to prevent gating collapse.

## Parameter Calculation Breakdown
* **Embeddings (Tied)**:
  $$8192 \times 320 = 2,621,440 \text{ parameters}$$
* **Transformer Block (Per layer)**:
  * **RMSNorm (Input & Post-Attention)**: $2 \times 320 = 640$
  * **MLA Attention Projections**:
    * Query projections ($W_q$, $W_{qr}$): $8 \times (24 + 16) \times 320 = 102,400$
    * KV down-projection ($W_{kv\_down}$): $320 \times 128 = 40,960$
    * KV up-projections ($W_{k\_up}$, $W_{v\_up}$): $128 \times (8 \times 24 + 8 \times 40) = 65,536$
    * RoPE Key projection ($W_{kr}$): $320 \times (8 \times 16) = 40,960$
    * Output projection ($W_o$): $320 \times 320 = 102,400$
    * Latent Norm: 128
    * Total Attention parameters: $351,924$
  * **MLP/MoE Layer**:
    * **Layers 0-1 (Dense Blocks)**:
      * Dense FFN (Gate, Up, Down): $3 \times 320 \times 512 = 491,520$
    * **Layers 2-15 (MoE Blocks)**:
      * Shared Experts (5): $5 \times (3 \times 320 \times 512) = 2,457,600$
      * Routed Experts (40): $40 \times (3 \times 320 \times 256) = 9,830,400$
      * Gating matrix ($W_g$): $320 \times 40 = 12,800$
      * Total MoE FFN parameters: $12,300,800$

## Implementation and Security Design
* **Checkpoint Layout**:
  * `m01_180m_base.pt`: Step 0 reference file (random initialization weights).
  * `m01_180m_latest.pt`: Current accumulated training state containing weights, optimizer state, scheduler state, and epoch counters.
  * `m01_180m_milestone_XXXXXX.pt`: Safety snapshots saved automatically every 5000 steps.
* **Continual Session Training**: Timed training runs using the `--duration-min` budget to prevent context bloat and local cluster failures.
