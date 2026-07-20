"""MoE scaling ladder calculator for M0.1."""
from src.transformer.config import M01Config
from src.model.lm import TransformerLM

fixed_base = 41.4e6  # embed + attn + 2 dense FFN
dense_act = 3 * 640 * 1728  # 3.32M

stages = [
    # (name, routed, shared, topk, dff_r, dff_s)
    ("E1: 4+1 tk1",  4,  1,  1, 640, 1024),
    ("E2: 8+2 tk2",  8,  2,  2, 448, 768),
    ("E3: 16+2 tk2", 16,  2,  2, 256, 640),
    ("E4: 32+4 tk4", 32,  4,  4, 128, 432),
    ("E5: 40+5 tk4", 40,  5,  4, 112, 384),
]

print("=" * 80)
print("ESCALERA MoE — M0.1 (10 capas MoE, 12 layers total, d_model=640)")
print("=" * 80)
print()
header = f"{'Etapa':<15} {'Routed':<8} {'Shared':<8} {'TopK':<5} {'dff_r':<7} {'dff_s':<7} {'Total':<9} {'Activos':<9} {'vsDense':<8} {'Mem_fp32':<9}"
print(header)
print("-" * 80)

for name, n_rt, n_sh, tk, dff_r, dff_s in stages:
    cfg = M01Config(num_experts=n_rt, num_shared_experts=n_sh, moe_top_k=tk,
                    d_ff_shared=dff_s, d_ff_routed=dff_r)
    model = TransformerLM(cfg)
    total = sum(p.numel() for p in model.parameters())
    act = n_sh * 3 * 640 * dff_s + tk * 3 * 640 * dff_r  # active per MoE layer
    # But also need first 2 dense layers: 3*640*1728 each
    act_total = 2 * dense_act + 10 * act  # total active across all layers
    ratio = act / dense_act if dense_act else 0
    mem = (total * 4 * 3) / (1024**3)

    print(f"{name:<15} {n_rt:<8} {n_sh:<8} {tk:<5} {dff_r:<7} {dff_s:<7} {total/1e6:<7.1f}M  {act/1e3:<6.0f}K  {ratio:<6.1f}x  {mem:<5.2f}G")

    # Per-layer breakdown
    gate = 10 * 640 * n_rt
    sh_p = 10 * n_sh * 3 * 640 * dff_s
    rt_p = 10 * n_rt * 3 * 640 * dff_r
    print(f"{'':>15}  gate={gate/1e6:.2f}M  shared={sh_p/1e6:.2f}M  routed={rt_p/1e6:.2f}M")
    print()

print(f"\nReferencia dense: {dense_act/1e3:.0f}K activos/capa | {3*dense_act/1e3:.0f}K activos totales (12 capas)")
print(f"Modelo base actual: 74.6M total (bug: 2 FFN por capa MoE)")
print()

# Check that each stage fits in 16GB
print("VRAM RX 9060 XT: ~16GB")
print("  fp32 total + Adam: ~2-3GB para todas las etapas")
print("  bf16: ~1-1.5GB")
print("  Con batch_size=4, seq_len=1024: ~2-4GB activaciones")
print("  TOTAL estimado: <8GB -> sobran 8GB+")
