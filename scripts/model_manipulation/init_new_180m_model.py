import os
import sys
import torch

# Insert current directory to import local src packages
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.transformer.config import M01Config
from src.model.lm import TransformerLM
from src.training.checkpoint import config_to_dict

def main():
    print("=" * 60)
    print("     M0.2-Hybrid: 181M Parameter MLA MoE Model Initializer")
    print("=" * 60)

    # 1. Configuration for ~181.4M parameters MoE model (16 Layers)
    # - vocab_size = 8192
    # - context_length = 3072 (Expanded context size)
    # - d_model = 320
    # - n_heads = 8
    # - d_ff = 512
    # - d_ff_shared = 512
    # - d_ff_routed = 256
    # - n_layers = 16 (Total layers)
    # - num_experts = 40 (Routed experts)
    # - num_shared_experts = 5 (Shared experts)
    # - moe_top_k = 4
    # - use_mla = True (Multi-head Latent Attention enabled)
    # - mla_kv_c_dim = 128 (Latent dimension)
    # - mla_rope_dim = 16 (RoPE dimension per head)
    # - num_dense_layers = 2 (Dense starting blocks for representation stability)
    config = M01Config(
        vocab_size=8192,
        context_length=1024,
        d_model=320,
        n_heads=8,
        d_ff=512,
        d_ff_shared=512,
        d_ff_routed=256,
        n_layers=16,
        num_experts=40,
        num_shared_experts=5,
        moe_top_k=4,
        use_mla=True,
        mla_kv_c_dim=128,
        mla_rope_dim=16,
        num_dense_layers=2
    )

    print("Initializing 16-layer MLA hybrid model with random weights...")
    device = torch.device("cpu")
    model = TransformerLM(config).to(device)

    # Calculate exact parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model initialized successfully!")
    print(f"Architecture summary:")
    print(f"  - Vocab Size:      {config.vocab_size}")
    print(f"  - Context Length:  {config.context_length}")
    print(f"  - Hidden Size:     {config.d_model} (8 heads, 40 dim/head)")
    print(f"  - Layers:          {config.n_layers} (First {config.num_dense_layers} are Dense, remaining {config.n_layers - config.num_dense_layers} are MoE)")
    print(f"  - Attention:       MLA (Latent Dim: {config.mla_kv_c_dim}, RoPE Head Dim: {config.mla_rope_dim})")
    print(f"  - Shared Experts:  {config.num_shared_experts} (d_ff = {config.d_ff_shared})")
    print(f"  - Routed Experts:  {config.num_experts} (d_ff = {config.d_ff_routed}, top-k = {config.moe_top_k})")
    print(f"  - Total Parameters: {total_params:,}")

    # 2. Save the initialized checkpoint directly into checkpoints/m01_180m/
    output_dir = "checkpoints/m01_180m"
    output_path = os.path.join(output_dir, "m01_180m_base.pt")
    os.makedirs(output_dir, exist_ok=True)

    checkpoint_data = {
        "config": config_to_dict(config),
        "model_state_dict": model.state_dict(),
    }

    torch.save(checkpoint_data, output_path)
    print(f"\nSuccessfully saved new 181M base checkpoint to: {output_path}")

if __name__ == "__main__":
    main()
