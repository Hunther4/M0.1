import os
import sys
import time
import torch
import torch.nn as nn
from src.transformer.config import M01Config
from src.model.lm import TransformerLM
from src.tokenizer.bpe import Tokenizer
from src.inference.generate import generate
from src.training.datasets import JsonlDataset
from src.training.checkpoint import save_checkpoint
from src.training.setup import setup_device, setup_stdout
from torch.utils.data import DataLoader
from torch.optim import AdamW

def main():
    setup_stdout()
        
    print("=" * 60)
    print("       M0.1-Lite: GPU Training Phase 2 of 3 (Synthetic MoE/Attention)")
    print("=" * 60)
    
    # Use GPU (AMD Radeon RX 9060 XT via ROCm)
    device = setup_device()
        
    # 1. Tokenizer
    tokenizer = Tokenizer()
    tokenizer.load("data/tokenizer.json")
    
    # 2. Config & Load Phase 1 checkpoint
    checkpoint_path = "checkpoints/phase1.pt"
    if os.path.exists(checkpoint_path):
        from src.training.checkpoint import load_checkpoint
        print(f"Loading Phase 1 weights from {checkpoint_path}...")
        model, config = load_checkpoint(checkpoint_path, device="cpu")
    else:
        print("Warning: Phase 1 checkpoint not found. Training from scratch.")
        config = M01Config(
            vocab_size=32768,
            context_length=256,
            d_model=256,
            n_heads=4,
            d_ff=512,
            n_layers=4,
            num_experts=4,
            num_shared_experts=2,
            moe_top_k=2,
            use_hybrid_attention=True,
            local_window_size=16
        )
        model = TransformerLM(config)
        
    model = model.to(device)
    
    # 3. Load Synthetic JSONL Dataset from D:/Proyectos/M0.2/data/corpus/synthetic
    jsonl_path = "D:/Proyectos/M0.2/data/corpus/synthetic/shard_0000.jsonl"
    if not os.path.exists(jsonl_path):
        print(f"Error: Dataset not found at {jsonl_path}")
        sys.exit(1)
        
    dataset = JsonlDataset(tokenizer, [jsonl_path], seq_len=config.context_length, max_lines_per_shard=1500)
    if len(dataset) == 0:
        print("Error: Empty dataset.")
        sys.exit(1)
        
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Configure optimizer & loss
    optimizer = AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # 4. Training (up to 2000 steps or loss < 0.0290)
    model.train()
    steps = 2000
    target_loss = 0.0290
    step = 0
    start_time = time.time()
    
    print(f"\nTraining on GPU. Max steps: {steps}, Target loss: {target_loss}...")
    
    done = False
    while not done:
        for x, y in dataloader:
            if step >= steps:
                done = True
                break
                
            x, y = x.to(device), y.to(device)
            
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if loss.item() < target_loss:
                print(f"Step {step + 1} | Target loss achieved! Loss: {loss.item():.4f} | Time: {time.time() - start_time:.1f}s")
                done = True
                break
            
            if (step + 1) % 100 == 0:
                elapsed = time.time() - start_time
                steps_per_sec = (step + 1) / elapsed
                print(f"Step {step + 1}/{steps} | Loss: {loss.item():.4f} | Speed: {steps_per_sec:.1f} steps/s | Time: {elapsed:.1f}s")
                
            step += 1
            
    total_time = time.time() - start_time
    print(f"\nGPU Training completed in {total_time:.2f} seconds!")
    
    # Save checkpoint Phase 2
    checkpoint_path_p2 = "checkpoints/phase2.pt"
    save_checkpoint(model, config, checkpoint_path_p2)
    print(f"Phase 2 checkpoint saved to {checkpoint_path_p2}")
    
    # 5. Generation verification
    model.eval()
    prompt = "<|user|>\n¿Qué hay en este proyecto?\n<|tool_call|>"
    print("\n" + "-" * 50)
    print(f"Generating after Phase 2 with prompt: '{prompt}'")
    print("-" * 50)
    generated_text = generate(model, tokenizer, prompt, max_gen_len=60, temperature=0.6, device=device)
    print(generated_text)
    print("-" * 50)

if __name__ == "__main__":
    main()
