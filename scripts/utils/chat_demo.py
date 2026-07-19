import time
import torch
import torch.nn as nn
from src.transformer.config import M01Config
from src.model.lm import TransformerLM
from src.tokenizer.bpe import Tokenizer
from src.inference.generate import generate
from src.training.dataset import TinyShakespeareDataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from src.training.config import TrainingConfig

import sys

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 60)
    print("          M0.1-Lite: DeepSeek MoE & Hybrid Attention Demo")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # 1. Initialize Tokenizer
    print("Loading tokenizer...")
    tokenizer = Tokenizer()
    tokenizer.load("data/tokenizer.json")
    vocab_size = len(tokenizer.vocab)
    print(f"Vocab size: {vocab_size}\n")
    
    # 2. Configure a Lite M0.1 model with V3/V4 DeepSeek-style options
    print("Configuring M0.1-Lite model...")
    config = M01Config(
        vocab_size=vocab_size,
        context_length=256,
        d_model=256,
        n_heads=4,
        d_ff=512,
        n_layers=4,
        num_experts=4,           # 4 routed experts
        num_shared_experts=2,    # 2 shared experts
        moe_top_k=2,             # Route to top-2
        use_hybrid_attention=True, # CSA + HCA active
        local_window_size=16
    )
    
    model = TransformerLM(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model instantiated. Total parameters: {total_params / 1e6:.2f}M")
    print(f"MoE: {config.num_experts} routed experts, {config.num_shared_experts} shared experts (top-{config.moe_top_k} routing)")
    print(f"Attention: Hybrid (CSA/HCA) active. Local window: {config.local_window_size}\n")
    
    # 3. Generate text before training (using random weights)
    prompt = "ROMEO:\nShall I speak more,"
    print("-" * 50)
    print(f"Generating before training with prompt: '{prompt}'")
    print("-" * 50)
    gen_before = generate(model, tokenizer, prompt, max_gen_len=40, temperature=0.8, device=device)
    print(gen_before)
    print("-" * 50)
    print()
    
    # 4. Set up quick training on Tiny Shakespeare
    print("Preparing dataset...")
    train_config = TrainingConfig(seq_len=config.context_length, data_dir="data")
    dataset = TinyShakespeareDataset(train_config)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    print("\nStarting micro-training run (50 steps)...")
    model.train()
    
    step = 0
    start_time = time.time()
    
    # Run training loop
    for epoch in range(1):
        for x, y in dataloader:
            if step >= 50:
                break
                
            x, y = x.to(device), y.to(device)
            
            # Forward
            logits = model(x)
            
            # Loss computation: flatten sequence
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (step + 1) % 10 == 0:
                print(f"Step {step + 1}/50 | Loss: {loss.item():.4f}")
                
            step += 1
            
    training_duration = time.time() - start_time
    print(f"Micro-training complete in {training_duration:.2f} seconds.\n")
    
    # 5. Generate text after training
    print("-" * 50)
    print(f"Generating after training with prompt: '{prompt}'")
    print("-" * 50)
    model.eval()
    gen_after = generate(model, tokenizer, prompt, max_gen_len=40, temperature=0.8, device=device)
    print(gen_after)
    print("-" * 50)
    print()
    
    print("Demo execution finished successfully!")

if __name__ == "__main__":
    main()
