import os
import sys
import time
import torch
import torch.nn as nn
from src.transformer.config import M01Config
from src.model.lm import TransformerLM
from src.tokenizer.bpe import Tokenizer
from src.inference.generate import generate
from src.training.dataset import TinyShakespeareDataset
from src.training.config import TrainingConfig
from torch.utils.data import DataLoader
from torch.optim import AdamW

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        
    print("=" * 60)
    print("          M0.1-Lite: Training Phase 1 of 3 (Shakespeare)")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 1. Tokenizer
    tokenizer = Tokenizer()
    tokenizer.load("data/tokenizer.json")
    vocab_size = len(tokenizer.vocab)
    
    # 2. Config
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
    print(f"Model initialized with {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters.")
    
    # 3. Dataset
    print("Loading dataset...")
    # Use the file in data/ or M0.2/data/
    data_dir = "data"
    if not os.path.exists(os.path.join(data_dir, "tinyshakespeare.txt")):
        data_dir = "D:/Proyectos/M0.2/data"
        
    train_config = TrainingConfig(seq_len=config.context_length, data_dir=data_dir)
    dataset = TinyShakespeareDataset(train_config)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # 4. Training loop (500 steps)
    model.train()
    steps = 500
    step = 0
    start_time = time.time()
    
    print(f"Training for {steps} steps on CPU...")
    
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
            
            if (step + 1) % 50 == 0:
                elapsed = time.time() - start_time
                print(f"Step {step + 1}/{steps} | Loss: {loss.item():.4f} | Time: {elapsed:.1f}s")
                
            step += 1
            
    # Save checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = "checkpoints/phase1.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {
            "vocab_size": config.vocab_size,
            "context_length": config.context_length,
            "d_model": config.d_model,
            "n_heads": config.n_heads,
            "d_ff": config.d_ff,
            "n_layers": config.n_layers,
            "num_experts": config.num_experts,
            "num_shared_experts": config.num_shared_experts,
            "moe_top_k": config.moe_top_k,
            "use_hybrid_attention": config.use_hybrid_attention,
            "local_window_size": config.local_window_size
        }
    }, checkpoint_path)
    print(f"\nPhase 1 checkpoint saved to {checkpoint_path}")
    
    # 5. Generate
    model.eval()
    prompt = "ROMEO:\nShall I speak more,"
    print("\n" + "-" * 50)
    print(f"Generating after Phase 1 with prompt: '{prompt}'")
    print("-" * 50)
    generated_text = generate(model, tokenizer, prompt, max_gen_len=50, temperature=0.7, device=device)
    print(generated_text)
    print("-" * 50)

if __name__ == "__main__":
    main()
