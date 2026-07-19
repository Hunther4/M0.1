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

def evaluate_val_loss(model, val_loader, device, criterion):
    model.eval()
    total_loss = 0.0
    steps = 0
    with torch.no_grad():
        for x, y in val_loader:
            if steps >= 30:
                break
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item()
            steps += 1
    model.train()
    return total_loss / max(steps, 1)

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        
    print("=" * 60)
    print("     M0.1-Lite: GPU Training in Spanish (5000 Steps, 8K Vocab)")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    
    # 1. Load Quijote and Split
    quijote_path = "data/quijote.txt"
    if not os.path.exists(quijote_path):
        print(f"Error: Dataset not found at {quijote_path}")
        sys.exit(1)
        
    text_corpus = open(quijote_path, encoding="utf-8").read()
    
    # Clean Project Gutenberg header/footer metadata roughly (optional, but keep it simple)
    # Let's split 90/10
    split_idx = int(len(text_corpus) * 0.90)
    train_text = text_corpus[:split_idx]
    val_text = text_corpus[split_idx:]
    
    # 2. Train Tokenizer on Spanish text
    print("\nTraining BPE Tokenizer (8K vocab) on Don Quijote...")
    tokenizer = Tokenizer()
    tokenizer.train(train_text, 8192, show_progress=False)
    tokenizer.save("data/tokenizer_quijote_8k.json")
    vocab_size = len(tokenizer.vocab)
    print(f"BPE Tokenizer trained. Vocab size: {vocab_size}")
    
    # 3. Save splits and tokenizer for TinyShakespeareDataset class compatibility
    os.makedirs("data/splits_quijote", exist_ok=True)
    os.makedirs("data/splits_quijote_val", exist_ok=True)
    with open("data/splits_quijote/tinyshakespeare.txt", "w", encoding="utf-8") as f:
        f.write(train_text)
    with open("data/splits_quijote_val/tinyshakespeare.txt", "w", encoding="utf-8") as f:
        f.write(val_text)
        
    tokenizer.save("data/splits_quijote/tokenizer.json")
    tokenizer.save("data/splits_quijote_val/tokenizer.json")
    
    # 4. Config & Initialize Model
    config = M01Config(
        vocab_size=vocab_size,
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
    model = TransformerLM(config).to(device)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters.")
    
    # 5. Datasets
    print("Loading datasets...")
    train_config = TrainingConfig(seq_len=config.context_length, data_dir="data/splits_quijote")
    train_dataset = TinyShakespeareDataset(train_config)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    
    val_config = TrainingConfig(seq_len=config.context_length, data_dir="data/splits_quijote_val")
    val_dataset = TinyShakespeareDataset(val_config)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    optimizer = AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # 6. Training (5000 steps)
    model.train()
    steps = 5000
    step = 0
    start_time = time.time()
    
    print(f"\nStarting GPU training for {steps} steps on Don Quijote...")
    
    done = False
    while not done:
        for x, y in train_loader:
            if step >= steps:
                done = True
                break
                
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (step + 1) % 500 == 0:
                val_loss = evaluate_val_loss(model, val_loader, device, criterion)
                val_ppl = torch.exp(torch.tensor(val_loss)).item()
                elapsed = time.time() - start_time
                steps_per_sec = (step + 1) / elapsed
                print(f"Step {step + 1}/{steps} | Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f} | Speed: {steps_per_sec:.1f} st/s | Time: {elapsed:.1f}s")
                
            step += 1
            
    print(f"\nGPU Training completed in {time.time() - start_time:.2f} seconds!")
    
    # Save checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = "checkpoints/quijote_8k.pt"
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
    print(f"Spanish checkpoint saved to {checkpoint_path}\n")
    
    # 7. Generation
    model.eval()
    prompt = "En un lugar de la Mancha, de cuyo nombre no quiero acordarme,"
    print("-" * 50)
    print(f"Generating prose in Spanish with prompt: '{prompt}'")
    print("-" * 50)
    story = generate(model, tokenizer, prompt, max_gen_len=50, temperature=0.6, device=device)
    print(story)
    print("-" * 50)

if __name__ == "__main__":
    main()
