import os
import sys
import json
import time
import torch
import torch.nn as nn
from src.transformer.config import M01Config
from src.model.lm import TransformerLM
from src.tokenizer.bpe import Tokenizer
from src.inference.generate import generate
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW

class MultiShardJSONLDataset(Dataset):
    def __init__(self, shards_paths, tokenizer, seq_len=256, max_lines_per_shard=1000):
        self.seq_len = seq_len
        all_tokens = []
        for path in shards_paths:
            if not os.path.exists(path):
                continue
            with open(path, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    if idx >= max_lines_per_shard:
                        break
                    try:
                        data = json.loads(line)
                        text = f"{data['system']}\n{data['conversation']}"
                        tokens = tokenizer.encode(text)
                        all_tokens.extend(tokens)
                        all_tokens.append(256)
                    except Exception:
                        continue
        self.tokens = torch.tensor(all_tokens, dtype=torch.long)
        
    def __len__(self):
        if len(self.tokens) <= self.seq_len:
            return 0
        return (len(self.tokens) - 1) // self.seq_len
        
    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len
        return self.tokens[start:end], self.tokens[start+1:end+1]

def evaluate_val_loss(model, val_loader, device, criterion):
    model.eval()
    total_loss = 0.0
    steps = 0
    with torch.no_grad():
        for x, y in val_loader:
            if steps >= 30: # Check 30 batches to keep it fast
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
    print("     M0.1-Lite: GPU Training & Generalization Audit (5800 Steps)")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    
    tokenizer = Tokenizer()
    tokenizer.load("data/tokenizer.json")
    
    checkpoint_path = "checkpoints/phase3.pt"
    if os.path.exists(checkpoint_path):
        print(f"Loading weights from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        ckpt_config = checkpoint["config"]
        config = M01Config(
            vocab_size=ckpt_config["vocab_size"],
            context_length=ckpt_config["context_length"],
            d_model=ckpt_config["d_model"],
            n_heads=ckpt_config["n_heads"],
            d_ff=ckpt_config["d_ff"],
            n_layers=ckpt_config["n_layers"],
            num_experts=ckpt_config["num_experts"],
            num_shared_experts=ckpt_config["num_shared_experts"],
            moe_top_k=ckpt_config["moe_top_k"],
            use_hybrid_attention=ckpt_config["use_hybrid_attention"],
            local_window_size=ckpt_config["local_window_size"]
        )
        model = TransformerLM(config)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print("Error: checkpoints/phase3.pt not found.")
        sys.exit(1)
        
    model = model.to(device)
    
    # Load 10 training shards (0 to 9) to improve data generalizability
    shards_dir = "D:/Proyectos/M0.2/data/corpus/synthetic"
    train_shards = [os.path.join(shards_dir, f"shard_{i:04d}.jsonl") for i in range(10)]
    train_dataset = MultiShardJSONLDataset(train_shards, tokenizer, seq_len=config.context_length, max_lines_per_shard=800)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    
    # Load 1 unseen validation shard (10) to verify generalization (Val Loss)
    val_shard = [os.path.join(shards_dir, "shard_0010.jsonl")]
    val_dataset = MultiShardJSONLDataset(val_shard, tokenizer, seq_len=config.context_length, max_lines_per_shard=300)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    
    optimizer = AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    steps = 5800
    step = 0
    start_time = time.time()
    
    print(f"\nStarting training on GPU for {steps} steps...")
    
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
                elapsed = time.time() - start_time
                steps_per_sec = (step + 1) / elapsed
                print(f"Step {step + 1}/{steps} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | Speed: {steps_per_sec:.1f} st/s | Time: {elapsed:.1f}s")
                
            step += 1
            
    total_time = time.time() - start_time
    print(f"\nGPU Training completed in {total_time:.2f} seconds!")
    
    checkpoint_path_final = "checkpoints/phase3_final.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": checkpoint["config"]
    }, checkpoint_path_final)
    print(f"Final checkpoint saved to {checkpoint_path_final}")
    
    model.eval()
    prompt = "<|user|>\n¿Qué hay en este proyecto?\n<|tool_call|>"
    print("\n" + "-" * 50)
    print(f"Generating after Phase 3 Final with prompt: '{prompt}'")
    print("-" * 50)
    generated_text = generate(model, tokenizer, prompt, max_gen_len=60, temperature=0.5, device=device)
    print(generated_text)
    print("-" * 50)

if __name__ == "__main__":
    main()
