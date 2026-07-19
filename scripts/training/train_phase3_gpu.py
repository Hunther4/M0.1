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
from src.training.config import TrainingConfig
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW

class MultiShardJSONLDataset(Dataset):
    """Dataset that reads conversations from multiple JSONL files (shards) dynamically."""
    def __init__(self, shards_paths, tokenizer, seq_len=256, max_lines_per_shard=1000):
        self.seq_len = seq_len
        
        all_tokens = []
        for path in shards_paths:
            if not os.path.exists(path):
                print(f"Warning: Shard {path} not found. Skipping.")
                continue
                
            print(f"Tokenizing JSONL data from {path}...")
            with open(path, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    if idx >= max_lines_per_shard:
                        break
                    try:
                        data = json.loads(line)
                        text = f"{data['system']}\n{data['conversation']}"
                        tokens = tokenizer.encode(text)
                        all_tokens.extend(tokens)
                        all_tokens.append(256) # End of text token
                    except Exception:
                        continue
                        
        self.tokens = torch.tensor(all_tokens, dtype=torch.long)
        print(f"Dataset initialization complete. Total tokens in memory: {len(self.tokens)}")
        
    def __len__(self):
        if len(self.tokens) <= self.seq_len:
            return 0
        return (len(self.tokens) - 1) // self.seq_len
        
    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len
        x = self.tokens[start:end]
        y = self.tokens[start+1:end+1]
        return x, y

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        
    print("=" * 60)
    print("       M0.1-Lite: GPU Training Phase 3 of 3 (7500 Steps)")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        
    # 1. Tokenizer
    tokenizer = Tokenizer()
    tokenizer.load("data/tokenizer.json")
    vocab_size = len(tokenizer.vocab)
    
    # 2. Load Phase 2 checkpoint
    checkpoint_path = "checkpoints/phase2.pt"
    if os.path.exists(checkpoint_path):
        print(f"Loading Phase 2 weights from {checkpoint_path}...")
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
        print("Warning: Phase 2 checkpoint not found. Fallback to phase1.")
        # Fallback check
        checkpoint_path_p1 = "checkpoints/phase1.pt"
        if os.path.exists(checkpoint_path_p1):
            checkpoint = torch.load(checkpoint_path_p1, map_location="cpu")
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
            print("Error: No checkpoints found to resume training.")
            sys.exit(1)
            
    model = model.to(device)
    
    # 3. Load Shards 0000 to 0004 for more data coverage
    shards_dir = "D:/Proyectos/M0.2/data/corpus/synthetic"
    shards_paths = [os.path.join(shards_dir, f"shard_{i:04d}.jsonl") for i in range(5)]
    
    dataset = MultiShardJSONLDataset(shards_paths, tokenizer, seq_len=config.context_length, max_lines_per_shard=1000)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # 4. Training (7500 steps)
    model.train()
    steps = 7500
    step = 0
    start_time = time.time()
    
    print(f"\nStarting GPU training for {steps} steps (Phase 3)...")
    
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
            
            # Print every 250 steps
            if (step + 1) % 250 == 0:
                elapsed = time.time() - start_time
                steps_per_sec = (step + 1) / elapsed
                print(f"Step {step + 1}/{steps} | Loss: {loss.item():.4f} | Speed: {steps_per_sec:.1f} steps/s | Time: {elapsed:.1f}s")
                
                # Intermediate save
                os.makedirs("checkpoints", exist_ok=True)
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "config": checkpoint["config"]
                }, "checkpoints/phase3_latest.pt")
                
            step += 1
            
    total_time = time.time() - start_time
    print(f"\nPhase 3 training complete in {total_time:.2f} seconds!")
    
    # Save final Phase 3 checkpoint
    checkpoint_path_p3 = "checkpoints/phase3.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": checkpoint["config"]
    }, checkpoint_path_p3)
    print(f"Phase 3 checkpoint saved to {checkpoint_path_p3}")
    
    # 5. Generate validation
    model.eval()
    prompt = "<|user|>\n¿Qué hay en este proyecto?\n<|tool_call|>"
    print("\n" + "-" * 50)
    print(f"Generating after Phase 3 with prompt: '{prompt}'")
    print("-" * 50)
    generated_text = generate(model, tokenizer, prompt, max_gen_len=60, temperature=0.5, device=device)
    print(generated_text)
    print("-" * 50)

if __name__ == "__main__":
    main()
