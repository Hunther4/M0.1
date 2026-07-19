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

class JSONLDataset(Dataset):
    """Dataset that reads conversations from JSONL files."""
    def __init__(self, file_path, tokenizer, seq_len=256, max_lines=1500):
        self.seq_len = seq_len
        
        print(f"Tokenizing JSONL data from {file_path}...")
        all_tokens = []
        with open(file_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx >= max_lines:
                    break
                try:
                    data = json.loads(line)
                    # Combine system prompt and conversation
                    text = f"{data['system']}\n{data['conversation']}"
                    tokens = tokenizer.encode(text)
                    all_tokens.extend(tokens)
                    # Add end of text token
                    all_tokens.append(256)
                except Exception as e:
                    continue
                    
        self.tokens = torch.tensor(all_tokens, dtype=torch.long)
        print(f"Loaded {idx+1} lines. Total tokens: {len(self.tokens)}")
        
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
    print("       M0.1-Lite: GPU Training Phase 2 of 3 (Synthetic MoE/Attention)")
    print("=" * 60)
    
    # Use GPU (AMD Radeon RX 9060 XT via ROCm)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        
    # 1. Tokenizer
    tokenizer = Tokenizer()
    tokenizer.load("data/tokenizer.json")
    vocab_size = len(tokenizer.vocab)
    
    # 2. Config & Load Phase 1 checkpoint
    checkpoint_path = "checkpoints/phase1.pt"
    if os.path.exists(checkpoint_path):
        print(f"Loading Phase 1 weights from {checkpoint_path}...")
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
        print("Warning: Phase 1 checkpoint not found. Training from scratch.")
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
        model = TransformerLM(config)
        
    model = model.to(device)
    
    # 3. Load Synthetic JSONL Dataset from D:/Proyectos/M0.2/data/corpus/synthetic
    jsonl_path = "D:/Proyectos/M0.2/data/corpus/synthetic/shard_0000.jsonl"
    if not os.path.exists(jsonl_path):
        print(f"Error: Dataset not found at {jsonl_path}")
        sys.exit(1)
        
    dataset = JSONLDataset(jsonl_path, tokenizer, seq_len=config.context_length, max_lines=1500)
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
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path_p2 = "checkpoints/phase2.pt"
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
    }, checkpoint_path_p2)
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
