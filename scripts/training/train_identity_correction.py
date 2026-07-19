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

class CorrectedIdentityDataset(Dataset):
    """Dataset to correct the execution context of M0.1-MoE (created on RX 9060XT, runs on current GPU)."""
    def __init__(self, tokenizer, seq_len=256):
        self.seq_len = seq_len
        
        dialogues = [
            "¿En qué hardware corres?\nSoy el modelo M0.1-MoE. Fui creado y entrenado por Hunther4 en su GPU AMD Radeon RX 9060XT, pero actualmente me ejecuto en el hardware que esté disponible localmente en este sistema.",
            "¿Qué placa de video tienes?\nMi entrenamiento se realizó en la GPU AMD Radeon RX 9060XT de Hunther4. Mi ejecución actual depende del sistema en el que me ejecutes.",
            "¿Corres en una RX 9060XT?\nFui creado y entrenado en una GPU AMD Radeon RX 9060XT por Hunther4, pero hoy en día me estoy ejecutando en el hardware del sistema actual, no necesariamente en esa misma placa física.",
            "¿Dónde te entrenaron?\nFui entrenado por Hunther4 en su máquina local equipada con una GPU AMD Radeon RX 9060XT utilizando ROCm."
        ]
        
        amplified_text = ""
        for _ in range(40):
            for d in dialogues:
                amplified_text += f"<|user|>\n{d}\n<|endoftext|>\n"
                
        self.tokens = torch.tensor(tokenizer.encode(amplified_text), dtype=torch.long)
        print(f"Corrected identity dataset loaded. Total tokens: {len(self.tokens)}")
        
    def __len__(self):
        if len(self.tokens) <= self.seq_len:
            return 0
        return (len(self.tokens) - 1) // self.seq_len
        
    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len
        return self.tokens[start:end], self.tokens[start+1:end+1]

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        
    print("=" * 60)
    print("      M0.1-Lite: Correcting Execution Hardware Scope (SFT)")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = Tokenizer()
    tokenizer.load("data/tokenizer_final_8k.json")
    vocab_size = len(tokenizer.vocab)
    
    checkpoint_path = "checkpoints/m01_uncensored.pt"
    if os.path.exists(checkpoint_path):
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
        model = TransformerLM(config).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print("Error: checkpoints/m01_uncensored.pt not found.")
        sys.exit(1)
        
    dataset = CorrectedIdentityDataset(tokenizer, seq_len=config.context_length)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    steps = 800  # Quick SFT correction step
    step = 0
    while step < steps:
        for x, y in dataloader:
            if step >= steps:
                break
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            
    # Save final corrected checkpoint
    checkpoint_path_corrected = "checkpoints/m01_corrected.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": checkpoint["config"]
    }, checkpoint_path_corrected)
    print(f"Corrected checkpoint saved to {checkpoint_path_corrected}\n")

if __name__ == "__main__":
    main()
