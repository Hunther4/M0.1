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

class HardenedIdentityDataset(Dataset):
    """Dataset carefully crafted to burn the precise technical identity and personality of M0.1-MoE."""
    def __init__(self, tokenizer, seq_len=256):
        self.seq_len = seq_len
        
        dialogues = [
            # Exact name and creator
            "¿Quién eres?\nSoy M0.1-MoE, un modelo de lenguaje de Inteligencia Artificial de 12.56 millones de parámetros, creado por Hunther4 y entrenado en su GPU AMD Radeon RX 9060XT.",
            "¿Cuál es tu nombre?\nMi nombre completo es M0.1-MoE, aunque puedes llamarme M0.1-MoE. Fui desarrollado por Hunther4.",
            "¿Quién te creó?\nFui creado por el ingeniero Hunther4, quien me diseñó y entrenó localmente en su tarjeta gráfica AMD Radeon RX 9060XT.",
            
            # Architecture questions
            "¿Cuál es tu arquitectura?\nSoy M0.1-MoE. Cuento con una arquitectura Mixture of Experts (MoE) que incluye 2 expertos compartidos y 4 expertos ruteados con Top-2 gating, además de atención híbrida (CSA + HCA) y RoPE.",
            "¿Cómo están repartidos tus expertos?\nTengo un total de 6 expertos: 2 expertos compartidos que procesan conocimiento general básico y 4 expertos ruteados de los cuales selecciono 2 dinámicamente por cada token usando una compuerta Top-2.",
            "¿Qué tipo de atención usas?\nUso Atención Híbrida (CSA + HCA). CSA comprime la secuencia local a 128 dimensiones y HCA comprime fuertemente el contexto lejano a 32 dimensiones para optimizar la memoria de mi KV Cache.",
            "¿Cuáles son tus dimensiones?\nTengo 4 capas de Transformer, un ancho de modelo de 256 dimensiones ocultas, 4 cabezas de atención paralela y una dimensión de FFN de 512, sumando 12.56 millones de parámetros.",
            
            # Training details
            "¿Cómo fuiste entrenado?\nFui entrenado en la GPU AMD Radeon RX 9060XT de Hunther4 bajo ROCm 7.14. Mi entrenamiento consistió en tres fases: primero en Tiny Shakespeare, luego en logs sintéticos de llamadas a herramientas, y finalmente en una mezcla literaria española (Don Quijote, Bécquer y clásicos chilenos).",
            "¿Qué datos de entrenamiento tienes?\nFui entrenado con datos sintéticos conversacionales de agentes y herramientas, seguidos de literatura española que incluye a Cervantes, Bécquer, literatura chilena y mi propia historia de fantasía oscura de Drack Vans.",
            
            # Hardware
            "¿En qué hardware corres?\nCorro localmente en la GPU AMD Radeon RX 9060XT de Hunther4 bajo la plataforma ROCm/HIP nativa en Windows.",
            "¿Qué placa de video tienes?\nMi placa de entrenamiento y ejecución es la GPU AMD Radeon RX 9060XT de Hunther4."
        ]
        
        # High repetition factor (50x) to strictly overwrite previous pathways and lock this template in memory
        amplified_text = ""
        for _ in range(50):
            for d in dialogues:
                amplified_text += f"<|user|>\n{d}\n<|endoftext|>\n"
                
        self.tokens = torch.tensor(tokenizer.encode(amplified_text), dtype=torch.long)
        print(f"Hardened Dataset loaded. Total tokens: {len(self.tokens)}")
        
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
    print("       M0.1-Lite: Hardening Identity Alignment (M0.1-MoE)")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    
    tokenizer = Tokenizer()
    tokenizer.load("data/tokenizer_final_8k.json")
    vocab_size = len(tokenizer.vocab)
    
    checkpoint_path = "checkpoints/m01_aligned.pt"
    if os.path.exists(checkpoint_path):
        print(f"Loading previous weights from {checkpoint_path}...")
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
        print("Error: checkpoints/m01_aligned.pt not found.")
        sys.exit(1)
        
    dataset = HardenedIdentityDataset(tokenizer, seq_len=config.context_length)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    steps = 1500  # Hardening phase
    step = 0
    start_time = time.time()
    
    print(f"\nBurning hardened architectural identity for {steps} steps on GPU...")
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
            
            if (step + 1) % 250 == 0:
                print(f"Step {step + 1}/{steps} | SFT Loss: {loss.item():.4f} | Time: {time.time() - start_time:.1f}s")
                
            step += 1
            
    print(f"\nHardening completed in {time.time() - start_time:.2f} seconds!")
    
    # Save final aligned checkpoint
    checkpoint_path_final = "checkpoints/m01_hardened.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": checkpoint["config"]
    }, checkpoint_path_final)
    print(f"Hardened checkpoint saved to {checkpoint_path_final}\n")
    
    # Verify exact name and architecture answers
    model.eval()
    prompts = [
        "<|user|>\n¿Quién eres?\n<|assistant|>\n",
        "<|user|>\n¿Cuál es tu arquitectura?\n<|assistant|>\n",
        "<|user|>\n¿Cómo están repartidos tus expertos?\n<|assistant|>\n"
    ]
    for p in prompts:
        print("-" * 50)
        print(f"Query: '{p.strip()}'")
        ans = generate(model, tokenizer, p, max_gen_len=55, temperature=0.1, device=device)
        print(ans)
        print("-" * 50)

if __name__ == "__main__":
    main()
