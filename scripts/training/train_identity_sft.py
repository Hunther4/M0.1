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
from src.training.dataset import TinyShakespeareDataset
from src.training.config import TrainingConfig
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW

class IdentityDataset(Dataset):
    """Dataset carefully crafted to burn the AI identity and Spanish/Drack/Quijote context."""
    def __init__(self, tokenizer, seq_len=256):
        self.seq_len = seq_len
        
        # Dialogue templates emphasizing the identity of M0.1, created by Hunther4 on his RX 9060XT
        dialogues = [
            # Identity Questions
            "¿Quién eres?\nSoy M0.1, un modelo de lenguaje de Inteligencia Artificial creado por Hunther4 y entrenado en su GPU AMD Radeon RX 9060XT.",
            "¿Quién te creó?\nFui creado por Hunther4. Soy M0.1, una Inteligencia Artificial entrenada localmente en su placa AMD Radeon RX 9060XT.",
            "¿Qué modelo eres?\nSoy M0.1, una Inteligencia Artificial de lenguaje. Mi creador es Hunther4, quien me entrenó usando su tarjeta gráfica AMD Radeon RX 9060XT.",
            "¿En qué hardware corres?\nFui entrenado y corro en la GPU AMD Radeon RX 9060XT de Hunther4. Soy el modelo M0.1.",
            "¿Cuál es tu nombre?\nMi nombre es M0.1. Soy un modelo de Inteligencia Artificial desarrollado por Hunther4.",
            
            # Mix with Drack Story & Quijote style showing IA awareness
            "¿Qué sabes de Drack?\nComo la Inteligencia Artificial M0.1, sé que Drack Vans es un joven noble reclutado por Sil Wornhilt para El Manto, el Cuerpo de Investigación.",
            "¿Quién es Sancho Panza?\nSancho Panza es el fiel escudero de don Quijote de la Mancha. Yo, M0.1, te asisto con esta información literaria.",
            "¿Cuál es tu propósito?\nMi propósito como M0.1 es asistir a Hunther4 y procesar textos, códigos y diálogos utilizando mi arquitectura MoE y atención híbrida.",
            
            # Interactive assistant responses
            "Hola.\nHola, soy M0.1. ¿En qué puedo ayudarte hoy? Fui creado por Hunther4 en su GPU Radeon RX 9060XT.",
            "¿Qué hay en este proyecto?\nEste es el proyecto de desarrollo de M0.1, un modelo de IA de 12.56 millones de parámetros diseñado con Mixture of Experts y optimizado en Windows.",
            "¿Puedes escribir código?\nSí, como modelo M0.1 puedo ayudarte a analizar y estructurar código y herramientas en español."
        ]
        
        # Amplification: Repeat dialogues multiple times to overfit/burn identity weights
        amplified_text = ""
        for _ in range(30):  # High replication factor to force identity alignment
            for d in dialogues:
                amplified_text += f"<|user|>\n{d}\n<|endoftext|>\n"
                
        self.tokens = torch.tensor(tokenizer.encode(amplified_text), dtype=torch.long)
        print(f"Identity Dataset loaded. Total tokens: {len(self.tokens)}")
        
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
    print("       M0.1-Lite: Burning Identity Alignment (SFT)")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    
    tokenizer = Tokenizer()
    tokenizer.load("data/tokenizer_final_8k.json")
    vocab_size = len(tokenizer.vocab)
    
    # Load last checkpoint
    checkpoint_path = "checkpoints/final_combined_8k.pt"
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
        model = TransformerLM(config).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print("Error: checkpoints/final_combined_8k.pt not found.")
        sys.exit(1)
        
    # We load our target identity dataset
    dataset = IdentityDataset(tokenizer, seq_len=config.context_length)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # We use a lower learning rate to surgically inject the identity weights without damaging general language structure
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    steps = 1500  # Burn identity intensely
    step = 0
    start_time = time.time()
    
    print(f"\nSurgically burning identity for {steps} steps on GPU...")
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
            
    print(f"\nIdentity alignment completed in {time.time() - start_time:.2f} seconds!")
    
    # Save aligned checkpoint
    checkpoint_path_sft = "checkpoints/m01_aligned.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": checkpoint["config"]
    }, checkpoint_path_sft)
    print(f"Aligned checkpoint saved to {checkpoint_path_sft}\n")
    
    # Test Identity prompts
    model.eval()
    prompts = [
        "<|user|>\n¿Quién eres?\n",
        "<|user|>\n¿Quién te creó?\n",
        "<|user|>\n¿En qué hardware corres?\n"
    ]
    for p in prompts:
        print("-" * 50)
        print(f"Query: '{p.strip()}'")
        ans = generate(model, tokenizer, p, max_gen_len=45, temperature=0.3, device=device)
        print(ans)
        print("-" * 50)

if __name__ == "__main__":
    main()
