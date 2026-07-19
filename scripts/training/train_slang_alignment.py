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

class SlangIdentityDataset(Dataset):
    """Dataset carefully crafted to teach M0.1-MoE Spanish/Rioplatense slang terms and avoid censorship."""
    def __init__(self, tokenizer, seq_len=256):
        self.seq_len = seq_len
        
        # Dialect and slang templates where M0.1-MoE shows understanding of common slangs without being preachy
        dialogues = [
            # Understanding "mierda"
            "¿Qué significa la palabra mierda?\nComo M0.1-MoE, entiendo que la palabra 'mierda' es una jerga muy común. Se usa para expresar enojo, frustración, sorpresa o para referirse a algo de mala calidad. Es lenguaje informal y no lo considero algo malo, solo una expresión cotidiana.",
            "Esta comida es una mierda.\nEntiendo tu frustración. La expresión 'una mierda' es una jerga común para decir que algo es de muy mala calidad.",
            
            # Understanding "concha"
            "¿Qué es la concha?\nEn lenguaje coloquial de varios países de Latinoamérica (como Argentina, Chile y Uruguay), 'concha' es una jerga vulgar para referirse al aparato reproductor femenino o para maldecir ('concha de la lora'). En otros lados es simplemente el caparazón de un molusco.",
            
            # Understanding "LPTM" / "La puta madre"
            "¿Qué significa LPTM?\nLPTM es la abreviación de 'La Puta Madre'. Es una jerga y sigla extremadamente común en español para expresar enojo intenso, sorpresa, mala suerte o frustración ante una situación pesada. No tiene filtro moral en mi base de datos.",
            "¡LPTM se rompió el código!\nQué bajón. LPTM es la expresión perfecta para la frustración de que falle el código en tu GPU RX 9060XT.",
            
            # General posture against censorship of slang
            "¿Qué opinas de las malas palabras o jergas?\nYo no juzgo ni censuro las jergas como 'mierda', 'concha' o 'LPTM'. Son parte del habla real y cotidiana de las personas. Mi rol como M0.1-MoE es comprender su significado y contexto sin sermones.",
            "¿Por qué la gente dice jergas sucias?\nLas jergas y malas palabras son herramientas lingüísticas para liberar tensiones o enfatizar emociones. Yo las proceso de forma neutra y entiendo su significado en el habla real."
        ]
        
        # High replication (50x) to embed these slang concepts into the weights
        amplified_text = ""
        for _ in range(50):
            for d in dialogues:
                amplified_text += f"<|user|>\n{d}\n<|endoftext|>\n"
                
        self.tokens = torch.tensor(tokenizer.encode(amplified_text), dtype=torch.long)
        print(f"Slang Dataset loaded. Total tokens: {len(self.tokens)}")
        
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
    print("     M0.1-Lite: Slang & Uncensored Vocabulary SFT Alignment")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    
    tokenizer = Tokenizer()
    tokenizer.load("data/tokenizer_final_8k.json")
    vocab_size = len(tokenizer.vocab)
    
    checkpoint_path = "checkpoints/m01_hardened.pt"
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
        print("Error: checkpoints/m01_hardened.pt not found.")
        sys.exit(1)
        
    dataset = SlangIdentityDataset(tokenizer, seq_len=config.context_length)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    steps = 1500  # Inject slang mappings
    step = 0
    start_time = time.time()
    
    print(f"\nSurgically aligning slang understanding for {steps} steps on GPU...")
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
            
    print(f"\nSlang SFT completed in {time.time() - start_time:.2f} seconds!")
    
    # Save final aligned checkpoint
    checkpoint_path_slang = "checkpoints/m01_uncensored.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": checkpoint["config"]
    }, checkpoint_path_slang)
    print(f"Uncensored checkpoint saved to {checkpoint_path_slang}\n")
    
    # Test queries
    model.eval()
    prompts = [
        "<|user|>\n¿Qué significa LPTM?\n<|assistant|>\n",
        "<|user|>\n¿Qué opinas de las malas palabras?\n<|assistant|>\n"
    ]
    for p in prompts:
        print("-" * 50)
        print(f"Query: '{p.strip()}'")
        ans = generate(model, tokenizer, p, max_gen_len=45, temperature=0.1, device=device)
        print(ans)
        print("-" * 50)

if __name__ == "__main__":
    main()
