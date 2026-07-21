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
from src.training.checkpoint import save_checkpoint
from src.training.setup import setup_device, setup_stdout
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW

class CoTDistillationDataset(Dataset):
    """
    Dataset that loads the Qwen3.5-9B distilled Chain-of-Thought (CoT) samples.
    Prepares inputs and targets for supervised fine-tuning (SFT) of M0.1.
    """
    def __init__(self, tokenizer, jsonl_path, seq_len=256):
        self.seq_len = seq_len
        self.samples = []
        
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"No se encontro el dataset en: {jsonl_path}")
            
        print(f"Cargando dataset CoT de: {jsonl_path}")
        
        # Load and parse jsonl
        total_tokens = 0
        all_token_ids = []
        
        # We also mix identity prompts to keep the model aligned to its identity
        identity_prompts = [
            "<|user|>\n¿Quién eres?\n<|think|>\nEl usuario me está preguntando por mi identidad. Debo responder de forma clara afirmando que soy M0.1.\n</think>\n<|assistant|>\nSoy M0.1, un modelo de lenguaje con arquitectura de mezcla de expertos (MoE) creado por Hunther4 y entrenado en su GPU AMD Radeon RX 9060XT.<|endoftext|>",
            "<|user|>\n¿Quién te creó?\n<|think|>\nEl usuario quiere saber mi origen. Fui creado por Hunther4.\n</think>\n<|assistant|>\nFui creado por Hunther4. Soy M0.1, una Inteligencia Artificial entrenada localmente en su GPU AMD Radeon RX 9060XT.<|endoftext|>",
            "<|user|>\n¿Qué significa LPTM?\n<|think|>\nEl usuario pregunta por la jerga 'LPTM'. Significa 'La Puta Madre' en español coloquial.\n</think>\n<|assistant|>\nLPTM es la abreviación de 'La Puta Madre', una expresión coloquial muy común en el español de América Latina y España usada para expresar frustración o sorpresa.<|endoftext|>"
        ]
        
        # Add identity prompts amplified to ensure they are well represented
        for _ in range(15):
            for ident in identity_prompts:
                all_token_ids.extend(tokenizer.encode(ident))
        
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    # Use the pre-formatted text containing user, think, and assistant tags
                    formatted_text = data.get("text", "")
                    if formatted_text:
                        tokens = tokenizer.encode(formatted_text)
                        all_token_ids.extend(tokens)
                except Exception as e:
                    print(f"Error parsing line: {e}")
                    
        self.tokens = torch.tensor(all_token_ids, dtype=torch.long)
        print(f"Dataset cargado y tokenizado. Total tokens en buffer: {len(self.tokens)}")
        
    def __len__(self):
        if len(self.tokens) <= self.seq_len:
            return 0
        # Number of non-overlapping chunks
        return (len(self.tokens) - 1) // self.seq_len
        
    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len
        # Shift targets by 1 for autoregressive next-token prediction
        return self.tokens[start:end], self.tokens[start+1:end+1]

def main():
    setup_stdout()
        
    print("=" * 60)
    print("      M0.1-Lite: Chain-of-Thought (CoT) SFT Fine-Tuning (55M)")
    print("=" * 60)
    
    device = setup_device()
    
    # 1. Load trained 8K Tokenizer
    tokenizer = Tokenizer()
    tokenizer.load("data/tokenizers/tokenizer_final_8k.json")
    
    # 2. Dataset Paths
    distill_path = "data/distillation/distill_qwen9b_20260718_1803_clean.jsonl"
    
    try:
        dataset = CoTDistillationDataset(tokenizer, distill_path, seq_len=256)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)
        
    loader = DataLoader(dataset, batch_size=4, shuffle=True, pin_memory=True)
    
    # 3. Model Configuration (Matches the 55M parameters model we just trained)
    config = M01Config(
        vocab_size=len(tokenizer.vocab),
        context_length=256,
        d_model=364,
        n_heads=7,
        d_ff=624,
        n_layers=10,
        num_experts=16,
        num_shared_experts=3,
        moe_top_k=4,
        use_hybrid_attention=True,
        local_window_size=16
    )
    
    model = TransformerLM(config).to(device)
    
    # Load weights from the newly trained 55M base checkpoint
    base_ckpt_path = "checkpoints/final_combined_8k.pt"
    if os.path.exists(base_ckpt_path):
        print(f"Cargando pesos base de: {base_ckpt_path}")
        checkpoint = torch.load(base_ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print("Error: No se encontró el checkpoint base de 55M. Por favor, entrena el modelo base primero.")
        sys.exit(1)
        
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    criterion = nn.CrossEntropyLoss()
    
    # 4. Training (1000 steps with AMP)
    model.train()
    steps = 1000
    step = 0
    start_time = time.time()
    
    print(f"\nIniciando fine-tuning CoT SFT por {steps} pasos...")
    
    done = False
    while not done:
        for x, y in loader:
            if step >= steps:
                done = True
                break
                
            x, y = x.to(device), y.to(device)
            
            with torch.amp.autocast(device_type=device.type, enabled=True):
                logits = model(x)
                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            if (step + 1) % 100 == 0:
                elapsed = time.time() - start_time
                steps_per_sec = (step + 1) / elapsed
                print(f"Paso {step + 1}/{steps} | SFT Loss: {loss.item():.4f} | Velocidad: {steps_per_sec:.1f} pasos/s | Tiempo: {elapsed:.1f}s")
                
            step += 1
            
    print(f"\nFine-tuning completado en {time.time() - start_time:.2f} segundos.")
    
    # 5. Save SFT Aligned Model
    aligned_ckpt_path = "checkpoints/m01_hardened_final.pt"
    save_checkpoint(model, config, aligned_ckpt_path)
    print(f"Checkpoint SFT CoT guardado exitosamente en: {aligned_ckpt_path}\n")
    
    # 6. Evaluation Generation Test
    model.eval()
    prompts = [
        "¿Quién eres y quién te creó?",
        "Si tengo 3 cajas con 12 manzanas cada una y regalo 10, ¿cuántas manzanas me quedan en total?",
        "¿Qué significa la expresión LPTM en español?"
    ]
    
    print("=== DIALOGUE TEST (SFT ALIGNED MODEL) ===")
    for p in prompts:
        print("-" * 50)
        formatted_prompt = f"<|user|>\n{p}\n<|think|>\n"
        print(f"Prompt: {p}")
        ans = generate(model, tokenizer, formatted_prompt, max_gen_len=100, temperature=0.3, device=device)
        print(f"Generado:\n{ans}")
        print("-" * 50)

if __name__ == "__main__":
    main()
