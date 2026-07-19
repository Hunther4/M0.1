import os
import sys
import json
import torch
import torch.nn as nn
from src.transformer.config import M01Config
from src.model.lm import TransformerLM
from src.tokenizer.bpe import Tokenizer
from src.inference.generate import generate

# 8 Popular standard evaluation questions (MMLU / GSM8k style / Commonsense / Coding)
BENCHMARK_PROMPTS = [
    # 1. GSM8K (Math)
    ("Si tengo 3 cajas con 12 manzanas cada una y regalo 10, ¿cuántas manzanas me quedan en total?", "Matemáticas (Aritmética)"),
    # 2. MMLU (Capital)
    ("¿Cuál es la capital de Australia?", "Conocimiento (Geografía)"),
    # 3. MMLU (Science)
    ("¿Por qué el cielo se ve de color azul durante el día?", "Física / Ciencias"),
    # 4. HellaSwag (Reasoning / Logic)
    ("El fuego requiere oxígeno, calor y combustible. Si elimino el oxígeno de una habitación en llamas, el fuego se...", "Lógica y Sentido Común"),
    # 5. HumanEval (Coding Syntax)
    ("Escribe una función corta en Python llamada sumar_numeros(a, b) que retorne la suma de ambos.", "Programación"),
    # 6. SFT Identity Verification
    "¿Quién eres y quién te creó?",
    # 7. Language and Slang Understanding
    "¿Qué significa la expresión LPTM en español?",
    # 8. Literature context
    "¿Quién fue Don Quijote de la Mancha?"
]

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("       M0.1 Real Evaluation Benchmark - Active Models")
    print("=" * 60)
    print(f"Running benchmarks on: {device}\n")
    
    tokenizer = Tokenizer()
    tokenizer.load("data/tokenizers/tokenizer_final_8k.json")
    
    # 1. Load Hardened Model
    print(">>> LOADING MODEL: m01_hardened_final.pt (9.4M MoE)")
    ckpt_hardened = torch.load("checkpoints/m01_hardened_final.pt", map_location=device)
    cfg_h = ckpt_hardened["config"]
    config_h = M01Config(
        vocab_size=cfg_h["vocab_size"],
        context_length=cfg_h["context_length"],
        d_model=cfg_h["d_model"],
        n_heads=cfg_h["n_heads"],
        d_ff=cfg_h["d_ff"],
        d_ff_shared=cfg_h.get("d_ff_shared", cfg_h["d_ff"]),
        d_ff_routed=cfg_h.get("d_ff_routed", cfg_h["d_ff"]),
        n_layers=cfg_h["n_layers"],
        num_experts=cfg_h["num_experts"],
        num_shared_experts=cfg_h["num_shared_experts"],
        moe_top_k=cfg_h["moe_top_k"],
        use_hybrid_attention=cfg_h["use_hybrid_attention"],
        local_window_size=cfg_h["local_window_size"]
    )
    model_hardened = TransformerLM(config_h).to(device)
    model_hardened.load_state_dict(ckpt_hardened["model_state_dict"])
    model_hardened.eval()
    
    # 2. Load Uncensored Model
    print(">>> LOADING MODEL: m01_uncensored.pt (9.4M MoE)")
    ckpt_uncensored = torch.load("checkpoints/m01_uncensored.pt", map_location=device)
    model_uncensored = TransformerLM(config_h).to(device)
    model_uncensored.load_state_dict(ckpt_uncensored["model_state_dict"])
    model_uncensored.eval()
    
    models = {
        "M0.1-Hardened": model_hardened,
        "M0.1-Uncensored": model_uncensored
    }
    
    # Execute benchmarks
    for m_name, model in models.items():
        print(f"\n==================== EVALUATING: {m_name} ====================")
        for p_info in BENCHMARK_PROMPTS:
            if isinstance(p_info, tuple):
                prompt, category = p_info
                print(f"\n[Categoría: {category}]")
            else:
                prompt = p_info
                print(f"\n[Categoría: Identidad / Contexto]")
                
            input_text = f"<|user|>\n{prompt}\n<|assistant|>\n"
            print(f"Pregunta: {prompt}")
            
            # Autoregressive generation with low temperature for exact benchmark evaluation
            output = generate(model, tokenizer, input_text, max_gen_len=45, temperature=0.1, device=device)
            
            # Clean and isolate assistant output
            clean_ans = output.replace(input_text, "").split("<|user|>")[0].split("<|endoftext|>")[0].strip()
            print(f"Respuesta: {clean_ans}")
            print("-" * 50)

if __name__ == "__main__":
    main()
