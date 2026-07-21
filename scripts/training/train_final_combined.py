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
from src.training.checkpoint import config_to_dict, save_checkpoint
from src.training.loop import train
from src.training.eval import evaluate_val_loss
from src.training.setup import setup_device, setup_stdout
from torch.utils.data import DataLoader
from torch.optim import AdamW

def main():
    setup_stdout()
    
    print("=" * 60)
    print("     M0.1-Lite: Final Combined Training (Chilean + Quijote + Custom Story)")
    print("=" * 60)
    
    device = setup_device()
    
    # 1. Combine all Chilean corpus, Quijote, and custom stories
    chilean_path = "data/raw_text/chilean_corpus.txt"
    quijote_path = "data/raw_text/quijote.txt"
    becquer_path = "data/raw_text/becquer.txt"
    
    combined_text = ""
    # Add custom story
    custom_story_text = """El sol se derramaba como un río de fuego frente al campo de batalla, tiñendo la hierba y reflejando la sangre derramada con un rojo profundo. Entre el estruendo de la guerra, el choque de espadas resonaba como un trueno. Un joven de ojos marrones, jadeante, se mantenía firme frente a su rival, un caballero experimentado que atacaba sin descanso, sus espadas se encontraban, acero contra acero, desgarrando el aire... (Drack Vans y el Cuerpo de Investigación)"""
    combined_text += custom_story_text + "\n"
    
    # Add Chilean corpus
    if os.path.exists(chilean_path):
        combined_text += open(chilean_path, encoding="utf-8").read() + "\n"
    # Add Bécquer
    if os.path.exists(becquer_path):
        combined_text += open(becquer_path, encoding="utf-8").read() + "\n"
    # Add Quijote
    if os.path.exists(quijote_path):
        combined_text += open(quijote_path, encoding="utf-8").read() + "\n"
        
    # Split train/val (90/10)
    split_idx = int(len(combined_text) * 0.90)
    train_text = combined_text[:split_idx]
    val_text = combined_text[split_idx:]
    
    # 2. Re-train 8K BPE Tokenizer on combined text to cover Chilean, Bécquer, and Quijote vocabulary
    print("\nTraining BPE Tokenizer (8K vocab) on complete mixed corpus...")
    tokenizer = Tokenizer()
    tokenizer.train(train_text, 8192, show_progress=False)
    tokenizer.save("data/tokenizer_final_8k.json")
    vocab_size = len(tokenizer.vocab)
    print(f"Tokenizer trained. Vocab size: {vocab_size}")
    
    # 3. Save splits
    os.makedirs("data/splits_final", exist_ok=True)
    os.makedirs("data/splits_final_val", exist_ok=True)
    with open("data/splits_final/tinyshakespeare.txt", "w", encoding="utf-8") as f:
        f.write(train_text)
    with open("data/splits_final_val/tinyshakespeare.txt", "w", encoding="utf-8") as f:
        f.write(val_text)
        
    tokenizer.save("data/splits_final/tokenizer.json")
    tokenizer.save("data/splits_final_val/tokenizer.json")
    
    # 4. Config & Initialize Model (55M Parameters, 19 Fine-Grained Experts)
    config = M01Config(
        vocab_size=vocab_size,
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
    
    # Try to load previous weights if vocab dimensions match
    prev_ckpt_path = "checkpoints/mixed_8k.pt"
    if os.path.exists(prev_ckpt_path):
        try:
            print("Loading previous checkpoint weights to warm-start...")
            checkpoint = torch.load(prev_ckpt_path, map_location="cpu", weights_only=True)
            # If vocab size is identical, load directly
            if checkpoint["config"]["vocab_size"] == vocab_size:
                model.load_state_dict(checkpoint["model_state_dict"], strict=False)
                print("Weights loaded successfully!")
            else:
                print("Vocabulary sizes match. Direct loading of compatible layers...")
                state_dict = model.state_dict()
                for name, param in checkpoint["model_state_dict"].items():
                    if name in state_dict and param.size() == state_dict[name].size():
                        state_dict[name].copy_(param)
                model.load_state_dict(state_dict, strict=False)
                print("Loaded compatible model layers.")
        except Exception as e:
            print(f"Warm-start initialization failed. Training from scratch. ({e})")
            
    # 5. Datasets
    print("Loading datasets...")
    train_config = TrainingConfig(seq_len=config.context_length, data_dir="data/splits_final")
    train_dataset = TinyShakespeareDataset(train_config)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, pin_memory=True)
    
    val_config = TrainingConfig(seq_len=config.context_length, data_dir="data/splits_final_val")
    val_dataset = TinyShakespeareDataset(val_config)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, pin_memory=True)
    
    optimizer = AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    criterion = nn.CrossEntropyLoss()
    
    # 6. Final Training (4000 steps with AMP)
    model.train()
    steps = 4000
    
    print(f"\nStarting GPU training for {steps} steps on Combined Spanish Corpus...")
    
    train_result = train(model, train_loader, optimizer, criterion, steps, device, log_interval=500, scaler=scaler, val_loader=val_loader, max_batches=20)
    
    print(f"\nGPU Training completed in {train_result['elapsed']:.2f} seconds!")
    
    # Save final checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = "checkpoints/final_combined_8k.pt"
    save_checkpoint(model, config, checkpoint_path)
    print(f"Final checkpoint saved to {checkpoint_path}\n")
    
    # 7. Generation Queries
    model.eval()
    prompts = [
        "En un lugar de la Mancha, de cuyo nombre no quiero acordarme,",
        "Drack sacó su daga del Dios Ciego y"
    ]
    for p in prompts:
        print("-" * 50)
        print(f"Generating for prompt: '{p}'")
        ans = generate(model, tokenizer, p, max_gen_len=45, temperature=0.6, device=device)
        print(ans)
        print("-" * 50)

if __name__ == "__main__":
    main()
