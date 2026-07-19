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
    print("     M0.1-Lite: GPU Training in Spanish (5000 Steps, 8K Vocab)")
    print("=" * 60)
    
    device = setup_device()
    
    # 1. Load Quijote and Split
    quijote_path = "data/quijote.txt"
    if not os.path.exists(quijote_path):
        print(f"Error: Dataset not found at {quijote_path}")
        sys.exit(1)
        
    text_corpus = open(quijote_path, encoding="utf-8").read()
    
    # Clean Project Gutenberg header/footer metadata roughly (optional, but keep it simple)
    # Let's split 90/10
    split_idx = int(len(text_corpus) * 0.90)
    train_text = text_corpus[:split_idx]
    val_text = text_corpus[split_idx:]
    
    # 2. Train Tokenizer on Spanish text
    print("\nTraining BPE Tokenizer (8K vocab) on Don Quijote...")
    tokenizer = Tokenizer()
    tokenizer.train(train_text, 8192, show_progress=False)
    tokenizer.save("data/tokenizer_quijote_8k.json")
    vocab_size = len(tokenizer.vocab)
    print(f"BPE Tokenizer trained. Vocab size: {vocab_size}")
    
    # 3. Save splits and tokenizer for TinyShakespeareDataset class compatibility
    os.makedirs("data/splits_quijote", exist_ok=True)
    os.makedirs("data/splits_quijote_val", exist_ok=True)
    with open("data/splits_quijote/tinyshakespeare.txt", "w", encoding="utf-8") as f:
        f.write(train_text)
    with open("data/splits_quijote_val/tinyshakespeare.txt", "w", encoding="utf-8") as f:
        f.write(val_text)
        
    tokenizer.save("data/splits_quijote/tokenizer.json")
    tokenizer.save("data/splits_quijote_val/tokenizer.json")
    
    # 4. Config & Initialize Model
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
    model = TransformerLM(config).to(device)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters.")
    
    # 5. Datasets
    print("Loading datasets...")
    train_config = TrainingConfig(seq_len=config.context_length, data_dir="data/splits_quijote")
    train_dataset = TinyShakespeareDataset(train_config)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    
    val_config = TrainingConfig(seq_len=config.context_length, data_dir="data/splits_quijote_val")
    val_dataset = TinyShakespeareDataset(val_config)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    optimizer = AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # 6. Training (5000 steps)
    model.train()
    steps = 5000
    
    print(f"\nStarting GPU training for {steps} steps on Don Quijote...")
    
    train_result = train(model, train_loader, optimizer, criterion, steps, device, log_interval=500, val_loader=val_loader)
    
    print(f"\nGPU Training completed in {train_result['elapsed']:.2f} seconds!")
    
    # Save checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = "checkpoints/quijote_8k.pt"
    save_checkpoint(model, config, checkpoint_path)
    print(f"Spanish checkpoint saved to {checkpoint_path}\n")
    
    # 7. Generation
    model.eval()
    prompt = "En un lugar de la Mancha, de cuyo nombre no quiero acordarme,"
    print("-" * 50)
    print(f"Generating prose in Spanish with prompt: '{prompt}'")
    print("-" * 50)
    story = generate(model, tokenizer, prompt, max_gen_len=50, temperature=0.6, device=device)
    print(story)
    print("-" * 50)

if __name__ == "__main__":
    main()
