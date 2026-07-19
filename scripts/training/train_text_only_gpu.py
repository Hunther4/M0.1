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

def run_story_test(model, tokenizer, device):
    """Prompt model to complete a dialogue or story and verify natural language coherence."""
    prompt = "ROMEO:\nIf I profane with my unworthiest hand\n"
    generated = generate(model, tokenizer, prompt, max_gen_len=50, temperature=0.7, device=device)
    return generated

def main():
    setup_stdout()
    
    print("=" * 60)
    print("      M0.1-Lite: Text-Only GPU Training (5000 Steps, 8K Vocab)")
    print("=" * 60)
    
    device = setup_device()
    
    # 1. Train Tokenizer on Shakespeare text only
    print("\nTraining BPE Tokenizer (8K vocab) on clean Shakespeare corpus...")
    tokenizer = Tokenizer()
    shakespeare_path = "data/tinyshakespeare.txt"
    if not os.path.exists(shakespeare_path):
        print(f"Error: Dataset not found at {shakespeare_path}")
        sys.exit(1)
        
    text_corpus = open(shakespeare_path, encoding="utf-8").read()
    
    # Split text into 90% train, 10% val for clean evaluation
    split_idx = int(len(text_corpus) * 0.9)
    train_text = text_corpus[:split_idx]
    val_text = text_corpus[split_idx:]
    
    # Save split texts to temp files for TinyShakespeareDataset to read
    os.makedirs("data/splits", exist_ok=True)
    os.makedirs("data/splits_val", exist_ok=True)
    with open("data/splits/tinyshakespeare.txt", "w", encoding="utf-8") as f:
        f.write(train_text)
    with open("data/splits_val/tinyshakespeare.txt", "w", encoding="utf-8") as f:
        f.write(val_text)
        
    tokenizer.train(train_text, 8192, show_progress=False)
    tokenizer.save("data/tokenizer_text_8k.json")
    vocab_size = len(tokenizer.vocab)
    print(f"BPE Tokenizer trained. Vocab size: {vocab_size}")
    
    # 2. Config & Initialize Model
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
    
    # 3. Datasets (using train and val splits)
    print("Loading datasets...")
    tokenizer.save("data/splits/tokenizer.json")
    tokenizer.save("data/splits_val/tokenizer.json")
    
    train_config = TrainingConfig(seq_len=config.context_length, data_dir="data/splits")
    train_dataset = TinyShakespeareDataset(train_config)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    
    val_config = TrainingConfig(seq_len=config.context_length, data_dir="data/splits_val")
    val_dataset = TinyShakespeareDataset(val_config)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    optimizer = AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # 4. Training (5000 steps)
    model.train()
    steps = 5000
    
    print(f"\nStarting GPU training for {steps} steps...")
    
    train_result = train(model, train_loader, optimizer, criterion, steps, device, log_interval=500, val_loader=val_loader)
    
    print(f"\nGPU Training completed in {train_result['elapsed']:.2f} seconds!")
    
    # Save checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = "checkpoints/text_only_8k.pt"
    save_checkpoint(model, config, checkpoint_path)
    print(f"Text-only checkpoint saved to {checkpoint_path}\n")
    
    # 5. Generation
    model.eval()
    print("-" * 50)
    print("Generating dialogue after training:")
    print("-" * 50)
    story = run_story_test(model, tokenizer, device)
    print(story)
    print("-" * 50)

if __name__ == "__main__":
    main()
