import os
import sys
import torch.nn as nn
from src.tokenizer.bpe import Tokenizer
from src.inference.generate import generate
from src.training.datasets import JsonlDataset
from src.training.checkpoint import save_checkpoint, load_checkpoint
from src.training.loop import train
from src.training.setup import setup_device, setup_stdout
from torch.utils.data import DataLoader
from torch.optim import AdamW


def main():
    setup_stdout()

    print("=" * 60)
    print("       M0.1-Lite: GPU Training Phase 3 of 3 (7500 Steps)")
    print("=" * 60)

    device = setup_device()

    # 1. Tokenizer
    tokenizer = Tokenizer()
    tokenizer.load("data/tokenizer.json")
    vocab_size = len(tokenizer.vocab)

    # 2. Load Phase 2 checkpoint (fallback to phase1)
    checkpoint_path = "checkpoints/phase2.pt"
    try:
        model, config = load_checkpoint(checkpoint_path, device=device)
    except FileNotFoundError:
        print("Warning: Phase 2 checkpoint not found. Fallback to phase1.")
        checkpoint_path_p1 = "checkpoints/phase1.pt"
        try:
            model, config = load_checkpoint(checkpoint_path_p1, device=device)
        except FileNotFoundError:
            print("Error: No checkpoints found to resume training.")
            sys.exit(1)

    # 3. Load Shards 0000 to 0004 for more data coverage
    shards_dir = "D:/Proyectos/M0.2/data/corpus/synthetic"
    shards_paths = [os.path.join(shards_dir, f"shard_{i:04d}.jsonl") for i in range(5)]

    dataset = JsonlDataset(tokenizer, shards_paths, seq_len=config.context_length, max_lines_per_shard=1000)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    # 4. Training (7500 steps)
    model.train()
    steps = 7500

    print(f"\nStarting GPU training for {steps} steps (Phase 3)...")
    result = train(model, dataloader, optimizer, criterion, steps, device)
    print(f"\nPhase 3 training complete in {result['elapsed']:.2f} seconds!")

    # Save final Phase 3 checkpoint
    checkpoint_path_p3 = "checkpoints/phase3.pt"
    save_checkpoint(model, config, checkpoint_path_p3)
    print(f"Phase 3 checkpoint saved to {checkpoint_path_p3}")

    # 5. Generate validation
    model.eval()
    prompt = "<|user|>\n¿Qué hay en este proyecto?\n<|tool_call|>"
    print("\n" + "-" * 50)
    print(f"Generating after Phase 3 with prompt: '{prompt}'")
    print("-" * 50)
    generated_text = generate(model, tokenizer, prompt, max_gen_len=60, temperature=0.5, device=device)
    print(generated_text)
    print("-" * 50)


if __name__ == "__main__":
    main()
