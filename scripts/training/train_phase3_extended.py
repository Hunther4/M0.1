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
    print("     M0.1-Lite: GPU Training & Generalization Audit (5800 Steps)")
    print("=" * 60)

    device = setup_device()

    tokenizer = Tokenizer()
    tokenizer.load("data/tokenizer.json")

    checkpoint_path = "checkpoints/phase3.pt"
    try:
        model, config = load_checkpoint(checkpoint_path, device=device)
    except FileNotFoundError:
        print("Error: checkpoints/phase3.pt not found.")
        sys.exit(1)

    # Load 10 training shards (0 to 9) to improve data generalizability
    shards_dir = "D:/Proyectos/M0.2/data/corpus/synthetic"
    train_shards = [os.path.join(shards_dir, f"shard_{i:04d}.jsonl") for i in range(10)]
    train_dataset = JsonlDataset(tokenizer, train_shards, seq_len=config.context_length, max_lines_per_shard=800)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    # Load 1 unseen validation shard (10) to verify generalization (Val Loss)
    val_shard = [os.path.join(shards_dir, "shard_0010.jsonl")]
    val_dataset = JsonlDataset(tokenizer, val_shard, seq_len=config.context_length, max_lines_per_shard=300)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    optimizer = AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    model.train()
    steps = 5800

    print(f"\nStarting training on GPU for {steps} steps...")
    result = train(
        model, train_loader, optimizer, criterion, steps, device,
        log_interval=500, val_loader=val_loader,
    )
    print(f"\nGPU Training completed in {result['elapsed']:.2f} seconds!")

    checkpoint_path_final = "checkpoints/phase3_final.pt"
    save_checkpoint(model, config, checkpoint_path_final)
    print(f"Final checkpoint saved to {checkpoint_path_final}")

    model.eval()
    prompt = "<|user|>\n¿Qué hay en este proyecto?\n<|tool_call|>"
    print("\n" + "-" * 50)
    print(f"Generating after Phase 3 Final with prompt: '{prompt}'")
    print("-" * 50)
    generated_text = generate(model, tokenizer, prompt, max_gen_len=60, temperature=0.5, device=device)
    print(generated_text)
    print("-" * 50)


if __name__ == "__main__":
    main()
