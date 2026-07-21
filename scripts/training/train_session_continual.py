import os
import sys
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW

# Insert current directory to import local src packages
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.transformer.config import M01Config
from src.model.lm import TransformerLM
from src.tokenizer.bpe import Tokenizer
from src.training.dataset import TinyShakespeareDataset
from src.training.config import TrainingConfig
from src.training.setup import setup_device, setup_stdout

def main():
    setup_stdout()

    parser = argparse.ArgumentParser(description="Session-based continual training for M0.1 (timed).")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/m01_180m", help="Directory for checkpoint storage.")
    parser.add_argument("--data-dir", type=str, default="data/splits_final", help="Directory containing raw data and tokenizer.")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size.")
    parser.add_argument("--duration-min", type=float, default=28.0, help="Maximum session duration in minutes.")
    parser.add_argument("--log-interval", type=int, default=50, help="Steps between log lines.")
    parser.add_argument("--milestone-interval", type=int, default=5000, help="Steps between milestone snapshots.")
    parser.add_argument("--steps", type=int, default=None, help="Maximum number of steps to train in this session.")
    args = parser.parse_args()

    print("=" * 60)
    print("     M0.1-Extended: Session-Based Continual Training")
    print("=" * 60)
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    print(f"Session limit:        {args.duration_min} minutes")
    print(f"Learning rate:        {args.lr}")

    device = setup_device()
    print(f"Using device:         {device}")

    # Ensure checkpoint directory exists
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    base_path = os.path.join(args.checkpoint_dir, "m01_180m_base.pt")
    latest_path = os.path.join(args.checkpoint_dir, "m01_180m_latest.pt")

    checkpoint = None
    start_step = 0
    start_epoch = 0

    # 1. Determine load path (latest -> base -> random init)
    if os.path.exists(latest_path):
        print(f"Resuming training: Loading latest checkpoint: {latest_path}")
        checkpoint = torch.load(latest_path, map_location="cpu", weights_only=True)
    elif os.path.exists(base_path):
        print(f"Starting from base weights: Loading base checkpoint: {base_path}")
        checkpoint = torch.load(base_path, map_location="cpu", weights_only=True)
    else:
        print("No existing checkpoints found. Initializing random weights...")
        # 180M Architecture definition
        config = M01Config(
            vocab_size=8192,
            context_length=256,
            d_model=320,
            n_heads=8,
            d_ff=512,
            d_ff_shared=512,
            d_ff_routed=256,
            n_layers=14,
            num_experts=40,
            num_shared_experts=5,
            moe_top_k=4,
            use_hybrid_attention=True,
            local_window_size=16
        )
        model = TransformerLM(config).to(device)
        
        # Save base checkpoint (step 0 reference)
        from src.training.checkpoint import config_to_dict
        base_checkpoint = {
            "config": config_to_dict(config),
            "model_state_dict": model.state_dict(),
        }
        torch.save(base_checkpoint, base_path)
        print(f"Saved random initialization weights to: {base_path}")

    # Load states if we loaded a checkpoint
    if checkpoint is not None:
        ckpt_config = checkpoint["config"]
        config = M01Config(
            vocab_size=ckpt_config["vocab_size"],
            context_length=ckpt_config["context_length"],
            d_model=ckpt_config["d_model"],
            n_heads=ckpt_config["n_heads"],
            d_ff=ckpt_config["d_ff"],
            d_ff_shared=ckpt_config.get("d_ff_shared"),
            d_ff_routed=ckpt_config.get("d_ff_routed"),
            n_layers=ckpt_config["n_layers"],
            num_experts=ckpt_config["num_experts"],
            num_shared_experts=ckpt_config["num_shared_experts"],
            moe_top_k=ckpt_config["moe_top_k"],
            use_hybrid_attention=ckpt_config["use_hybrid_attention"],
            local_window_size=ckpt_config["local_window_size"],
        )
        model = TransformerLM(config).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Weights loaded successfully!")

        if "step" in checkpoint:
            start_step = checkpoint["step"]
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"]

    # 2. Datasets & DataLoader
    print(f"Loading dataset from: {args.data_dir}")
    train_config = TrainingConfig(seq_len=config.context_length, data_dir=args.data_dir)
    train_dataset = TinyShakespeareDataset(train_config)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    # 3. Optimizer & Scaler Setup
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    criterion = nn.CrossEntropyLoss()

    # Load optimizer and scaler states if present in checkpoint for full continuity
    if checkpoint is not None:
        if "optimizer_state_dict" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                print("Optimizer state restored.")
            except Exception as e:
                print(f"Warning: Could not restore optimizer state ({e}). Initializing fresh optimizer.")
        
        if "scaler_state_dict" in checkpoint and checkpoint["scaler_state_dict"] is not None:
            try:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
                print("GradScaler state restored.")
            except Exception as e:
                print(f"Warning: Could not restore scaler state ({e}).")

    # 4. Training Loop with Time-Budget Guard
    model.train()
    max_duration_seconds = args.duration_min * 60
    start_time = time.time()
    step = start_step
    epoch = start_epoch
    done = False
    last_loss = 0.0

    print(f"\nStarting timed training loop (budget: {args.duration_min} min)...")
    
    while not done:
        for x, y in train_loader:
             # Check step limit
            if args.steps is not None and (step - start_step) >= args.steps:
                print(f"\n[Step Guard] Session step limit of {args.steps} steps reached. Stopping training.")
                done = True
                break

            # Check elapsed time
            elapsed = time.time() - start_time
            if elapsed >= max_duration_seconds:
                print(f"\n[Time Guard] Budget of {args.duration_min} minutes reached. Stopping training.")
                done = True
                break

            x, y = x.to(device), y.to(device)

            with torch.amp.autocast(device_type=device.type, enabled=True):
                logits = model(x)
                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                aux_loss = model.get_aux_loss()
                total_loss = loss + 0.1 * aux_loss
            
            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            last_loss = total_loss.item()

            if (step + 1) % args.log_interval == 0:
                steps_per_sec = (step - start_step + 1) / max(elapsed, 1e-6)
                print(
                    f"Epoch {epoch} | Step {step + 1} | Total Loss: {total_loss.item():.4f} "
                    f"(CE: {loss.item():.4f}, Aux: {aux_loss.item():.4f}) "
                    f"| Speed: {steps_per_sec:.1f} steps/s | Elapsed: {elapsed/60:.2f} min"
                )

            step += 1

            # Save milestone snapshot if reached
            if step % args.milestone_interval == 0:
                milestone_path = os.path.join(args.checkpoint_dir, f"m01_180m_milestone_{step:06d}.pt")
                print(f"\n[Milestone] Saving safety snapshot: {milestone_path}")
                from src.training.checkpoint import config_to_dict
                milestone_dict = {
                    "model_state_dict": model.state_dict(),
                    "config": config_to_dict(config),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
                    "step": step,
                    "epoch": epoch,
                    "loss": last_loss,
                }
                torch.save(milestone_dict, milestone_path)

        if not done:
            epoch += 1

    total_elapsed = time.time() - start_time
    print(f"\nSession training completed in {total_elapsed/60:.2f} minutes.")
    print(f"Completed steps in this session: {step - start_step}")
    print(f"Final step reached: {step} | Loss: {last_loss:.4f}")

    # 5. Save latest checkpoint
    print(f"Saving accumulated state to latest checkpoint: {latest_path}")
    from src.training.checkpoint import config_to_dict
    save_dict = {
        "model_state_dict": model.state_dict(),
        "config": config_to_dict(config),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "step": step,
        "epoch": epoch,
        "loss": last_loss,
    }
    torch.save(save_dict, latest_path)
    print("Latest checkpoint saved successfully!")

if __name__ == "__main__":
    main()
