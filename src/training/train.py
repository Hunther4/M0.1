"""Training loop for the M0.1 decoder-only transformer.

CLI entry: ``python -m src.training.train``

Provides configure_optimizer, get_lr_scheduler, train(), and a CLI entry
point that builds the model, dataset, optimizer, and scheduler, then runs
the autoregressive language model training loop in fp32.
"""

import argparse
import math
import sys
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from src.model.lm import TransformerLM
from src.training.checkpoint import CheckpointManager
from src.training.config import TrainingConfig
from src.training.dataset import TinyShakespeareDataset
from src.transformer.config import M01Config


def configure_optimizer(model: nn.Module, config: TrainingConfig) -> AdamW:
    """Create AdamW optimizer with separate param groups for weight decay.

    Parameters with ``bias`` or ``gamma`` in their name are excluded from
    weight decay. All other parameters receive ``config.weight_decay``.

    Args:
        model: The model whose parameters to optimize.
        config: TrainingConfig with weight_decay, max_lr, beta1, beta2.

    Returns:
        AdamW optimizer with two param groups (decay, no_decay).
    """
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "gamma" in name:
            no_decay.append(param)
        else:
            decay.append(param)

    return AdamW(
        [
            {"params": decay, "weight_decay": config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=config.max_lr,
        betas=(config.beta1, config.beta2),
        eps=1e-8,
    )


def get_lr_scheduler(
    optimizer: AdamW,
    warmup_steps: int,
    max_steps: int,
    min_lr_ratio: float,
) -> LambdaLR:
    """Create a LambdaLR scheduler with linear warmup then cosine decay.

    The learning rate follows:
    - ``0 → max_lr`` linearly for the first ``warmup_steps`` steps.
    - ``max_lr → min_lr_ratio * max_lr`` via cosine decay for the remaining steps.

    Args:
        optimizer: The optimizer whose LR will be scheduled.
        warmup_steps: Number of linear warmup steps.
        max_steps: Total training steps.
        min_lr_ratio: Minimum LR as fraction of peak LR (decay floor).

    Returns:
        LambdaLR scheduler.
    """

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


def train(
    config: TrainingConfig,
    model_config: M01Config,
    resume_checkpoint: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the training loop for the M0.1 model.

    Args:
        config: TrainingConfig with all hyperparameters.
        model_config: M01Config defining the model architecture.
        resume_checkpoint: Optional path to a checkpoint directory to resume from.

    Returns:
        Dict with final ``step`` and ``loss`` values.
    """
    # --- Build model, dataset, dataloader ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerLM(model_config).to(device)
    dataset = TinyShakespeareDataset(config)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    data_iter = iter(loader)

    # --- Optimizer and scheduler ---
    optimizer = configure_optimizer(model, config)
    scheduler = get_lr_scheduler(
        optimizer,
        config.warmup_steps,
        config.max_steps,
        config.min_lr_ratio,
    )

    # --- Checkpoint manager ---
    checkpoint_manager = CheckpointManager(config.checkpoint_dir)

    start_step = 0

    # --- Resume from checkpoint if provided ---
    if resume_checkpoint is not None:
        alt_manager = CheckpointManager(resume_checkpoint)
        state = alt_manager.load(model, optimizer, scheduler)
        start_step = state["step"] + 1

    # --- AMP Gradient Scaler ---
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # --- Training loop ---
    model.train()
    step_loss: float = 0.0

    for step in range(start_step, config.max_steps):
        # Get next batch (restart iterator if exhausted)
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x, y = next(data_iter)

        # Move data to device
        x, y = x.to(device), y.to(device)

        # Forward with autocast
        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits = model(x)
            loss = F.cross_entropy(
                logits.view(-1, model_config.vocab_size),
                y.view(-1),
            )

        # Backward with scaling
        scaler.scale(loss).backward()
        step_loss = loss.item()

        # Non-finite loss guard (NaN or Inf) — check BEFORE optimizer.step
        if not math.isfinite(step_loss):
            print(
                f"Non-finite loss at step {step}: {step_loss:.4e}, "
                "aborting training",
                flush=True,
            )
            break

        # Unscale for gradient clipping
        scaler.unscale_(optimizer)
        total_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.max_norm
        )
        
        # Step and update scaler
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad()

        # Logging
        if (step + 1) % config.log_interval == 0:
            lr = scheduler.get_last_lr()[0]
            print(
                f"step {step + 1:>6d} | "
                f"loss {step_loss:.4f} | "
                f"lr {lr:.2e} | "
                f"norm {total_norm:.1f}",
                flush=True,
            )

        # Checkpoint save
        if (step + 1) % config.save_interval == 0:
            checkpoint_manager.save(
                step=step,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                loss=step_loss,
                config=model_config.__dict__,
            )

    return {"step": step, "loss": step_loss}


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse CLI arguments, overriding TrainingConfig defaults.

    Args:
        argv: Command-line argument list (typically ``sys.argv[1:]``).

    Returns:
        Parsed namespace with all TrainingConfig fields as attributes.
    """
    parser = argparse.ArgumentParser(
        description="Train the M0.1 decoder-only transformer.",
    )

    # Data / checkpoint paths
    parser.add_argument(
        "--data-dir", type=str, default="data",
        help="Directory containing training data (default: data)",
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default="checkpoints",
        help="Directory for model checkpoints (default: checkpoints)",
    )

    # Training hyperparameters
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Batch size (default: 4)",
    )
    parser.add_argument(
        "--seq-len", type=int, default=1024,
        help="Sequence length (default: 1024)",
    )
    parser.add_argument(
        "--max-lr", type=float, default=3e-4,
        help="Peak learning rate (default: 3e-4)",
    )
    parser.add_argument(
        "--min-lr-ratio", type=float, default=0.1,
        help="Minimum LR as fraction of max_lr (default: 0.1)",
    )
    parser.add_argument(
        "--warmup-steps", type=int, default=200,
        help="Linear warmup steps (default: 200)",
    )
    parser.add_argument(
        "--max-steps", type=int, default=100_000,
        help="Total training steps (default: 100000)",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.1,
        help="AdamW weight decay (default: 0.1)",
    )
    parser.add_argument(
        "--beta1", type=float, default=0.9,
        help="Adam beta1 (default: 0.9)",
    )
    parser.add_argument(
        "--beta2", type=float, default=0.95,
        help="Adam beta2 (default: 0.95)",
    )
    parser.add_argument(
        "--max-norm", type=float, default=1.0,
        help="Gradient clipping max norm (default: 1.0)",
    )
    parser.add_argument(
        "--log-interval", type=int, default=10,
        help="Steps between logging (default: 10)",
    )
    parser.add_argument(
        "--save-interval", type=int, default=500,
        help="Steps between checkpoint saves (default: 500)",
    )

    return parser.parse_args(argv)


def main() -> None:
    """CLI entry point for ``python -m src.training.train``."""
    args = parse_args(sys.argv[1:])

    config = TrainingConfig(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        max_lr=args.max_lr,
        min_lr_ratio=args.min_lr_ratio,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
        max_norm=args.max_norm,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        checkpoint_dir=args.checkpoint_dir,
        data_dir=args.data_dir,
    )

    model_config = M01Config()

    result = train(config, model_config)
    print(
        f"\nTraining complete. Final loss: {result['loss']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
