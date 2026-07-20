"""CLI Entry Point for the M0.1 Research Framework V2 (TrainingEngineV2).

CLI entry: ``python -m src.training.train``
ROCm GPU entry: ``.\\venv_rocm\\Scripts\\python.exe -m src.training.train``
"""

import argparse
import sys
import math
import time
import psutil
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from src.model.lm import TransformerLM
from src.transformer.config import M01Config
from src.training.config import TrainingConfig
from src.training.dataset import TinyShakespeareDataset
from src.engine_v2.engine import TrainingEngineV2
from src.engine_v2.experiment import ExperimentManager


def configure_optimizer(model: torch.nn.Module, config: TrainingConfig) -> AdamW:
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []

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
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
        progress = min(progress, 1.0)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


def worker_init_fn(worker_id: int) -> None:
    """Worker init function to set deterministic seeds per DataLoader worker."""
    worker_seed = torch.initial_seed() % 2**32
    import numpy as np
    import random
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train M0.1 Transformer with TrainingEngineV2.")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--max-lr", type=float, default=3e-4)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--max-norm", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--val-interval", type=int, default=500)
    parser.add_argument("--resume", nargs="?", const="__canonical__", default=None,
                        help="Resume training. No path -> canonical checkpoint of this run; "
                             "with a path -> stack knowledge on top of that checkpoint file.")
    parser.add_argument("--vocab-size", type=int, default=16384, help="Model vocab size (must match tokenizer; 16384 = new 16k tokenizer)")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args(sys.argv[1:])

    train_config = TrainingConfig(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        max_lr=args.max_lr,
        min_lr_ratio=args.min_lr_ratio,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        weight_decay=args.weight_decay,
        max_norm=args.max_norm,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        data_dir=args.data_dir,
    )

    model_config = M01Config(vocab_size=args.vocab_size)
    model = TransformerLM(model_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[M0.1 TRAINING ENGINE V2] Running on {device}")

    # Dataset & Loaders with worker_init_fn
    full_dataset = TinyShakespeareDataset(train_config)
    train_size = int(0.95 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    num_workers = 0 if sys.platform == "win32" else 4

    train_loader = DataLoader(
        train_ds,
        batch_size=train_config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=worker_init_fn,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=worker_init_fn,
    )

    optimizer = configure_optimizer(model, train_config)
    scheduler = get_lr_scheduler(optimizer, train_config.warmup_steps, train_config.max_steps, train_config.min_lr_ratio)

    experiment_mgr = ExperimentManager(base_dir="runs")

    engine = TrainingEngineV2(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_config,
        experiment_manager=experiment_mgr,
        gradient_accumulation_steps=args.grad_accum_steps,
        max_norm=train_config.max_norm,
        device=device,
    )

    if args.resume:
        resume_path = None if args.resume == "__canonical__" else args.resume
        engine.resume(resume_path)

    t_start = time.time()
    summary = engine.fit(max_steps=train_config.max_steps)
    t_end = time.time()

    # Hardware & Speed Stats
    cpu_usage = psutil.cpu_percent()
    ram_usage = psutil.virtual_memory().percent
    vram_mb = torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0
    elapsed = t_end - t_start

    # Print validation & performance report
    print("\n" + "=" * 60)
    print("           M0.1 — RUN COMPLETION REPORT")
    print("=" * 60)
    print(f"  Total steps:       {summary['final_step']}")
    print(f"  Total tokens seen: {summary['total_tokens']:,}")
    print(f"  Elapsed time:      {elapsed:.2f} seconds ({elapsed/60:.2f} mins)")
    print(f"  Final Loss:        {summary['final_loss']:.4f}")
    print(f"  Final LR:          {scheduler.get_last_lr()[0]:.2e}")
    print(f"  GPU VRAM:          {vram_mb:.1f} MB")
    print(f"  CPU Usage:         {cpu_usage:.1f}%")
    print(f"  RAM Usage:         {ram_usage:.1f}%")
    print(f"  Run Directory:     {engine.experiment.run_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
