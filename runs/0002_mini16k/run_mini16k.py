"""Mini training run for the new 16k byte-level BPE tokenizer (M0.1 run 0002_mini16k).

Reuses the existing TrainingEngineV2 / optimizer / scheduler "philosophy" from
src/training/train.py unchanged (minimal change: no engine rewrite). The only
integration points touched:
  * model vocab overridden to 16384 (M01Config(vocab_size=16384))
  * data comes from a BinTokenDataset that reads the uint16 .bin shards built by
    build_dataset.py (the project's own bpe.py Tokenizer produced them)
  * run directory pinned to runs/0002_mini16k via a FixedExperimentManager

Target: ~300 steps on the ROCm GPU (cuda).
"""

import os
import sys
import time
import glob as _glob

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "E:\\M0.1")

from src.model.lm import TransformerLM
from src.transformer.config import M01Config
from src.training.config import TrainingConfig
from src.training.train import configure_optimizer, get_lr_scheduler, worker_init_fn
from src.engine_v2.engine import TrainingEngineV2
from src.engine_v2.experiment import ExperimentManager
from src.engine_v2.checkpoint_v2 import AsyncCheckpointManagerV2

RUN_DIR = "E:\\M0.1\\runs\\0002_mini16k"
DATA_DIR = os.path.join(RUN_DIR, "data")


class FixedExperimentManager(ExperimentManager):
    """ExperimentManager pinned to a specific run dir (no auto-numbered subdir)."""

    def __init__(self, run_dir: str) -> None:
        from pathlib import Path
        self.base_dir = Path(os.path.dirname(run_dir))
        self.run_dir = Path(os.path.abspath(run_dir))
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.checkpoints_dir.mkdir(exist_ok=True)
        self.metrics_file = self.run_dir / "metrics.jsonl"


class BinTokenDataset(torch.utils.data.Dataset):
    """Sliding-window dataset over concatenated uint16 .bin token shards."""

    def __init__(self, data_dir: str, seq_len: int) -> None:
        self.seq_len = seq_len
        import numpy as np
        shard_files = sorted(_glob.glob(os.path.join(data_dir, "shard_*.bin")))
        if not shard_files:
            raise FileNotFoundError(f"No shard_*.bin found in {data_dir}")
        chunks = [torch.from_numpy(np.fromfile(f, dtype=np.uint16).astype(np.int64))
                  for f in shard_files]
        self.tokens = torch.cat(chunks, dim=0)
        print(f"[DATA] loaded {len(self.tokens):,} tokens from {len(shard_files)} shards")
        if len(self.tokens) < seq_len:
            raise ValueError("corpus shorter than seq_len")

    def __len__(self) -> int:
        return max(0, len(self.tokens) - self.seq_len)

    def __getitem__(self, idx: int):
        x = self.tokens[idx: idx + self.seq_len]
        y = self.tokens[idx + 1: idx + self.seq_len + 1]
        return x, y


def main() -> None:
    batch_size = 4          # reduced slightly for the mini test; real run can raise
    seq_len = 1024
    max_steps = 300
    warmup_steps = 200      # keep philosophy's warmup; cosine decays after

    train_config = TrainingConfig(
        batch_size=batch_size,
        seq_len=seq_len,
        max_lr=3e-4,
        min_lr_ratio=0.1,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        weight_decay=0.1,
        max_norm=1.0,
        log_interval=10,
        save_interval=1000,
        data_dir=DATA_DIR,
    )

    model_config = M01Config(vocab_size=16384)   # <-- NEW tokenizer vocab override
    model = TransformerLM(model_config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] params={n_params/1e6:.1f}M  vocab={model_config.vocab_size}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'})")

    dataset = BinTokenDataset(DATA_DIR, seq_len)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=worker_init_fn,
    )

    optimizer = configure_optimizer(model, train_config)
    scheduler = get_lr_scheduler(
        optimizer, warmup_steps, max_steps, train_config.min_lr_ratio
    )

    experiment_mgr = FixedExperimentManager(RUN_DIR)
    engine = TrainingEngineV2(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=loader,
        val_loader=None,
        config=train_config,
        experiment_manager=experiment_mgr,
        gradient_accumulation_steps=1,
        max_norm=train_config.max_norm,
        device=device,
    )

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    summary = engine.fit(max_steps=max_steps)
    elapsed = time.time() - t0

    peak_vram = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0
    peak_vram_gb = peak_vram / 1024.0
    final_loss = summary["final_loss"]
    total_tokens = summary["total_tokens"]
    tok_s = total_tokens / max(elapsed, 1e-6)
    ms_step = (elapsed * 1000.0) / max_steps

    print("\n" + "=" * 64)
    print("        M0.1 MINI 16k RUN 0002_mini16k — SUMMARY")
    print("=" * 64)
    print(f"  Final loss:        {final_loss:.4f}")
    print(f"  Total tokens:      {total_tokens:,}")
    print(f"  Elapsed:           {elapsed:.2f} s ({elapsed/60:.2f} min)")
    print(f"  Throughput:        {tok_s:,.0f} tok/s")
    print(f"  ms/step:           {ms_step:.1f}")
    print(f"  Peak VRAM:         {peak_vram:,.0f} MB  ({peak_vram_gb:.2f} GB)")
    print(f"  Run dir:           {RUN_DIR}")
    print("=" * 64)

    # Persist a compact run summary for the report.
    with open(os.path.join(RUN_DIR, "run_summary.json"), "w", encoding="utf-8") as f:
        import json
        json.dump({
            "final_loss": final_loss,
            "total_tokens": total_tokens,
            "elapsed_s": elapsed,
            "tok_s": tok_s,
            "ms_per_step": ms_step,
            "peak_vram_mb": peak_vram,
            "params_m": n_params / 1e6,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "max_steps": max_steps,
            "vocab_size": model_config.vocab_size,
            "device": str(device),
        }, f, indent=2)


if __name__ == "__main__":
    main()
