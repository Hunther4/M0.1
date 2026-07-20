"""Run Manager: runs/run_XXXX/ auto-structured experiment directories.

Creates reproducible experiment folders with:
  runs/
    run_0001/
      config.yaml       # frozen copy of model + training config
      metrics.jsonl      # step-by-step metrics (JSONL)
      train.log          # stdout log
      checkpoint/        # checkpoints go here
      summary.json       # training summary at end
      plots/             # future: auto-generated plots
"""

import os
import json
import yaml
import hashlib
from typing import Any
from dataclasses import dataclass
from datetime import datetime


def config_hash(config: Any) -> str:
    """SHA256 hash of config for reproducibility checks.

    Args:
        config: Any dataclass or dict-like config object.

    Returns:
        8-char hex hash string.
    """
    if hasattr(config, "__dict__"):
        raw = json.dumps(config.__dict__, sort_keys=True, default=str)
    elif isinstance(config, dict):
        raw = json.dumps(config, sort_keys=True, default=str)
    else:
        raw = str(config)
    return hashlib.sha256(raw.encode()).hexdigest()[:8].upper()


class RunManager:
    """Manages a single experiment run directory.

    Usage:
        run = RunManager(base_dir="runs", config=model_config, tag="Stage1")
        run.log_message("Starting training...")
        run.log_metrics(step=100, loss=2.3, entropy=0.8)
        run.save_summary({"best_loss": 2.1, "total_steps": 10000})
    """

    def __init__(
        self,
        base_dir: str = "runs",
        config: Any = None,
        tag: str = "",
    ) -> None:
        self.base_dir = base_dir
        self.config = config
        self.tag = tag

        # Find next run number
        os.makedirs(base_dir, exist_ok=True)
        existing = [d for d in os.listdir(base_dir) if d.startswith("run_")]
        run_num = 0
        for d in existing:
            try:
                n = int(d.split("_")[1])
                run_num = max(run_num, n)
            except (IndexError, ValueError):
                continue
        self.run_num = run_num + 1
        self.run_id = f"run_{self.run_num:04d}"
        self.run_dir = os.path.join(base_dir, self.run_id)

        # Create structure
        self.checkpoint_dir = os.path.join(self.run_dir, "checkpoint")
        self.plots_dir = os.path.join(self.run_dir, "plots")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)

        # Open log file
        self.log_path = os.path.join(self.run_dir, "train.log")
        self._log_file = open(self.log_path, "w", encoding="utf-8")

        # Compute config hash
        self.hash = config_hash(config) if config else "NO_CONFIG"

        # Save config
        if config is not None:
            self._save_config()

        # Init metrics file
        self.metrics_path = os.path.join(self.run_dir, "metrics.jsonl")
        self._metrics_file = open(self.metrics_path, "w", encoding="utf-8")

        # Write header
        self.log_message(f"Run {self.run_id} | Hash: {self.hash} | Tag: {tag} | {datetime.now().isoformat()}")
        self.log_message(f"Config hash: {self.hash}")
        self.log_message(f"Checkpoints: {self.checkpoint_dir}")
        self.log_message("─" * 60)

    def _save_config(self) -> None:
        """Save model config as YAML."""
        path = os.path.join(self.run_dir, "config.yaml")
        if hasattr(self.config, "__dict__"):
            data = {}
            for k, v in self.config.__dict__.items():
                if not k.startswith("_"):
                    data[k] = str(v) if callable(v) else v
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            self.log_message(f"Config saved: {path}")
        else:
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump({"config": str(self.config)}, f)

    def log_message(self, msg: str) -> None:
        """Write a line to train.log."""
        self._log_file.write(msg + "\n")
        self._log_file.flush()

    def log_metrics(self, step: int, **metrics: float | int | str | bool) -> None:
        """Append one JSON line to metrics.jsonl.

        Call after every step or at log_interval.
        """
        record = {"step": step}
        record.update(metrics)
        line = json.dumps(record, default=str)
        self._metrics_file.write(line + "\n")
        self._metrics_file.flush()

    def save_summary(self, summary: dict) -> None:
        """Save training summary as summary.json."""
        summary["run_id"] = self.run_id
        summary["config_hash"] = self.hash
        summary["tag"] = self.tag
        summary["finished_at"] = datetime.now().isoformat()
        path = os.path.join(self.run_dir, "summary.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        self.log_message(f"Summary saved: {path}")

    def close(self) -> None:
        """Close open file handles."""
        self._log_file.close()
        self._metrics_file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
