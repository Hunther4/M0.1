"""ExperimentManager for full experiment traceability and run directory structure."""

import os
import sys
import json
import time
import socket
import platform
import psutil
from pathlib import Path
from typing import Dict, Any


class ExperimentManager:
    """Manages experiment directory structure under runs/XXXX/."""

    def __init__(self, base_dir: str = "runs", run_name: str | None = None) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = self._create_next_run_dir(run_name)
        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.checkpoints_dir.mkdir(exist_ok=True)
        self.metrics_file = self.run_dir / "metrics.jsonl"

    def _create_next_run_dir(self, run_name: str | None = None) -> Path:
        """Find next available run number (0001, 0002, etc.)."""
        if run_name is not None:
            if not run_name or Path(run_name).name != run_name or run_name in {".", ".."}:
                raise ValueError("run_name must be a single non-empty directory name")
            run_dir = self.base_dir / run_name
            run_dir.mkdir(parents=True, exist_ok=True)
            return run_dir
        existing = [d for d in self.base_dir.iterdir() if d.is_dir() and d.name.isdigit()]
        if not existing:
            next_num = 1
        else:
            next_num = max(int(d.name) for d in existing) + 1
        run_dir = self.base_dir / f"{next_num:04d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def capture_full_metadata(self, seed: int = 42) -> Dict[str, Any]:
        """Capture hardware, OS, Python, PyTorch, command line, CPU, RAM, and Git metadata."""
        return {
            "hostname": socket.gethostname(),
            "os": platform.platform(),
            "python_version": sys.version,
            "command_line": " ".join(sys.argv),
            "seed": seed,
            "cpu": platform.processor(),
            "cpu_cores": psutil.cpu_count(logical=True),
            "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        }

    def log_environment(self, env_dict: Dict[str, Any]) -> None:
        """Save environment details to environment.txt."""
        with open(self.run_dir / "environment.txt", "w", encoding="utf-8") as f:
            for k, v in env_dict.items():
                f.write(f"{k}: {v}\n")

    def log_config(self, config_dict: Dict[str, Any]) -> None:
        """Save configuration details to config.json."""
        with open(self.run_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2)

    def log_metrics(self, step: int, metrics_dict: Dict[str, Any]) -> None:
        """Append step metrics to metrics.jsonl."""
        payload = {"step": step, "timestamp": time.time(), **metrics_dict}
        with open(self.metrics_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")

    def save_summary(self, summary_dict: Dict[str, Any]) -> None:
        """Save final run summary."""
        with open(self.run_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary_dict, f, indent=2)
