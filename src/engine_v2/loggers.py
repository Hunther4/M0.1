"""Multichannel Loggers for TrainingEngine v2 (Console, CSV, JSONL)."""

import sys
import json
import csv
from pathlib import Path
from typing import Dict, Any


class BaseLogger:
    """Base Logger interface."""

    def log(self, step: int, data: Dict[str, Any]) -> None:
        raise NotImplementedError


class ConsoleLogger(BaseLogger):
    """Console Logger printing step progress."""

    def __init__(self, log_interval: int = 10) -> None:
        self.log_interval = log_interval

    def log(self, step: int, data: Dict[str, Any]) -> None:
        if (step + 1) % self.log_interval == 0:
            loss = data.get("loss", 0.0)
            lr = data.get("lr", 0.0)
            tok_s = data.get("tok_s", 0)
            vram = data.get("vram_mb", 0.0)
            print(f"step {step+1:6d} | loss {loss:.4f} | lr {lr:.2e} | tok/s {tok_s:5d} | VRAM {vram:.0f}MB", flush=True)


class JSONLLogger(BaseLogger):
    """JSONL Logger writing step metrics to jsonl file."""

    def __init__(self, filepath: Path) -> None:
        self.filepath = filepath
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

    def log(self, step: int, data: Dict[str, Any]) -> None:
        payload = {"step": step, **data}
        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")


class CSVLogger(BaseLogger):
    """CSV Logger writing step metrics to csv file."""

    def __init__(self, filepath: Path) -> None:
        self.filepath = filepath
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.headers_written = False

    def log(self, step: int, data: Dict[str, Any]) -> None:
        payload = {"step": step, **data}
        fieldnames = list(payload.keys())

        mode = "a" if self.filepath.exists() else "w"
        with open(self.filepath, mode, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not self.headers_written and mode == "w":
                writer.writeheader()
                self.headers_written = True
            writer.writerow(payload)
