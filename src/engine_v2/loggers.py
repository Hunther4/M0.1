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
            allocated = data.get("vram_alloc_mb", data.get("vram_mb", 0.0))
            reserved = data.get("vram_reserved_mb", 0.0)
            peak_reserved = data.get("vram_reserved_peak_mb", 0.0)
            print(
                f"step {step+1:6d} | loss {loss:.4f} | lr {lr:.2e} | tok/s {tok_s:5d} | "
                f"VRAM alloc {allocated:.0f}MB | res {reserved:.0f}MB | "
                f"peak {peak_reserved:.0f}MB",
                flush=True,
            )


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
        self.headers_written = self.filepath.exists() and self.filepath.stat().st_size > 0
        self.fieldnames: list[str] | None = None
        if self.headers_written:
            with open(self.filepath, newline="", encoding="utf-8") as f:
                self.fieldnames = next(csv.reader(f), None)

    def log(self, step: int, data: Dict[str, Any]) -> None:
        payload = {"step": step, **data}
        if self.fieldnames is None:
            self.fieldnames = list(payload.keys())
        else:
            missing_fields = [key for key in payload if key not in self.fieldnames]
            if missing_fields:
                self._add_columns(missing_fields)

        mode = "a" if self.filepath.exists() else "w"
        with open(self.filepath, mode, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            if not self.headers_written and mode == "w":
                writer.writeheader()
                self.headers_written = True
            writer.writerow(payload)

    def _add_columns(self, columns: list[str]) -> None:
        """Rewrite an existing metrics CSV with blank values for new columns."""
        assert self.fieldnames is not None
        fieldnames = [*self.fieldnames, *columns]
        temporary_path = self.filepath.with_suffix(f"{self.filepath.suffix}.tmp")

        with (
            open(self.filepath, newline="", encoding="utf-8") as source,
            open(temporary_path, "w", newline="", encoding="utf-8") as target,
        ):
            reader = csv.DictReader(source)
            writer = csv.DictWriter(target, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(reader)

        temporary_path.replace(self.filepath)
        self.fieldnames = fieldnames
