"""Granular Component Profiler producing training_profile.json with P95/P99 stats."""

import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, List


class GranularProfiler:
    """Component Profiler measuring execution breakdown with P95 and P99 percentiles."""

    def __init__(self) -> None:
        self.timings: Dict[str, List[float]] = {
            "dataloader": [],
            "forward": [],
            "backward": [],
            "optimizer": [],
            "checkpoint": [],
        }
        self.active_timers: Dict[str, float] = {}

    def start(self, key: str) -> None:
        self.active_timers[key] = time.perf_counter()

    def stop(self, key: str) -> float:
        if key in self.active_timers:
            elapsed = (time.perf_counter() - self.active_timers.pop(key)) * 1000.0  # ms
            if key not in self.timings:
                self.timings[key] = []
            self.timings[key].append(elapsed)
            return elapsed
        return 0.0

    def export(self, filepath: Path, total_tokens: int = 1, total_batches: int = 1) -> None:
        """Export timing averages and percentiles to JSON."""
        summary = {}
        for key, values in self.timings.items():
            if values:
                arr = np.array(values)
                summary[key] = {
                    "mean_ms": float(np.mean(arr)),
                    "std_ms": float(np.std(arr)),
                    "p95_ms": float(np.percentile(arr, 95)),
                    "p99_ms": float(np.percentile(arr, 99)),
                    "min_ms": float(np.min(arr)),
                    "max_ms": float(np.max(arr)),
                    "total_ms": float(np.sum(arr)),
                    "calls": len(values),
                }

        summary["global_throughput"] = {
            "total_tokens": total_tokens,
            "total_batches": total_batches,
            "ms_per_token": (sum(summary[k]["total_ms"] for k in summary) / max(total_tokens, 1)),
            "ms_per_batch": (sum(summary[k]["total_ms"] for k in summary) / max(total_batches, 1)),
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
