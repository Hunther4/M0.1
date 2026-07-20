"""Metrics Registry for M0.1 TrainingEngine."""

import math
from typing import Dict, Any
import torch

from .state import TrainerState


class Metric:
    """Base interface for computation metrics."""

    def compute(self, state: TrainerState) -> Dict[str, Any]:
        raise NotImplementedError


class CrossEntropyMetric(Metric):
    def compute(self, state: TrainerState) -> Dict[str, Any]:
        return {"ce_loss": state.ce_loss}


class PerplexityMetric(Metric):
    def compute(self, state: TrainerState) -> Dict[str, Any]:
        ppl = math.exp(min(state.val_loss if state.val_loss < float("inf") else state.loss, 50.0))
        return {"perplexity": ppl}


class ThroughputMetric(Metric):
    def compute(self, state: TrainerState) -> Dict[str, Any]:
        return {
            "tokens_per_sec": state.tokens_per_sec,
            "samples_per_sec": state.samples_per_sec,
            "global_tokens": state.global_tokens,
            "billions_of_tokens": state.global_tokens / 1e9,
        }


class MemoryMetric(Metric):
    def compute(self, state: TrainerState) -> Dict[str, Any]:
        return {
            "vram_allocated_mb": state.vram_allocated_mb,
            "vram_peak_mb": state.vram_peak_mb,
        }


class MetricRegistry:
    """Registry coordinating metric computations."""

    def __init__(self) -> None:
        self.metrics: Dict[str, Metric] = {
            "ce": CrossEntropyMetric(),
            "ppl": PerplexityMetric(),
            "throughput": ThroughputMetric(),
            "memory": MemoryMetric(),
        }

    def compute_all(self, state: TrainerState) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        for m in self.metrics.values():
            results.update(m.compute(state))
        return results
