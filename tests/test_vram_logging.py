"""Tests for CUDA/ROCm allocator memory logging."""

import csv

import torch

from src.engine_v2.engine import (
    get_cuda_memory_metrics_mb,
    reset_cuda_peak_memory_stats,
)
from src.engine_v2.loggers import CSVLogger, ConsoleLogger


def test_cuda_memory_metrics_include_allocated_reserved_and_peak(monkeypatch):
    """Allocator metrics report current and run-peak reserved VRAM in MB."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 256_000_000)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 512_000_000)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 768_000_000)

    assert get_cuda_memory_metrics_mb() == {
        "vram_mb": 256.0,
        "vram_alloc_mb": 256.0,
        "vram_reserved_mb": 512.0,
        "vram_reserved_peak_mb": 768.0,
    }


def test_cuda_memory_metrics_are_zero_without_cuda(monkeypatch):
    """CPU runs retain the legacy VRAM key and report zero allocator metrics."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert get_cuda_memory_metrics_mb() == {
        "vram_mb": 0.0,
        "vram_alloc_mb": 0.0,
        "vram_reserved_mb": 0.0,
        "vram_reserved_peak_mb": 0.0,
    }


def test_peak_memory_reset_runs_once_only_when_cuda_is_available(monkeypatch):
    """The reset helper invokes PyTorch's CUDA API once and skips CPU runs."""
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: calls.append("reset"))

    reset_cuda_peak_memory_stats()
    assert calls == ["reset"]

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    reset_cuda_peak_memory_stats()
    assert calls == ["reset"]


def test_console_logger_prints_all_allocator_metrics(capsys):
    """Step output names allocated, reserved, and peak-reserved VRAM explicitly."""
    logger = ConsoleLogger(log_interval=1)

    logger.log(
        0,
        {
            "loss": 1.25,
            "lr": 0.001,
            "tok_s": 42,
            "vram_alloc_mb": 256.0,
            "vram_reserved_mb": 512.0,
            "vram_reserved_peak_mb": 768.0,
        },
    )

    assert capsys.readouterr().out == (
        "step      1 | loss 1.2500 | lr 1.00e-03 | tok/s    42 | "
        "VRAM alloc 256MB | res 512MB | peak 768MB\n"
    )


def test_csv_logger_migrates_legacy_vram_header(tmp_path):
    """Resumed runs keep legacy rows while adding the new allocator columns."""
    path = tmp_path / "metrics.csv"
    path.write_text(
        "step,loss,lr,tok_s,vram_mb\n0,1.0,0.001,42,256.0\n",
        encoding="utf-8",
    )
    logger = CSVLogger(path)

    logger.log(
        1,
        {
            "loss": 0.5,
            "lr": 0.001,
            "tok_s": 84,
            "vram_mb": 300.0,
            "vram_alloc_mb": 300.0,
            "vram_reserved_mb": 600.0,
            "vram_reserved_peak_mb": 750.0,
        },
    )

    with path.open(newline="", encoding="utf-8") as metrics_file:
        reader = csv.DictReader(metrics_file)
        rows = list(reader)

    assert reader.fieldnames == [
        "step",
        "loss",
        "lr",
        "tok_s",
        "vram_mb",
        "vram_alloc_mb",
        "vram_reserved_mb",
        "vram_reserved_peak_mb",
    ]
    assert rows[0] == {
        "step": "0",
        "loss": "1.0",
        "lr": "0.001",
        "tok_s": "42",
        "vram_mb": "256.0",
        "vram_alloc_mb": "",
        "vram_reserved_mb": "",
        "vram_reserved_peak_mb": "",
    }
    assert rows[1]["vram_reserved_mb"] == "600.0"
    assert rows[1]["vram_reserved_peak_mb"] == "750.0"
