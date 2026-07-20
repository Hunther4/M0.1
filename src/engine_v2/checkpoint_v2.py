"""Canonical Single Checkpoint System v2 with SHA256 Integrity & Automatic Backup Recovery."""

import os
import sys
import json
import hashlib
import threading
import subprocess
import socket
import platform
import shutil
import random
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import torch


class AsyncCheckpointManagerV2:
    """Canonical Single Checkpoint System v2 for TrainingEngine v2."""

    def __init__(self, checkpoint_dir: str) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.canonical_path = self.checkpoint_dir / "checkpoint.pt"
        self.backup_path = self.checkpoint_dir / "checkpoint.previous.pt"
        self.checksum_path = self.checkpoint_dir / "checkpoint.pt.sha256"
        self._write_thread: Optional[threading.Thread] = None

    @staticmethod
    def calculate_sha256(filepath: Path) -> str:
        """Compute SHA256 checksum of a file."""
        hasher = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def capture_environment_metadata(self) -> Dict[str, Any]:
        """Capture complete system, Python, PyTorch, ROCm/CUDA, and Git environment metadata."""
        git_hash = "unknown"
        git_diff = ""
        try:
            git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            git_diff = subprocess.check_output(["git", "diff"], text=True).strip()
        except Exception:
            pass

        return {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "rocm_version": getattr(torch.version, "hip", None),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "git_hash": git_hash,
            "git_diff": git_diff,
        }

    @staticmethod
    def capture_rng_states() -> Dict[str, Any]:
        """Capture Python, NumPy, PyTorch CPU, and CUDA/ROCm RNG states."""
        states = {
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_cpu_rng": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            states["torch_cuda_rng"] = torch.cuda.get_rng_state_all()
        return states

    @staticmethod
    def restore_rng_states(states: Dict[str, Any]) -> None:
        """Restore Python, NumPy, PyTorch CPU, and CUDA/ROCm RNG states."""
        if "python_rng" in states:
            random.setstate(states["python_rng"])
        if "numpy_rng" in states:
            np.random.set_state(states["numpy_rng"])
        if "torch_cpu_rng" in states:
            torch.set_rng_state(states["torch_cpu_rng"])
        if "torch_cuda_rng" in states and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(states["torch_cuda_rng"])

    def save_canonical_async(
        self,
        state_dict: Dict[str, Any],
        callback: Optional[callable] = None,
    ) -> None:
        """Asynchronously save state to canonical checkpoint.pt with backup and SHA256 checksum."""

        def _target():
            tmp_path = self.checkpoint_dir / "checkpoint.pt.tmp"
            torch.save(state_dict, tmp_path)

            # Calculate SHA256 of tmp file
            sha256_val = self.calculate_sha256(tmp_path)

            # Move current canonical checkpoint to backup if it exists
            if self.canonical_path.exists():
                shutil.copy2(self.canonical_path, self.backup_path)

            # Write checksum file
            with open(self.checksum_path, "w", encoding="utf-8") as f:
                f.write(sha256_val)

            # Atomic rename tmp -> canonical
            os.replace(tmp_path, self.canonical_path)

            if callback:
                callback(str(self.canonical_path))

        if self._write_thread is not None and self._write_thread.is_alive():
            self._write_thread.join()

        self._write_thread = threading.Thread(target=_target, daemon=True)
        self._write_thread.start()

    def load_canonical(self) -> Dict[str, Any]:
        """Load canonical checkpoint, verifying SHA256 checksum and falling back to backup if corrupt."""
        self.wait_completion()

        target_file = self.canonical_path
        if not target_file.exists():
            if self.backup_path.exists():
                print("[CHECKPOINT] Canonical missing. Falling back to backup checkpoint.previous.pt")
                target_file = self.backup_path
            else:
                raise FileNotFoundError(f"No checkpoint found in {self.checkpoint_dir}")

        # Verify SHA256 if loading canonical
        if target_file == self.canonical_path and self.checksum_path.exists():
            expected_sha = open(self.checksum_path, "r", encoding="utf-8").read().strip()
            actual_sha = self.calculate_sha256(target_file)
            if expected_sha != actual_sha:
                print("[CORRUPTION DETECTED] SHA256 mismatch! Falling back to checkpoint.previous.pt")
                target_file = self.backup_path

        state = torch.load(target_file, map_location="cpu", weights_only=False)
        print(f"[CHECKPOINT RESTORED] Successfully loaded from {target_file}")
        return state

    def wait_completion(self) -> None:
        """Wait for any active background save thread."""
        if self._write_thread is not None and self._write_thread.is_alive():
            self._write_thread.join()
