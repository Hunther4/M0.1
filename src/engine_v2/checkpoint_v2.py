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
import pickle
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import torch

from src.transformer.config import M01Config


def _normalize_legacy_config_metadata(state: Any) -> Dict[str, Any]:
    """Convert explicitly supported legacy config objects to plain dictionaries."""
    if not isinstance(state, dict):
        raise ValueError("Checkpoint root must be a dictionary")

    normalized = dict(state)
    for key in ("config", "model_config"):
        value = normalized.get(key)
        if type(value) is M01Config:
            normalized[key] = {
                field.name: getattr(value, field.name)
                for field in fields(M01Config)
                if hasattr(value, field.name)
            }
    return normalized


def safe_load_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    """Load tensors safely, retrying with progressively broader safe globals.

    Checkpoints saved with PyTorch <2.6 may contain serialized metadata types
    (TorchVersion, NumPy globals) that are restricted by ``weights_only=True``
    in PyTorch 2.6+.  This function catches those errors and re-adds the missing
    types to a scoped safe allowlist, retrying until the load succeeds or an
    unrecognised error is hit. Legacy ``M01Config`` objects are immediately
    converted to the current dictionary format.
    """
    known_safe: list[Any] = []
    # Pre-resolve NumPy globals once (avoid repeated getattr in the loop).
    _numpy_core = getattr(np, "_core", getattr(np, "core", None))
    _numpy_reconstruct = getattr(getattr(_numpy_core, "multiarray", None), "_reconstruct", None)
    _numpy_scalar_repr = getattr(getattr(_numpy_core, "multiarray", None), "scalar_repr", None)

    for attempt in range(10):
        try:
            with torch.serialization.safe_globals(known_safe):
                state = torch.load(path, map_location=map_location, weights_only=True)
            return _normalize_legacy_config_metadata(state)
        except (pickle.UnpicklingError, RuntimeError) as exc:
            msg = str(exc).lower()
            candidates: list[Any] = []

            if "src.transformer.config.m01config" in msg:
                candidates.append(M01Config)

            if "torch_version" in msg or "torchversion" in msg:
                candidates.append(torch.torch_version.TorchVersion)

            if "numpy" in msg:
                for obj in (_numpy_reconstruct, np.ndarray, np.dtype, _numpy_scalar_repr):
                    if obj is not None:
                        candidates.append(obj)
                dtypes = getattr(np, "dtypes", None)
                if dtypes is not None and hasattr(dtypes, "UInt32DType"):
                    candidates.append(dtypes.UInt32DType)

            added = [candidate for candidate in candidates if candidate not in known_safe]
            if not added:
                raise  # Unknown or already-allowed type: preserve the safe-load failure.

            known_safe.extend(added)

    raise RuntimeError(
        f"Failed to load checkpoint after 10 attempts: {path}"
    )


def normalize_checkpoint_state(state: Dict[str, Any], *, require_architecture: bool = False) -> Dict[str, Any]:
    """Normalize V1/V2 checkpoint aliases at the deserialization boundary."""
    state = _normalize_legacy_config_metadata(state)
    aliases = {
        "model_state": ("model_state", "model_state_dict"),
        "model_config": ("model_config", "config"),
        "optimizer_state": ("optimizer_state", "optimizer_state_dict"),
        "scheduler_state": ("scheduler_state", "scheduler_state_dict"),
    }
    normalized = dict(state)
    for canonical, keys in aliases.items():
        if canonical not in normalized:
            for key in keys:
                if key in state:
                    normalized[canonical] = state[key]
                    break
    if "model_state" not in normalized:
        raise ValueError("Checkpoint is missing model weights (model_state/model_state_dict)")
    if require_architecture and not isinstance(normalized.get("model_config"), dict):
        raise ValueError(
            "Checkpoint is missing architecture metadata (config/model_config); "
            "cannot safely construct the model."
        )
    return normalized


class AsyncCheckpointManagerV2:
    """Canonical Single Checkpoint System v2 for TrainingEngine v2."""

    def __init__(self, checkpoint_dir: str) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.canonical_path = self.checkpoint_dir / "checkpoint.pt"
        self.backup_path = self.checkpoint_dir / "checkpoint.previous.pt"
        self.checksum_path = self.checkpoint_dir / "checkpoint.pt.sha256"
        self._write_thread: Optional[threading.Thread] = None
        self._write_exception: Optional[BaseException] = None

    @staticmethod
    def calculate_sha256(filepath: Path) -> str:
        """Compute SHA256 checksum of a file."""
        hasher = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    @staticmethod
    def calculate_manifest_sha256(filepaths: list[Path]) -> str:
        """Hash an ordered file manifest, including names and file contents."""
        hasher = hashlib.sha256()
        for filepath in sorted((Path(path) for path in filepaths), key=lambda path: path.as_posix()):
            hasher.update(filepath.name.encode("utf-8"))
            with open(filepath, "rb") as source:
                while chunk := source.read(65536):
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
            "torch_version": str(torch.__version__),
            "cuda_available": torch.cuda.is_available(),
            "rocm_version": getattr(torch.version, "hip", None),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "git_hash": git_hash,
            "git_diff": git_diff,
        }

    @staticmethod
    def capture_rng_states() -> Dict[str, Any]:
        """Capture Python, NumPy, PyTorch CPU, and CUDA/ROCm RNG states."""
        numpy_state = np.random.get_state()
        states = {
            "python_rng": random.getstate(),
            "torch_cpu_rng": torch.get_rng_state(),
        }
        states["numpy_rng"] = {
            "bit_generator": numpy_state[0],
            "state": numpy_state[1].tolist(),
            "pos": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
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
            numpy_rng = states["numpy_rng"]
            if isinstance(numpy_rng, dict):
                np.random.set_state(
                    (
                        numpy_rng["bit_generator"],
                        np.array(numpy_rng["state"], dtype=np.uint32),
                        numpy_rng["pos"],
                        numpy_rng["has_gauss"],
                        numpy_rng["cached_gaussian"],
                    )
                )
            else:
                np.random.set_state(numpy_rng)
        if "torch_cpu_rng" in states:
            torch.set_rng_state(states["torch_cpu_rng"])
        if "torch_cuda_rng" in states and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(states["torch_cuda_rng"])

    def save_canonical_async(
        self,
        state_dict: Dict[str, Any],
        callback: Optional[callable] = None,
    ) -> None:
        """Asynchronously save state to canonical checkpoint.pt with backup and SHA256 checksum.

        Moves tensors to CPU before launching the background writer so CUDA D2H
        transfers cannot race with the training loop.
        """

        def _to_cpu(value: Any) -> Any:
            if isinstance(value, torch.Tensor):
                return value.detach().cpu()
            if isinstance(value, dict):
                return {key: _to_cpu(item) for key, item in value.items()}
            if isinstance(value, list):
                return [_to_cpu(item) for item in value]
            if isinstance(value, tuple):
                return tuple(_to_cpu(item) for item in value)
            return value

        cpu_state = _to_cpu(state_dict)

        def _target():
            tmp_path = self.checkpoint_dir / "checkpoint.pt.tmp"
            try:
                torch.save(cpu_state, tmp_path)
                sha256_val = self.calculate_sha256(tmp_path)
                if self.canonical_path.exists():
                    shutil.copy2(self.canonical_path, self.backup_path)
                    if self.checksum_path.exists():
                        shutil.copy2(self.checksum_path, self.checkpoint_dir / "checkpoint.previous.pt.sha256")
                with open(self.checksum_path, "w", encoding="utf-8") as f:
                    f.write(sha256_val)
                os.replace(tmp_path, self.canonical_path)
                if callback:
                    callback(str(self.canonical_path))
            except BaseException as exc:
                self._write_exception = exc

        if self._write_thread is not None:
            self.wait_completion()

        self._write_thread = threading.Thread(target=_target, daemon=True)
        self._write_exception = None
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

        candidates = []
        if self.canonical_path.exists():
            candidates.append((self.canonical_path, self.checksum_path))
        if self.backup_path.exists():
            candidates.append((self.backup_path, self.checkpoint_dir / "checkpoint.previous.pt.sha256"))
        errors = []
        for candidate, sidecar in candidates:
            if not sidecar.exists():
                errors.append(f"{candidate.name} checksum sidecar is missing")
                continue
            expected_sha = sidecar.read_text(encoding="utf-8").strip()
            actual_sha = self.calculate_sha256(candidate)
            if expected_sha != actual_sha:
                errors.append(f"{candidate.name} checksum mismatch")
                continue
            target_file = candidate
            break
        else:
            detail = "; ".join(errors) or "no checkpoint candidates"
            raise IOError(f"No checkpoint passed checksum validation: {detail}")

        state = safe_load_checkpoint(target_file)
        state = normalize_checkpoint_state(state)
        print(f"[CHECKPOINT RESTORED] Successfully loaded from {target_file}")
        return state

    def wait_completion(self) -> None:
        """Wait for any active background save thread."""
        if self._write_thread is not None and self._write_thread.is_alive():
            self._write_thread.join()
        if self._write_exception is not None:
            exc = self._write_exception
            self._write_exception = None
            raise RuntimeError("Background checkpoint save failed") from exc
