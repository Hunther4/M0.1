"""Regression tests for the highest-priority audit fixes."""

import pickle
import threading

import numpy as np
import pytest
import torch

from src.engine_v2.checkpoint_v2 import (
    AsyncCheckpointManagerV2,
    normalize_checkpoint_state,
    safe_load_checkpoint,
)
from src.inference.sampling import sample
from src.transformer.kv_cache import KVCache


def test_checkpoint_normalizes_v1_aliases():
    state = normalize_checkpoint_state(
        {
            "model_state_dict": {"weight": torch.ones(1)},
            "config": {"vocab_size": 16384},
            "optimizer_state_dict": {},
            "scheduler_state_dict": {},
        },
        require_architecture=True,
    )
    assert state["model_state"] == {"weight": torch.ones(1)}
    assert state["model_config"]["vocab_size"] == 16384
    assert state["optimizer_state"] == {}


def test_checkpoint_rejects_missing_architecture_metadata():
    with pytest.raises(ValueError, match="architecture metadata"):
        normalize_checkpoint_state({"model_state": {}}, require_architecture=True)


def test_checkpoint_rejects_malicious_pickle(tmp_path):
    class Malicious:
        def __reduce__(self):
            return (eval, ("1 + 1",))

    path = tmp_path / "malicious.pt"
    torch.save({"model_state": {"bad": Malicious()}}, path)
    with pytest.raises(Exception):
        safe_load_checkpoint(path)


def test_legacy_numpy_rng_tuple_loads_safely(tmp_path):
    manager = AsyncCheckpointManagerV2(str(tmp_path))
    manager.save_canonical_async({"model_state": {}, "numpy_rng": np.random.get_state()})
    manager.wait_completion()
    loaded = manager.load_canonical()
    manager.restore_rng_states(loaded)
    assert isinstance(loaded["numpy_rng"], tuple)


def test_async_checkpoint_moves_nested_tensors_to_cpu_before_threading(tmp_path, monkeypatch):
    """Checkpoint tensor snapshots must complete before the writer thread starts."""
    manager = AsyncCheckpointManagerV2(str(tmp_path))
    caller_thread = threading.current_thread()
    transfer_threads = []
    save_threads = []
    original_detach = torch.Tensor.detach
    original_cpu = torch.Tensor.cpu
    original_save = torch.save

    def track_detach(tensor):
        transfer_threads.append(("detach", threading.current_thread()))
        return original_detach(tensor)

    def track_cpu(tensor):
        transfer_threads.append(("cpu", threading.current_thread()))
        return original_cpu(tensor)

    def track_save(state, path):
        save_threads.append(threading.current_thread())
        return original_save(state, path)

    monkeypatch.setattr(torch.Tensor, "detach", track_detach)
    monkeypatch.setattr(torch.Tensor, "cpu", track_cpu)
    monkeypatch.setattr(torch, "save", track_save)

    manager.save_canonical_async(
        {
            "model_state": {"weight": torch.tensor([1.0])},
            "rng_states": {"torch_cuda_rng": [torch.tensor([2], dtype=torch.uint8)]},
        }
    )
    manager.wait_completion()

    assert transfer_threads == [
        ("detach", caller_thread),
        ("cpu", caller_thread),
        ("detach", caller_thread),
        ("cpu", caller_thread),
    ]
    assert save_threads and save_threads[0] is not caller_thread


def test_checksum_corruption_uses_verified_backup(tmp_path):
    manager = AsyncCheckpointManagerV2(str(tmp_path))
    manager.save_canonical_async({"model_state": {"v": torch.tensor(1)}})
    manager.wait_completion()
    manager.save_canonical_async({"model_state": {"v": torch.tensor(2)}})
    manager.wait_completion()
    manager.canonical_path.write_bytes(b"corrupt")
    loaded = manager.load_canonical()
    assert loaded["model_state"]["v"].item() == 1


def test_missing_canonical_sidecar_falls_back_to_verified_backup(tmp_path):
    manager = AsyncCheckpointManagerV2(str(tmp_path))
    manager.save_canonical_async({"model_state": {"v": torch.tensor(1)}})
    manager.wait_completion()
    manager.save_canonical_async({"model_state": {"v": torch.tensor(2)}})
    manager.wait_completion()
    manager.checksum_path.unlink()

    loaded = manager.load_canonical()
    assert loaded["model_state"]["v"].item() == 1


def test_missing_backup_sidecar_fails_closed(tmp_path):
    manager = AsyncCheckpointManagerV2(str(tmp_path))
    manager.save_canonical_async({"model_state": {"v": torch.tensor(1)}})
    manager.wait_completion()
    manager.save_canonical_async({"model_state": {"v": torch.tensor(2)}})
    manager.wait_completion()
    manager.canonical_path.write_bytes(b"corrupt")
    (tmp_path / "checkpoint.previous.pt.sha256").unlink()

    with pytest.raises(IOError, match="checksum"):
        manager.load_canonical()


def test_top_p_sampling_has_finite_normalized_distribution(monkeypatch):
    captured = {}

    def capture(probs, count):
        captured["probs"] = probs
        return torch.tensor([0])

    monkeypatch.setattr(torch, "multinomial", capture)
    sample(torch.tensor([4.0, 3.0, 2.0, 1.0]), top_p=0.5)
    assert torch.isfinite(captured["probs"]).all()
    assert torch.allclose(captured["probs"].sum(), torch.tensor(1.0))


def test_kv_cache_rejects_dtype_change_after_cache():
    cache = KVCache(4, 1, 2, dtype=torch.float32)
    cache.append(torch.zeros(1, 1, 1, 2), torch.zeros(1, 1, 1, 2))
    with pytest.raises(ValueError, match="dtype"):
        cache.append(
            torch.zeros(1, 1, 1, 2, dtype=torch.float16),
            torch.zeros(1, 1, 1, 2, dtype=torch.float16),
        )
