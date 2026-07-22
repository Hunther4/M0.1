"""Focused tests for safe M01Config checkpoint compatibility."""

import pickle

import pytest
import torch

from src.engine_v2.checkpoint_v2 import (
    normalize_checkpoint_state,
    safe_load_checkpoint,
)
from src.transformer.config import M01Config
from src.training.checkpoint import config_to_dict, load_checkpoint


class UnsupportedCheckpointGlobal:
    """Pickleable sentinel that must never be added to the safe allowlist."""


def _tiny_config() -> M01Config:
    return M01Config(
        vocab_size=64,
        context_length=16,
        d_model=16,
        n_heads=2,
        d_ff=32,
        n_layers=1,
        num_experts=1,
        num_shared_experts=0,
        moe_top_k=1,
        use_mla=False,
        num_dense_layers=1,
    )


def test_safe_loader_preserves_current_dict_format(tmp_path):
    config = _tiny_config()
    path = tmp_path / "current.pt"
    torch.save({"model_state": {}, "model_config": config_to_dict(config)}, path)

    loaded = safe_load_checkpoint(path)

    assert type(loaded["model_config"]) is dict
    assert loaded["model_config"] == config_to_dict(config)


def test_safe_loader_explicitly_converts_legacy_m01config(tmp_path):
    config = _tiny_config()
    path = tmp_path / "legacy.pt"
    torch.save({"model_state": {}, "model_config": config}, path)
    safe_globals_before = list(torch.serialization.get_safe_globals())

    loaded = safe_load_checkpoint(path)

    assert type(loaded["model_config"]) is dict
    assert loaded["model_config"] == config_to_dict(config)
    assert set(torch.serialization.get_safe_globals()) == set(safe_globals_before)


def test_safe_loader_allows_legacy_config_with_missing_new_fields(tmp_path):
    config = _tiny_config()
    del config.attention_backend
    path = tmp_path / "legacy_missing_field.pt"
    torch.save({"model_state": {}, "model_config": config}, path)

    loaded = safe_load_checkpoint(path)

    assert loaded["model_config"]["attention_backend"] == "auto"
    reconstructed = M01Config(**loaded["model_config"])
    assert reconstructed.attention_backend == "auto"


def test_training_loader_accepts_legacy_config_object(tmp_path):
    from src.model.lm import TransformerLM

    config = _tiny_config()
    model = TransformerLM(config)
    path = tmp_path / "legacy_training.pt"
    torch.save(
        {"model_state_dict": model.state_dict(), "config": config},
        path,
    )

    loaded_model, loaded_config = load_checkpoint(path)

    assert config_to_dict(loaded_config) == config_to_dict(config)
    assert torch.equal(
        loaded_model.embedding.embedding.weight,
        model.embedding.embedding.weight,
    )


def test_normalization_converts_legacy_config_before_aliasing():
    config = _tiny_config()

    normalized = normalize_checkpoint_state(
        {"model_state_dict": {}, "config": config},
        require_architecture=True,
    )

    assert normalized["model_config"] == config_to_dict(config)
    assert type(normalized["model_config"]) is dict


def test_safe_loader_does_not_allow_unrelated_pickle_globals(tmp_path):
    path = tmp_path / "unsupported.pt"
    torch.save(
        {"model_state": {}, "config": UnsupportedCheckpointGlobal()},
        path,
    )

    with pytest.raises((pickle.UnpicklingError, RuntimeError)):
        safe_load_checkpoint(path)
