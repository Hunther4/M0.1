"""Safely merge checkpoints with identical architectures."""

import argparse
import math
import os
from typing import Any

import torch


_STATE_KEYS = ("model_state_dict", "model_state")
_CONFIG_KEYS = ("config", "model_config")
_MOE_MARKERS = (".ff.experts.", ".ff.shared_experts.", ".ff.gate.")


def _checkpoint_parts(checkpoint: dict[str, Any]) -> tuple[str, str, dict, dict]:
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a dictionary")

    state_keys = [key for key in _STATE_KEYS if key in checkpoint]
    config_keys = [key for key in _CONFIG_KEYS if key in checkpoint]
    if len(state_keys) != 1 or len(config_keys) != 1:
        raise ValueError(
            "Checkpoint must contain exactly one model state key and one config key"
        )

    state_key, config_key = state_keys[0], config_keys[0]
    state, config = checkpoint[state_key], checkpoint[config_key]
    if not isinstance(state, dict) or not isinstance(config, dict):
        raise ValueError("Model state and configuration must both be dictionaries")
    return state_key, config_key, state, config


def _is_moe_parameter(key: str) -> bool:
    return any(marker in key for marker in _MOE_MARKERS)


def _validate_compatible(
    config1: dict, config2: dict, state1: dict, state2: dict
) -> None:
    if config1 != config2:
        differing = sorted(
            key
            for key in config1.keys() | config2.keys()
            if config1.get(key) != config2.get(key)
        )
        raise ValueError(f"Checkpoint configs differ at keys: {differing}")

    keys1, keys2 = set(state1), set(state2)
    if keys1 != keys2:
        missing_from_1 = sorted(keys2 - keys1)
        missing_from_2 = sorted(keys1 - keys2)
        raise ValueError(
            "State dict keys differ; "
            f"missing from checkpoint 1: {missing_from_1}; "
            f"missing from checkpoint 2: {missing_from_2}"
        )

    for key in sorted(keys1):
        tensor1, tensor2 = state1[key], state2[key]
        if not isinstance(tensor1, torch.Tensor) or not isinstance(tensor2, torch.Tensor):
            raise ValueError(f"State value {key!r} is not a tensor in both checkpoints")
        if tensor1.shape != tensor2.shape:
            raise ValueError(
                f"Shape mismatch for {key}: {tuple(tensor1.shape)} vs "
                f"{tuple(tensor2.shape)}"
            )
        if tensor1.dtype != tensor2.dtype:
            raise ValueError(
                f"Dtype mismatch for {key}: {tensor1.dtype} vs {tensor2.dtype}"
            )


def merge_checkpoints(
    path1: str,
    path2: str,
    output_path: str,
    alpha: float = 0.5,
    moe_policy: str = "reject",
) -> None:
    """Merge identical checkpoints, handling MoE parameters explicitly.

    Expert order is permutation-invariant, so interpolating routed experts or
    their gate is not generally meaningful. ``moe_policy`` therefore defaults
    to ``reject``. ``copy-first`` and ``copy-second`` copy every MoE parameter
    as one coherent set from the selected checkpoint.
    """
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be finite and between 0 and 1")
    if moe_policy not in {"reject", "copy-first", "copy-second"}:
        raise ValueError("moe_policy must be reject, copy-first, or copy-second")
    for path in (path1, path2):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint1 = torch.load(path1, map_location="cpu", weights_only=True)
    checkpoint2 = torch.load(path2, map_location="cpu", weights_only=True)
    state_key1, config_key1, state1, config1 = _checkpoint_parts(checkpoint1)
    _, _, state2, config2 = _checkpoint_parts(checkpoint2)
    _validate_compatible(config1, config2, state1, state2)

    moe_keys = sorted(key for key in state1 if _is_moe_parameter(key))
    if moe_keys and moe_policy == "reject":
        raise ValueError(
            "MoE parameters cannot be safely interpolated without expert "
            "alignment; choose --moe-policy copy-first or copy-second"
        )

    merged_state = {}
    for key in sorted(state1):
        tensor1, tensor2 = state1[key], state2[key]
        if _is_moe_parameter(key):
            source = tensor1 if moe_policy == "copy-first" else tensor2
            merged_state[key] = source.clone()
        elif tensor1.is_floating_point() or tensor1.is_complex():
            merged_state[key] = torch.lerp(tensor2, tensor1, alpha)
        else:
            if not torch.equal(tensor1, tensor2):
                raise ValueError(f"Non-floating state differs for {key}")
            merged_state[key] = tensor1.clone()

    merged_checkpoint = {
        config_key1: dict(config1),
        state_key1: merged_state,
        "model_manipulation": {
            "operation": "merge",
            "alpha": alpha,
            "moe_policy": moe_policy,
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.save(merged_checkpoint, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Interpolate compatible checkpoints with explicit MoE handling."
    )
    parser.add_argument("checkpoint1")
    parser.add_argument("checkpoint2")
    parser.add_argument("output")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument(
        "--moe-policy",
        choices=("reject", "copy-first", "copy-second"),
        default="reject",
    )
    arguments = parser.parse_args()
    merge_checkpoints(
        arguments.checkpoint1,
        arguments.checkpoint2,
        arguments.output,
        alpha=arguments.alpha,
        moe_policy=arguments.moe_policy,
    )
