"""Expand checkpoint depth with architecture-aware layer mapping."""

import argparse
import math
import os
import re
from typing import Any

import torch


_STATE_KEYS = ("model_state_dict", "model_state")
_CONFIG_KEYS = ("config", "model_config")
_LAYER_KEY = re.compile(r"^blocks\.(\d+)\.(.+)$")
_RESIDUAL_OUTPUT_SUFFIXES = ("attn.W_o.weight", "down_proj.weight")


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


def _validate_layers(state: dict, old_layers: int) -> dict[int, dict[str, torch.Tensor]]:
    layers: dict[int, dict[str, torch.Tensor]] = {index: {} for index in range(old_layers)}
    for key, value in state.items():
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"State value {key!r} is not a tensor")
        match = _LAYER_KEY.match(key)
        if match is None:
            continue
        layer_index = int(match.group(1))
        if layer_index >= old_layers:
            raise ValueError(
                f"State contains layer {layer_index}, outside configured depth {old_layers}"
            )
        layers[layer_index][match.group(2)] = value
    missing = [index for index, values in layers.items() if not values]
    if missing:
        raise ValueError(f"Checkpoint has no parameters for configured layers: {missing}")
    return layers


def _progressive_mapping(start: int, old_count: int, new_count: int) -> list[int]:
    if new_count == 0:
        return []
    if old_count == 0:
        raise ValueError("Cannot create layers of a type absent from the source model")
    return [start + int(index * old_count / new_count) for index in range(new_count)]


def _layer_mapping(old_layers: int, new_layers: int, dense_layers: int) -> list[int]:
    if not 0 <= dense_layers <= old_layers:
        raise ValueError("num_dense_layers must be between 0 and n_layers")

    # Keep the dense/MoE boundary fixed. New depth is allocated to the trailing
    # region so a dense source block is never copied into an MoE target slot.
    new_dense_layers = dense_layers if dense_layers < old_layers else new_layers
    dense_mapping = _progressive_mapping(0, dense_layers, new_dense_layers)
    old_moe_layers = old_layers - dense_layers
    new_moe_layers = new_layers - new_dense_layers
    moe_mapping = _progressive_mapping(dense_layers, old_moe_layers, new_moe_layers)
    return dense_mapping + moe_mapping


def _is_residual_output(suffix: str) -> bool:
    return suffix.endswith(_RESIDUAL_OUTPUT_SUFFIXES)


def expand_model_depth(
    checkpoint_path: str,
    output_path: str,
    new_layers: int,
    residual_strategy: str = "depth-ratio",
) -> None:
    """Expand model depth while preserving load-compatible state structure.

    ``depth-ratio`` scales attention output and FFN/MoE down projections by
    ``sqrt(old_layers / new_layers)``. Under the usual independent-residual
    approximation, this keeps aggregate residual variance stable after depth
    growth. ``none`` is available for exact weight copying when desired.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if residual_strategy not in {"depth-ratio", "none"}:
        raise ValueError("residual_strategy must be depth-ratio or none")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_key, config_key, state, config = _checkpoint_parts(checkpoint)
    if not isinstance(config.get("n_layers"), int) or config["n_layers"] <= 0:
        raise ValueError("Config n_layers must be a positive integer")
    old_layers = config["n_layers"]
    if not isinstance(new_layers, int) or new_layers <= old_layers:
        raise ValueError(
            f"new_layers ({new_layers}) must be greater than old depth ({old_layers})"
        )

    dense_layers = config.get("num_dense_layers", 0)
    if not isinstance(dense_layers, int):
        raise ValueError("Config num_dense_layers must be an integer")
    layers = _validate_layers(state, old_layers)
    mapping = _layer_mapping(old_layers, new_layers, dense_layers)
    residual_scale = (
        math.sqrt(old_layers / new_layers)
        if residual_strategy == "depth-ratio"
        else 1.0
    )

    expanded_state = {
        key: value.clone() for key, value in state.items() if _LAYER_KEY.match(key) is None
    }
    for new_index, old_index in enumerate(mapping):
        for suffix, value in layers[old_index].items():
            copied = value.clone()
            if _is_residual_output(suffix) and copied.is_floating_point():
                copied.mul_(residual_scale)
            expanded_state[f"blocks.{new_index}.{suffix}"] = copied

    expanded_config = dict(config)
    expanded_config["n_layers"] = new_layers
    expanded_checkpoint = {
        config_key: expanded_config,
        state_key: expanded_state,
        "model_manipulation": {
            "operation": "expand-depth",
            "source_layers": old_layers,
            "target_layers": new_layers,
            "layer_mapping": mapping,
            "residual_strategy": residual_strategy,
            "residual_scale": residual_scale,
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.save(expanded_checkpoint, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Expand model depth safely.")
    parser.add_argument("--checkpoint", required=True, help="Source checkpoint path")
    parser.add_argument("--output", required=True, help="Output checkpoint path")
    parser.add_argument("--layers", type=int, required=True, help="Target layer count")
    parser.add_argument(
        "--residual-strategy",
        choices=("depth-ratio", "none"),
        default="depth-ratio",
    )
    arguments = parser.parse_args()
    expand_model_depth(
        arguments.checkpoint,
        arguments.output,
        arguments.layers,
        residual_strategy=arguments.residual_strategy,
    )
