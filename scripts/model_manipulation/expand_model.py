import os
import sys
import torch

def expand_model_depth(checkpoint_path: str, output_path: str, new_layers: int):
    """Expands model depth by duplicating/interleaving trained layer weights (depth up-scaling)."""
    if not os.path.exists(checkpoint_path):
        print(f"Error: {checkpoint_path} does not exist.")
        return

    print(f"Loading base checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    config = ckpt["config"]
    old_layers = config["n_layers"]

    if new_layers <= old_layers:
        print(f"Error: New layers count ({new_layers}) must be greater than current layers count ({old_layers}).")
        return

    print(f"Expanding architecture depth: {old_layers} layers -> {new_layers} layers")
    sd = ckpt["model_state_dict"]
    new_sd = {}

    # Copy non-layer parameters (embedding, final norm, output head etc.) directly
    for key, value in sd.items():
        if not key.startswith("blocks."):
            new_sd[key] = value.clone()

    # Determine mapping from new layer indices to old layer indices
    # We use progressive layer stacking mapping: old_idx = int(new_idx * old_layers / new_layers)
    mapping = [int(i * old_layers / new_layers) for i in range(new_layers)]
    print(f"Layer mapping (new_layer_idx -> source_old_layer_idx):")
    for new_idx, old_idx in enumerate(mapping):
        print(f"  Layer {new_idx} <- Layer {old_idx}")

    # Copy and map layer weights
    for new_idx, old_idx in enumerate(mapping):
        prefix_old = f"blocks.{old_idx}."
        prefix_new = f"blocks.{new_idx}."
        
        # Find all keys belonging to the source old layer
        for key, value in sd.items():
            if key.startswith(prefix_old):
                suffix = key[len(prefix_old):]
                new_key = prefix_new + suffix
                new_sd[new_key] = value.clone()

    # Create new config with expanded layers
    new_config = config.copy()
    new_config["n_layers"] = new_layers

    new_ckpt = {
        "config": new_config,
        "model_state_dict": new_sd
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(new_ckpt, output_path)
    print(f"Successfully saved expanded checkpoint ({new_layers} layers) to: {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Expand model depth by stacking layers.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/m01_hardened_final.pt", help="Source checkpoint path.")
    parser.add_argument("--output", type=str, default="checkpoints/m01_extended_14layers.pt", help="Output checkpoint path.")
    parser.add_argument("--layers", type=int, default=14, help="Target number of layers.")
    args = parser.parse_args()

    expand_model_depth(args.checkpoint, args.output, args.layers)
