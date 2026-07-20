import os
import sys
import torch

def merge_checkpoints(path1: str, path2: str, output_path: str, alpha: float = 0.5):
    """Merges two checkpoints of the identical architecture using weight interpolation."""
    if not os.path.exists(path1):
        print(f"Error: {path1} does not exist.")
        return
    if not os.path.exists(path2):
        print(f"Error: {path2} does not exist.")
        return

    print(f"Loading checkpoint 1: {path1}")
    ckpt1 = torch.load(path1, map_location="cpu")
    print(f"Loading checkpoint 2: {path2}")
    ckpt2 = torch.load(path2, map_location="cpu")

    config1 = ckpt1["config"]
    config2 = ckpt2["config"]

    # Verify configs match
    for k in config1.keys():
        if config1[k] != config2.get(k):
            print(f"Warning: Config mismatch for key '{k}': {config1[k]} vs {config2.get(k)}")

    sd1 = ckpt1["model_state_dict"]
    sd2 = ckpt2["model_state_dict"]

    merged_sd = {}
    print(f"Merging state dicts with alpha={alpha} (weight_merged = alpha*ckpt1 + (1-alpha)*ckpt2)...")
    
    for key in sd1.keys():
        if key not in sd2:
            print(f"Warning: Key {key} only present in checkpoint 1. Copying directly.")
            merged_sd[key] = sd1[key].clone()
            continue
        
        t1 = sd1[key]
        t2 = sd2[key]
        if t1.shape != t2.shape:
            print(f"Error: Shape mismatch for {key}: {t1.shape} vs {t2.shape}. Cannot merge.")
            return

        # Perform interpolation
        if t1.is_floating_point():
            merged_sd[key] = alpha * t1 + (1 - alpha) * t2
        else:
            # For non-floating point (like integers/indices if any), copy from ckpt1
            merged_sd[key] = t1.clone()

    for key in sd2.keys():
        if key not in sd1:
            print(f"Warning: Key {key} only present in checkpoint 2. Copying directly.")
            merged_sd[key] = sd2[key].clone()

    # Save the merged checkpoint
    merged_ckpt = {
        "config": config1,
        "model_state_dict": merged_sd
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(merged_ckpt, output_path)
    print(f"Successfully saved merged checkpoint to: {output_path}")

if __name__ == "__main__":
    # Default merge for the smaller resilient & uncensored models
    c1 = "checkpoints/m01_resilient.pt"
    c2 = "checkpoints/m01_uncensored.pt"
    out = "checkpoints/m01_resilient_uncensored_merged.pt"
    merge_checkpoints(c1, c2, out, alpha=0.5)
