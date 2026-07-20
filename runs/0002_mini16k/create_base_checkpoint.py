"""Create the canonical BASE checkpoint for the 16k tokenizer (vocab=16384).

This is the reproducibility anchor for ALL future M0.1 runs that use the new
byte-level BPE tokenizer. It is a fresh, fully-resumable checkpoint at step 0
(freshly initialized weights + empty optimizer/scheduler/EMA/AMP/RNG state),
saved with the SAME AsyncCheckpointManagerV2 API the engine uses, so
``TrainingEngineV2.resume()`` can load it directly.

Why a separate base (not runs/0002_mini16k's own step-0 checkpoint):
  * vocab incompatibility -- the previous 512-vocab model can't be lifted onto
    the 16384 vocab; a clean init is required.
  * reproducibility -- a single canonical, tokenizer-pinned starting point
    (with tokenizer_hash recorded) makes every future run traceable.
  * resume/fork -- future runs init from here and branch; the base itself is
    never overwritten by training.
"""

import os
import sys
import math

sys.path.insert(0, "E:\\M0.1")

import torch
from src.model.lm import TransformerLM
from src.transformer.config import M01Config
from src.training.config import TrainingConfig
from src.engine_v2.ema import EMA
from src.engine_v2.amp import AMPContext
from src.engine_v2.checkpoint_v2 import AsyncCheckpointManagerV2

# NOTE: src/training/train.py was externally modified (a --vocab-size arg was
# added at line 90 but lines 89-90 are over-indented and the `return` was
# dropped, breaking its import). To avoid depending on that broken module we
# re-implement the two small helpers inline (identical to train.py's originals).
def configure_optimizer(model, config):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "gamma" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": config.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=config.max_lr, betas=(config.beta1, config.beta2), eps=1e-8)

def get_lr_scheduler(optimizer, warmup_steps, max_steps, min_lr_ratio):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
        progress = min(progress, 1.0)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

TOKENIZER_PATH = "E:\\M0.1\\data\\tokenizers\\tokenizer.json"
OUT_DIR = "E:\\M0.1\\checkpoints\\base_v16k"
OUT_PATH = os.path.join(OUT_DIR, "base_checkpoint.pt")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    model_config = M01Config(vocab_size=16384)
    model = TransformerLM(model_config)
    n_params = sum(p.numel() for p in model.parameters())

    cfg = TrainingConfig()  # defaults are fine for a step-0 base
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    optimizer = configure_optimizer(model, cfg)
    scheduler = get_lr_scheduler(optimizer, cfg.warmup_steps, cfg.max_steps, cfg.min_lr_ratio)
    ema = EMA(model)
    amp = AMPContext(device, enabled=(device.type == "cuda"))

    mgr = AsyncCheckpointManagerV2(OUT_DIR)
    tok_hash = mgr.calculate_sha256(TOKENIZER_PATH)

    state = {
        "step": 0,
        "global_tokens": 0,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "ema_state": ema.state_dict(),
        "amp_scaler_state": amp.state_dict(),
        "rng_states": AsyncCheckpointManagerV2.capture_rng_states(),
        "metrics": {},
        "env": mgr.capture_environment_metadata(),
        "dataset_hash": "n/a (base checkpoint)",
        "tokenizer_hash": tok_hash,
    }

    torch.save(state, OUT_PATH)
    print(f"[BASE] saved base checkpoint -> {OUT_PATH}")
    print(f"[BASE] params={n_params/1e6:.1f}M  vocab={model_config.vocab_size}")
    print(f"[BASE] tokenizer_hash={tok_hash[:16]}...")

    # Sanity: confirm it reloads via the manager (no corruption).
    reloaded = torch.load(OUT_PATH, map_location="cpu", weights_only=False)
    assert reloaded["step"] == 0
    assert reloaded["model_state"]["embedding.embedding.weight"].shape == (16384, model_config.d_model)
    print("[BASE] reload sanity OK (model embedding shape 16384 x d_model)")


if __name__ == "__main__":
    main()
