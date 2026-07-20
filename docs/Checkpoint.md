# 💾 Canonical Single Checkpoint System

- **Canonical File:** `checkpoint.pt`
- **Backup File:** `checkpoint.previous.pt`
- **Integrity:** Computes SHA256 checksum saved to `checkpoint.pt.sha256`. Atomic `checkpoint.pt.tmp` -> `rename()` overwrite.
- **State Preserved:** Model, Optimizer, Scheduler, EMA, GradScaler, TrainerState, MetricRegistry, LossPipeline, Python/NumPy/Torch CPU/ROCm RNGs.
- **Resume API:** `engine.resume("checkpoint.pt")`
