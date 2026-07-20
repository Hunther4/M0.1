# 🏛️ M0.1 Architecture Specification

- **Core Engine:** `TrainingEngineV2` built on a Finite State Machine (`INIT -> LOAD -> TRAIN -> VALIDATE -> SAVE -> EVALUATE -> EXPORT -> FINISHED`).
- **Telemetry:** Model-agnostic forward hooks (`enable_hooks=True/False`).
- **Hardware Acceleration:** Native PyTorch HIP/ROCm acceleration for AMD Radeon RX 9060 XT.
