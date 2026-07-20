# 🚀 M0.1 Training System

- **Automatic Recovery:** Catches NaNs/Infs/Overflows, rolls back to previous checkpoint (`checkpoint.previous.pt`), halves LR, flushes VRAM cache, and resumes training automatically.
- **Graceful Shutdown:** Intercepts SIGINT/SIGTERM, saves canonical `checkpoint.pt`, flushes loggers, and exports profiler.
