"""Shared utilities for evaluation."""
import json
import logging
from datetime import datetime
from pathlib import Path


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure logging for evaluation runs."""
    logger = logging.getLogger("eval")
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)
    return logger


def get_timestamp() -> str:
    """Return ISO timestamp string."""
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def save_results(results: dict, checkpoint_path: str) -> Path:
    """Save evaluation results to timestamped JSON file.

    Args:
        results: Evaluation results dictionary with metrics and qa scores
        checkpoint_path: Path to the evaluated checkpoint

    Returns:
        Path to the saved JSON file
    """
    artifacts_dir = Path("artifacts/evals")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    timestamp = get_timestamp()
    filename = f"results_{timestamp}.json"
    filepath = artifacts_dir / filename

    output = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "checkpoint": checkpoint_path,
        **results
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    return filepath


def load_results(filepath: str | Path) -> dict:
    """Load evaluation results from JSON file."""
    with open(filepath, encoding="utf-8") as f:
        return json.load(f)