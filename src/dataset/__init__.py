"""Backward-compatibility shim — prefer ``src.data`` for new code.

This package is kept for existing scripts that import ``src.dataset.prep``.
The canonical data-preparation code now lives in ``src.data.prep``.
"""

from src.data.prep import download_tiny_shakespeare, ingest_local_file, main

__all__ = [
    "download_tiny_shakespeare",
    "ingest_local_file",
    "main",
]
