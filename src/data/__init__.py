"""Data preparation and dataset loading utilities for M0.1.

This package unifies data functionality that previously lived in
``src.dataset`` (prep) and ``src.training.dataset`` (datasets).

Modules:
    prep: Download/ingest raw text corpora.
"""

from .prep import download_tiny_shakespeare, ingest_local_file, main

__all__ = [
    "download_tiny_shakespeare",
    "ingest_local_file",
    "main",
]
