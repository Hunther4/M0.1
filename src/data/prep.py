"""Data preparation utilities for M0.1.

Provides CLI and programmatic access to dataset preparation:
- Download TinyShakespeare corpus
- Ingest a local UTF-8 text file into the project data directory

This module was migrated from ``src.dataset.prep`` (still available as a
backward-compatible alias).
"""

import argparse
import os
from typing import List, Optional

import requests
from tqdm import tqdm

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def download_tiny_shakespeare(output_path: str) -> None:
    """Download TinyShakespeare to *output_path* using chunked stream writing."""
    dir_name = os.path.dirname(os.path.abspath(output_path))
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    print(f"Downloading TinyShakespeare from {TINY_SHAKESPEARE_URL} to {output_path}...")
    response = requests.get(TINY_SHAKESPEARE_URL, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    chunk_size = 8192

    with open(output_path, "wb") as f:
        with tqdm(
            total=total_size, unit="B", unit_scale=True, desc="TinyShakespeare"
        ) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    print("Download completed successfully.")


def ingest_local_file(input_path: str, output_path: str) -> None:
    """Validate *input_path* as UTF-8 and copy it to *output_path*.

    Raises:
        FileNotFoundError: If *input_path* does not exist.
        ValueError: If the input file is not valid UTF-8.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    try:
        with open(input_path, "rb") as f:
            content = f.read()
        decoded_content = content.decode("utf-8")
    except UnicodeDecodeError as e:
        raise ValueError(f"Input file is not valid UTF-8: {e}") from e

    dir_name = os.path.dirname(os.path.abspath(output_path))
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(decoded_content)

    print(f"Ingested local file from {input_path} to {output_path} successfully.")


def main(argv: Optional[List[str]] = None) -> None:
    """Main CLI entry point for data preparation.

    Examples:
        python -m src.data.prep --download
        python -m src.data.prep -i path/to/corpus.txt -o data/raw_text/corpus.txt
    """
    parser = argparse.ArgumentParser(
        description="Data preparation utility for M0.1."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--download", action="store_true", help="Download TinyShakespeare dataset."
    )
    group.add_argument(
        "--input", "-i", type=str, help="Path to a local input file to ingest."
    )

    parser.add_argument(
        "--output", "-o", type=str, help="Path to write the output file."
    )

    args = parser.parse_args(argv)

    if args.download:
        output_path = (
            args.output if args.output else os.path.join("data", "tinyshakespeare.txt")
        )
        download_tiny_shakespeare(output_path)
    elif args.input:
        if not args.output:
            parser.error("--output is required when --input is provided.")
        ingest_local_file(args.input, args.output)


if __name__ == "__main__":
    main()
