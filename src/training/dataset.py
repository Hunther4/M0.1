"""TinyShakespeare dataset for training M0.1.

Provides a sliding-window PyTorch Dataset over the TinyShakespeare
corpus tokenized with the trained BPE tokenizer.
"""

import os

import numpy as np
import torch
from torch import LongTensor

from src.tokenizer.bpe import Tokenizer
from src.training.config import TrainingConfig


class TinyShakespeareDataset:
    """Sliding-window dataset over the TinyShakespeare corpus.

    Loads the canonical BPE tokenizer from ``data/tokenizers/tokenizer.json``, tokenizes
    ``data_dir/tinyshakespeare.txt`` once at init, and provides overlapping
    ``(input, target)`` pairs of length ``seq_len`` for autoregressive
    language model training.

    Compatible with ``DataLoader(dataset, batch_size=N, num_workers=0)``.

    Args:
        config: TrainingConfig providing seq_len and data_dir.

    Attributes:
        tokens: LongTensor of all token IDs (N,).
        seq_len: Sliding window length.
    """

    def __init__(self, config: TrainingConfig) -> None:
        self.seq_len = config.seq_len
        data_dir = config.data_dir

        # All datasets use the single canonical 16K tokenizer.
        tokenizer = Tokenizer()
        tokenizer_path = os.path.join("data", "tokenizers", "tokenizer.json")
        tokenizer.load(tokenizer_path)

        # Tokenize the full corpus once at init (check subfolder first, fallback to root)
        text_paths = [
            os.path.join(data_dir, "raw_text", "spanish_pretrain.txt"),
            os.path.join(data_dir, "spanish_pretrain.txt"),
            os.path.join(data_dir, "raw_text", "tinyshakespeare.txt"),
            os.path.join(data_dir, "tinyshakespeare.txt"),
        ]
        text_path = None
        for path in text_paths:
            if os.path.exists(path):
                text_path = path
                break
        if text_path is None:
            raise FileNotFoundError(f"No training text corpus found in {data_dir}")
        self.source_files = [text_path]

        with open(text_path, "r", encoding="utf-8") as f:
            text = f.read()

        tokens: list[int] = tokenizer.encode(text)
        self.tokens: LongTensor = torch.tensor(tokens, dtype=torch.long)

        if len(self.tokens) < self.seq_len:
            raise ValueError(
                f"Corpus length ({len(self.tokens)}) is shorter than sequence "
                f"length ({self.seq_len}). Reduce seq_len or use a larger corpus."
            )

    def __len__(self) -> int:
        """Number of sliding-window samples = total_tokens - seq_len."""
        return max(0, len(self.tokens) - self.seq_len)

    def __getitem__(self, idx: int):
        """Return (input, target) LongTensor pair of shape (seq_len,).

        Args:
            idx: Starting index for the sliding window.

        Returns:
            Tuple of (input_ids, target_ids) where input is
            tokens[idx:idx + seq_len] and target is
            tokens[idx + 1:idx + seq_len + 1].
        """
        input_ids = self.tokens[idx: idx + self.seq_len]
        target_ids = self.tokens[idx + 1: idx + self.seq_len + 1]
        return input_ids, target_ids


class BinaryCorpusDataset:
    """Sliding-window dataset over pre-tokenized binary shards.

    Reads ``shard_*.bin`` (uint16 big-endian, per build_info.txt) from a corpus
    directory, concatenates the token IDs, and exposes the same ``(input,
    target)`` sliding-window interface as TinyShakespeareDataset.

    ``corpus_dir`` is REQUIRED — there is no default. Pass the explicit path
    to the corpus shards (e.g. ``data/corpus/corpus2_es_wiki_gutenberg_17M``).
    Use ``build_training_dataset()`` in ``train.py`` for the standard resolution
    (checks the default path, falls back to TinyShakespeareDataset).
    """

    def __init__(self, config: TrainingConfig, corpus_dir: str) -> None:
        self.seq_len = config.seq_len
        self.corpus_dir = corpus_dir

        shard_files = sorted(
            f for f in os.listdir(corpus_dir)
            if f.startswith("shard_") and f.endswith(".bin")
        )
        if not shard_files:
            raise FileNotFoundError(f"No shard_*.bin found in {corpus_dir}")
        self.source_files = [os.path.join(corpus_dir, fname) for fname in shard_files]

        chunks = []
        for fname in shard_files:
            with open(os.path.join(corpus_dir, fname), "rb") as fh:
                raw = fh.read()
            if not raw or len(raw) % 2:
                raise ValueError(f"Invalid binary shard {fname}: expected non-empty uint16 data")
            # uint16 big-endian per build_info.txt (vocab 16384 fits in uint16)

        mm_chunks = []
        for path in self.source_files:
            file_bytes = os.path.getsize(path)
            num_tokens = file_bytes // 2
            if num_tokens > 0:
                mm = np.memmap(path, dtype=">u2", mode="r", shape=(num_tokens,))
                mm_chunks.append(mm)

        if not mm_chunks:
            raise ValueError(f"All shard files in {corpus_dir} were empty.")

        self.tokens_np = np.concatenate(mm_chunks)
        self.total_tokens = len(self.tokens_np)

    def __len__(self) -> int:
        return max(0, self.total_tokens - self.seq_len)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Extraer el trozo de uint16 y convertir solo la ventana activa a int64
        chunk = self.tokens_np[idx : idx + self.seq_len + 1].astype(np.int64)
        input_ids = torch.from_numpy(chunk[:-1])
        target_ids = torch.from_numpy(chunk[1:])
        return input_ids, target_ids
