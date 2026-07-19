"""TinyShakespeare dataset for training M0.1.

Provides a sliding-window PyTorch Dataset over the TinyShakespeare
corpus tokenized with the trained BPE tokenizer.
"""

import os

import torch
from torch import LongTensor

from src.tokenizer.bpe import Tokenizer
from src.training.config import TrainingConfig


class TinyShakespeareDataset:
    """Sliding-window dataset over the TinyShakespeare corpus.

    Loads the BPE tokenizer from ``data_dir/tokenizer.json``, tokenizes
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

        # Load trained BPE tokenizer (check subfolder first, fallback to root)
        tokenizer = Tokenizer()
        tokenizer_path = os.path.join(data_dir, "tokenizers", "tokenizer.json")
        if not os.path.exists(tokenizer_path):
            tokenizer_path = os.path.join(data_dir, "tokenizer.json")
        tokenizer.load(tokenizer_path)

        # Tokenize the full corpus once at init (check subfolder first, fallback to root)
        text_path = os.path.join(data_dir, "raw_text", "tinyshakespeare.txt")
        if not os.path.exists(text_path):
            text_path = os.path.join(data_dir, "tinyshakespeare.txt")

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
