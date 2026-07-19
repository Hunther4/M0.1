"""Shared dataset classes for M0.1 training scripts.

Provides AmplifiedDialogueDataset for identity/slang/defence SFT training
and JsonlDataset for single or multi-shard JSONL readers used in phase
training scripts.

Window Behavior: All datasets use OVERLAPPING sliding windows
(``__len__ = len(tokens) - seq_len``) to maximize training data from
limited corpora. This matches TinyShakespeareDataset's canonical behavior
and provides better gradient diversity during autoregressive training.
"""

import json
import os

import torch
from torch.utils.data import Dataset


class AmplifiedDialogueDataset(Dataset):
    """Dataset that tokenizes a list of dialogue strings with amplification.

    Each dialogue is repeated `amp_factor` times to overfit specific
    knowledge (identity, slang, defence) into the model weights. This is
    the base class pattern used by IdentityDataset, SlangIdentityDataset,
    AdversarialDefenceDataset, and similar SFT datasets.

    Args:
        tokenizer: Tokenizer instance with an encode() method.
        dialogues: List of dialogue strings to tokenize.
        amp_factor: Number of times to repeat the full dialogue list.
        seq_len: Sequence length for sliding window chunks (default 256).
    """

    def __init__(self, tokenizer, dialogues, amp_factor, seq_len=256):
        self.seq_len = seq_len

        amplified_text = ""
        for _ in range(amp_factor):
            for d in dialogues:
                amplified_text += f"<|user|>\n{d}\n\n\n"

        self.tokens = torch.tensor(tokenizer.encode(amplified_text), dtype=torch.long)
        print(f"AmplifiedDialogueDataset loaded. Total tokens: {len(self.tokens)}")

    def __len__(self):
        """Number of overlapping sliding-window samples."""
        if len(self.tokens) <= self.seq_len:
            return 0
        return len(self.tokens) - self.seq_len

    def __getitem__(self, idx):
        start = idx
        end = start + self.seq_len
        return self.tokens[start:end], self.tokens[start + 1 : end + 1]


class JsonlDataset(Dataset):
    """Dataset that reads conversations from one or more JSONL shard files.

    Each JSONL line must contain 'system' and 'conversation' keys. The
    dataset concatenates all tokens from all shards into a single flat
    tensor and provides sliding window chunks.

    Replaces the MultiShardJSONLDataset pattern from training scripts.

    Args:
        tokenizer: Tokenizer instance with an encode() method.
        paths: List of file paths to JSONL shard files.
        seq_len: Sequence length for sliding window chunks (default 256).
        max_lines_per_shard: Maximum lines to read per shard (default 1000).
    """

    def __init__(self, tokenizer, paths, seq_len=256, max_lines_per_shard=1000):
        self.seq_len = seq_len

        all_tokens = []
        for path in paths:
            if not os.path.exists(path):
                print(f"Warning: Shard {path} not found. Skipping.")
                continue

            print(f"Tokenizing JSONL data from {path}...")
            with open(path, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    if idx >= max_lines_per_shard:
                        break
                    try:
                        data = json.loads(line)
                        text = f"{data['system']}\n{data['conversation']}"
                        tokens = tokenizer.encode(text)
                        all_tokens.extend(tokens)
                        all_tokens.append(256)  # End of text token
                    except Exception:
                        continue

        self.tokens = torch.tensor(all_tokens, dtype=torch.long)
        print(
            f"JsonlDataset initialization complete. "
            f"Total tokens in memory: {len(self.tokens)}"
        )

    def __len__(self):
        """Number of overlapping sliding-window samples."""
        if len(self.tokens) <= self.seq_len:
            return 0
        return len(self.tokens) - self.seq_len

    def __getitem__(self, idx):
        start = idx
        end = start + self.seq_len
        x = self.tokens[start:end]
        y = self.tokens[start + 1 : end + 1]
        return x, y
