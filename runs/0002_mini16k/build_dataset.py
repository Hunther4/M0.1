"""Build a uint16 .bin token-shard dataset for the M0.1 mini 16k run.

Integrates the NEW canonical byte-level BPE tokenizer
(E:\\M0.1\\data\\tokenizers\\tokenizer.json, vocab 16384) using the project's
own ``src.tokenizer.bpe.Tokenizer`` (the same loader the training harness uses
in ``src/training/dataset.py``). The ``tokenizers`` library cannot parse this
file because it is NOT in HuggingFace ``tokenizers`` JSON format -- it is the
bpe.py save format ({"vocab": {...}, "merges": [...]}).

Output: uint16 .bin shards under E:\\M0.1\\runs\\0002_mini16k\\data\\ plus a copy
of tokenizer.json so the engine's checkpoint hashing records the correct digest.
"""

import os
import sys
import shutil

import numpy as np

sys.path.insert(0, "E:\\M0.1")

from src.tokenizer.bpe import Tokenizer

TOKENIZER_PATH = "E:\\M0.1\\data\\tokenizers\\tokenizer.json"
OUT_DIR = "E:\\M0.1\\runs\\0002_mini16k\\data"
SHARD_SIZE = 1_000_000          # tokens per .bin shard
MAX_TOKENS = 6_000_000          # subsample cap for the mini run (note in report)
EOT_ID = 256                    # <|endoftext|> from the tokenizer's special tokens

RAW_TEXT_FILES = [
    "E:\\M0.1\\data\\raw_text\\spanish_pretrain.txt",
    "E:\\M0.1\\data\\raw_text\\combinado.txt",
]
PARQUET_DIR = "E:\\LLM_DATASETS\\FINAL"


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    tok = Tokenizer()
    tok.load(TOKENIZER_PATH)
    print(f"[BUILD] tokenizer loaded; vocab={len(tok.vocab)}")

    docs: list[str] = []

    # 1) Plain raw-text corpus
    for p in RAW_TEXT_FILES:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                text = f.read()
            # Split into documents on blank lines to get natural boundaries.
            for chunk in text.split("\n\n"):
                chunk = chunk.strip()
                if chunk:
                    docs.append(chunk)
            print(f"[BUILD] +raw {os.path.basename(p)} -> {len(docs)} docs so far")

    # 2) Parquet FINAL corpus (use the 'text' column; tokenize fresh with 16k tok)
    import pyarrow.parquet as pq
    if os.path.isdir(PARQUET_DIR):
        for fn in sorted(os.listdir(PARQUET_DIR)):
            if not fn.endswith(".parquet"):
                continue
            pf = pq.ParquetFile(os.path.join(PARQUET_DIR, fn))
            for rg in range(pf.num_row_groups):
                tbl = pf.read_row_group(rg, columns=["text"])
                col = tbl.column("text").to_pylist()
                for c in col:
                    if isinstance(c, str) and c.strip():
                        docs.append(c.strip())
            print(f"[BUILD] +parquet {fn} -> {len(docs)} docs total")

    print(f"[BUILD] total docs: {len(docs)}")

    # 3) Encode, inserting <|endoftext|> between documents as a boundary.
    all_ids: list[int] = []
    doc_count = 0
    for doc in docs:
        ids = tok.encode(doc, allowed_special={"<|endoftext|>"})
        if not ids:
            continue
        if all_ids:
            all_ids.append(EOT_ID)
        all_ids.extend(ids)
        doc_count += 1
        if len(all_ids) >= MAX_TOKENS:
            print(f"[BUILD] reached subsample cap {MAX_TOKENS} at doc {doc_count}")
            break

    total = len(all_ids)
    print(f"[BUILD] total tokens: {total:,} from {doc_count} docs")

    # 4) Write uint16 shards
    arr = np.array(all_ids, dtype=np.uint16)
    n_shards = (total + SHARD_SIZE - 1) // SHARD_SIZE
    for i in range(n_shards):
        shard = arr[i * SHARD_SIZE:(i + 1) * SHARD_SIZE]
        out_path = os.path.join(OUT_DIR, f"shard_{i:03d}.bin")
        shard.tofile(out_path)
    print(f"[BUILD] wrote {n_shards} shard(s) to {OUT_DIR}")

    # 5) Copy canonical tokenizer.json into data_dir so the engine hashes it.
    shutil.copyfile(TOKENIZER_PATH, os.path.join(OUT_DIR, "tokenizer.json"))
    print("[BUILD] copied tokenizer.json into data_dir for engine hash")

    with open(os.path.join(OUT_DIR, "build_info.txt"), "w", encoding="utf-8") as f:
        f.write(f"total_tokens={total}\n")
        f.write(f"n_shards={n_shards}\n")
        f.write(f"vocab_size={len(tok.vocab)}\n")
        f.write(f"shard_size={SHARD_SIZE}\n")
        f.write(f"max_tokens_cap={MAX_TOKENS}\n")
        f.write(f"docs={doc_count}\n")
        f.write(f"tokenizer={TOKENIZER_PATH}\n")


if __name__ == "__main__":
    main()
