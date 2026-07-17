import argparse
import sys
import os
from typing import List, Optional
from src.tokenizer.bpe import Tokenizer

def make_safe(text: str) -> str:
    """Replaces control characters with safe visual representations."""
    safe = []
    for char in text:
        o = ord(char)
        if char == '\n':
            safe.append('\\n\n')
        elif char == '\t':
            safe.append('\\t')
        elif char == '\r':
            safe.append('\\r')
        elif o < 32 or o == 127:
            safe.append(f'\\x{o:02x}')
        else:
            safe.append(char)
    return "".join(safe)

def main(argv: Optional[List[str]] = None) -> None:
    if os.name == 'nt':
        os.system('')
    parser = argparse.ArgumentParser(
        description="CLI tool to count and visualize tokens in a given text or file."
    )
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="Path to the trained tokenizer JSON file."
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str, help="Direct string text to tokenize.")
    group.add_argument("--file", type=str, help="Path to a text file to tokenize.")

    args = parser.parse_args(argv)

    if not os.path.exists(args.tokenizer):
        print(f"Error: Tokenizer file not found at '{args.tokenizer}'", file=sys.stderr)
        sys.exit(1)

    tokenizer = Tokenizer()
    try:
        tokenizer.load(args.tokenizer)
    except Exception as e:
        print(f"Error loading tokenizer: {e}", file=sys.stderr)
        sys.exit(1)

    if args.text is not None:
        text = args.text
    else:
        if not os.path.exists(args.file):
            print(f"Error: File not found at '{args.file}'", file=sys.stderr)
            sys.exit(1)
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                text = f.read()
        except UnicodeDecodeError as e:
            print(f"Error: File is not valid UTF-8: {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error reading file: {e}", file=sys.stderr)
            sys.exit(1)

    try:
        token_ids = tokenizer.encode(text)
    except Exception as e:
        print(f"Error tokenizing text: {e}", file=sys.stderr)
        sys.exit(1)

    print("--- Token Visualization ---")
    colors = ["\033[36m", "\033[33m"]
    reset = "\033[0m"
    
    for idx, token_id in enumerate(token_ids):
        color = colors[idx % 2]
        try:
            # Check if token ID is a special token
            inverse_special = {val: name for name, val in tokenizer.special_tokens.items()}
            if token_id in inverse_special:
                decoded = inverse_special[token_id]
                safe_decoded = make_safe(decoded)
                print(f"{color}{safe_decoded}{reset}", end="")
            elif token_id in tokenizer.vocab:
                raw_bytes = tokenizer.vocab[token_id]
                try:
                    # Attempt decoding as valid UTF-8
                    decoded = raw_bytes.decode('utf-8')
                    safe_decoded = make_safe(decoded)
                    print(f"{color}{safe_decoded}{reset}", end="")
                except UnicodeDecodeError:
                    # Fallback to hex for partial UTF-8 sequences
                    hex_repr = "".join(f"\\x{b:02x}" for b in raw_bytes)
                    print(f"{color}{hex_repr}{reset}", end="")
            else:
                print(f"{color}[UNK:{token_id}]{reset}", end="")
        except Exception as e:
            print(f"{color}[ERR:{token_id}]{reset}", end="")
    print("\n---------------------------")

    total_tokens = len(token_ids)
    total_bytes = len(text.encode('utf-8'))
    compression_ratio = total_bytes / total_tokens if total_tokens > 0 else 0.0
    
    vocab_size = len(tokenizer.vocab)
    unique_tokens = len(set(token_ids))
    coverage = (unique_tokens / vocab_size * 100) if vocab_size > 0 else 0.0

    print(f"Total Tokens:       {total_tokens}")
    print(f"Total Bytes:        {total_bytes}")
    print(f"Compression Ratio:  {compression_ratio:.2f}x (Bytes/Tokens)")
    print(f"Vocab Coverage:     {coverage:.2f}% ({unique_tokens} unique tokens out of {vocab_size})")
    print(f"Token IDs:          {token_ids}")

if __name__ == "__main__":
    main()
