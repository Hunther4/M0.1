"""CLI entry point for text generation.

Usage:
    python -m src.inference.cli --prompt "Hello world"
    python -m src.inference.cli --prompt "Once upon a time" --max-len 200 --temp 0.8 --top-k 50
"""

import argparse
import os
import sys

import torch

from src.inference.generate import generate
from src.model.lm import TransformerLM
from src.tokenizer.bpe import Tokenizer
from src.training.checkpoint import CheckpointManager
from src.transformer.config import M01Config
from src.engine_v2.checkpoint_v2 import normalize_checkpoint_state
from src.engine_v2.checkpoint_v2 import safe_load_checkpoint


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate text with the M0.1 transformer model.",
    )
    parser.add_argument(
        "--prompt", type=str, default="Hello",
        help="Input prompt text (default: Hello)",
    )
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/checkpoint.pt",
        help="Path to model checkpoint (default: checkpoints/checkpoint.pt)",
    )
    parser.add_argument(
        "--max-len", type=int, default=100,
        help="Maximum number of new tokens to generate (default: 100)",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="Sampling temperature. 0 = greedy, 1.0 = default (default: 1.0)",
    )
    parser.add_argument(
        "--top-k", type=int, default=None,
        help="Top-k sampling threshold (default: disabled)",
    )
    parser.add_argument(
        "--top-p", type=float, default=None,
        help="Nucleus sampling threshold, e.g. 0.9 (default: disabled)",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help='Device: "auto" (detect), "cuda", or "cpu" (default: auto)',
    )
    return parser.parse_args(argv)


def resolve_device(device_arg: str) -> torch.device:
    """Resolve device string to torch.device."""
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv or sys.argv[1:])
    device = resolve_device(args.device)

    print(f"Device: {device}")
    print(f"Loading checkpoint: {args.checkpoint}")

    # Load trained weights and architecture metadata from checkpoint.
    state = safe_load_checkpoint(args.checkpoint, map_location=device)
    state = normalize_checkpoint_state(state, require_architecture=True)
    config_dict = state["model_config"]
    valid_fields = set(M01Config.__dataclass_fields__)
    config = M01Config(**{k: v for k, v in config_dict.items() if k in valid_fields})
    model = TransformerLM(config).to(device)
    model.load_state_dict(state["model_state"])
    model.eval()
    print(f"Loaded step {state.get('step', 0)}, loss {state.get('loss', 'N/A')}")

    # Load tokenizer (use the 16K vocab matching the checkpoint tokenizer_hash)
    tokenizer = Tokenizer()
    tokenizer.load(os.path.join(os.path.dirname(__file__), "..", "..", "data", "tokenizers", "tokenizer.json"))

    print(f"\nPrompt: {args.prompt}")
    print("Generating...\n")

    output = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_gen_len=args.max_len,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=device,
    )

    print(output)
    print()


if __name__ == "__main__":
    main()
