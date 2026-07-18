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

    # Build model with default config
    config = M01Config()
    model = TransformerLM(config)
    model.to(device)
    model.eval()

    # Load trained weights from checkpoint
    manager = CheckpointManager(os.path.dirname(args.checkpoint))
    # load from specific path instead of default checkpoint.pt
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    print(f"Loaded step {state['step']}, loss {state['loss']:.4f}")

    # Load tokenizer
    tokenizer = Tokenizer()
    tokenizer.load("data/tokenizer.json")

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
