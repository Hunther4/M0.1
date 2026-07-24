"""CLI for real checkpoint inference and prompt-cache benchmarks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

from src.engine_v2.checkpoint_v2 import normalize_checkpoint_state, safe_load_checkpoint
from src.eval.inference_benchmark import benchmark_inference, compare_prompt_cache
from src.model.lm import TransformerLM
from src.tokenizer.bpe import Tokenizer
from src.transformer.config import M01Config


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--prompt", action="append", dest="prompts")
    parser.add_argument("--prompts-file")
    parser.add_argument("--max-len", type=int, default=32)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--mode", choices=("off", "on", "compare"), default="compare")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output")
    return parser.parse_args(argv)


def _load_prompts(args: argparse.Namespace) -> list[str]:
    prompts = list(args.prompts or [])
    if args.prompts_file:
        prompts.extend(
            line.strip()
            for line in Path(args.prompts_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    if not prompts:
        raise ValueError("Provide at least one --prompt or --prompts-file")
    return prompts


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv or sys.argv[1:])
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    state = normalize_checkpoint_state(
        safe_load_checkpoint(args.checkpoint, map_location=device),
        require_architecture=True,
    )
    valid_fields = set(M01Config.__dataclass_fields__)
    config = M01Config(
        **{key: value for key, value in state["model_config"].items() if key in valid_fields}
    )
    model = TransformerLM(config).to(device)
    model.load_state_dict(state["model_state"])
    model.eval()

    tokenizer = Tokenizer()
    tokenizer.load(args.tokenizer)
    prompts = _load_prompts(args)
    kwargs = {
        "max_gen_len": args.max_len,
        "repetitions": args.repetitions,
    }
    if args.mode == "compare":
        report = compare_prompt_cache(model, tokenizer, prompts, **kwargs)
    else:
        report = benchmark_inference(
            model,
            tokenizer,
            prompts,
            use_prompt_cache=args.mode == "on",
            **kwargs,
        )

    serialized = json.dumps(report, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized + "\n", encoding="utf-8")
        print(f"Benchmark written to {output}")
    else:
        print(serialized)


if __name__ == "__main__":
    main()
