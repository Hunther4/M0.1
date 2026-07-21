#!/usr/bin/env python3
"""CLI entry point for evaluation suite."""
import argparse
import logging
import sys
from pathlib import Path

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.eval import calculate_perplexity, coherence_test, niah_test, save_results, setup_logging
from src.model.lm import TransformerLM
from src.tokenizer.bpe import Tokenizer
from src.transformer.config import M01Config
from src.engine_v2.checkpoint_v2 import normalize_checkpoint_state, safe_load_checkpoint


def load_checkpoint(checkpoint_path: str, device: str = "cpu") -> dict:
    """Load model checkpoint.
    
    Args:
        checkpoint_path: Path to .pt checkpoint file
        device: Device to load model to
        
    Returns:
        Checkpoint dictionary with model_state_dict and config
    """
    checkpoint = safe_load_checkpoint(Path(checkpoint_path), map_location=device)
    return normalize_checkpoint_state(checkpoint, require_architecture=True)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate M0.1 model checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/eval/evaluate.py --checkpoint checkpoints/model.pt
  python src/eval/evaluate.py --checkpoint checkpoints/model.pt --coherence --niah
  python src/eval/evaluate.py --checkpoint checkpoints/model.pt --verbose
        """
    )
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--dataset", default="data/tiny_shakespeare_val.txt", help="Validation dataset path")
    parser.add_argument("--coherence", action="store_true", help="Run coherence test")
    parser.add_argument("--niah", action="store_true", help="Run NIAH test")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging(log_level)
    
    logger.info(f"Loading checkpoint from {args.checkpoint}")
    
    try:
        checkpoint = load_checkpoint(args.checkpoint, args.device)
    except FileNotFoundError:
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        sys.exit(1)
    
    # Initialize model from checkpoint config
    config_dict = checkpoint["model_config"]
    valid_fields = set(M01Config.__dataclass_fields__)
    config = M01Config(**{k: v for k, v in config_dict.items() if k in valid_fields})
    model = TransformerLM(config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(args.device)
    model.eval()
    
    logger.info(f"Model loaded successfully (device: {args.device})")
    
    # Load tokenizer
    tokenizer = Tokenizer()
    tokenizer_path = Path("data/tokenizers/tokenizer.json")
    if not tokenizer_path.exists():
        tokenizer_path = Path("data/tokenizer.json")

    if tokenizer_path.exists():
        tokenizer.load(str(tokenizer_path))
    else:
        logger.warning(f"Tokenizer not found: {tokenizer_path}, using untrained tokenizer")
    
    # Load validation data
    val_path = Path(args.dataset)
    if val_path.exists():
        val_text = val_path.read_text(encoding="utf-8")
    else:
        logger.warning(f"Validation dataset not found: {args.dataset}, using fallback")
        val_text = "The quick brown fox jumps over the lazy dog. " * 100
    
    # Use first 10k chars for eval
    val_text = val_text[:10000]
    val_tokens = tokenizer.encode(val_text)
    
    logger.info(f"Evaluating on first {min(len(val_tokens), 512)} of {len(val_tokens)} tokens")
    
    # Quantitative metrics — use 512 token window for speed
    input_ids = torch.tensor(val_tokens[:512]).unsqueeze(0).to(args.device)
    perplexity = calculate_perplexity(model, input_ids)
    logger.info(f"Perplexity: {perplexity:.4f}")
    
    results = {
        "metrics": {
            "perplexity": perplexity
        }
    }
    
    # Qualitative benchmarks (if requested)
    if args.coherence:
        logger.info("Running coherence test...")
        prompt = val_text[:200]
        coherence_result = coherence_test(model, prompt, tokenizer)
        results["qa"] = results.get("qa", {})
        results["qa"]["coherence"] = coherence_result
        logger.info(f"Coherence: {coherence_result['average_coherence']:.4f}")
    
    if args.niah:
        logger.info("Running NIAH test...")
        # Create a haystack with a needle
        prompt = "The secret code is 42. " * 30
        needle = "42"
        niah_result = niah_test(model, prompt, needle, tokenizer)
        results["qa"] = results.get("qa", {})
        results["qa"]["niah"] = niah_result
        logger.info(f"NIAH accuracy: {niah_result['accuracy']:.4f}")
    
    # Save results
    output_path = save_results(results, args.checkpoint)
    logger.info(f"Results saved to {output_path}")
    
    print(f"\nEvaluation complete. Results saved to {output_path}")
    print(f"Perplexity: {perplexity:.4f}")


if __name__ == "__main__":
    main()
