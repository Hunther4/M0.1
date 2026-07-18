"""Qualitative benchmarks: coherence and NIAH testing."""
import math
import torch
from torch import Tensor
from torch.nn import Module


def coherence_test(
    model: Module,
    prompt: str,
    tokenizer,
    interval: int = 128,
    max_length: int = 512
) -> dict:
    """Test coherence at token intervals.
    
    Args:
        model: The language model to evaluate
        prompt: Input prompt text
        tokenizer: Tokenizer to encode the prompt
        interval: Token interval for local perplexity measurement (default 128)
        max_length: Maximum generation length
        
    Returns:
        Dictionary with interval perplexities and average coherence score
    """
    model.eval()
    
    encoded = tokenizer.encode(prompt)
    input_ids = torch.tensor(encoded).unsqueeze(0)
    
    interval_perplexities = []
    
    with torch.no_grad():
        for start in range(0, min(len(encoded) - 1, max_length), interval):
            end = min(start + interval, len(encoded) - 1)
            segment = input_ids[:, start:end]
            
            logits = model(segment)
            probs = torch.softmax(logits, dim=-1)
            
            # Get next token probability
            if end < input_ids.size(1):
                next_token = input_ids[:, end]
                next_token_prob = probs[0, -1, next_token.item()].item()
                # Perplexity = 1/p, not exp(-p)
                segment_perplexity = 1.0 / max(next_token_prob, 1e-10)
                interval_perplexities.append(segment_perplexity)
    
    avg_coherence = sum(interval_perplexities) / len(interval_perplexities) if interval_perplexities else float("inf")
    
    return {
        "interval_perplexities": interval_perplexities,
        "average_coherence": avg_coherence,
        "interval": interval
    }


def niah_test(
    model: Module,
    prompt: str,
    needle: str,
    tokenizer,
    context_length: int = 512
) -> dict:
    """Needle in a Haystack test - retrieve a specific fact from context.
    
    Args:
        model: The language model to evaluate
        prompt: Base context/prompt containing the needle
        needle: The specific piece of information to retrieve
        tokenizer: Tokenizer to encode inputs
        context_length: Length of context window (default 512)
        
    Returns:
        Dictionary with retrieval accuracy and details
    """
    model.eval()
    
    # Combine prompt with needle
    haystack = f"{prompt} {needle}"
    encoded = tokenizer.encode(haystack)
    
    # Truncate to context length
    if len(encoded) > context_length:
        encoded = encoded[:context_length]
    
    input_ids = torch.tensor(encoded).unsqueeze(0)
    
    with torch.no_grad():
        logits = model(input_ids)
        probs = torch.softmax(logits, dim=-1)
        
        # Find needle tokens — they are at the END of the haystack
        needle_tokens = tokenizer.encode(needle)
        
        # Calculate probability of generating needle tokens at their correct positions
        # (needle is appended to prompt, so it sits at the end of the sequence)
        needle_probs = []
        needle_start_pos = len(encoded) - len(needle_tokens)
        for offset, token in enumerate(needle_tokens[:4]):  # Check first 4 tokens of needle
            pos = needle_start_pos + offset
            if pos < input_ids.size(1):
                token_prob = probs[0, pos, token].item()
                needle_probs.append(token_prob)
        
        avg_needle_prob = sum(needle_probs) / len(needle_probs) if needle_probs else 0.0
        # Accuracy: fraction of needle tokens that were predicted with high probability (>0.01)
        accuracy = sum(1 for p in needle_probs if p > 0.01) / len(needle_probs) if needle_probs else 0.0
    
    return {
        "needle": needle,
        "avg_probability": avg_needle_prob,
        "accuracy": accuracy,
        "context_length": context_length
    }