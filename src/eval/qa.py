"""Qualitative benchmarks: coherence and NIAH testing."""
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
                segment_perplexity = math.exp(-next_token_prob)
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
        
        # Find needle tokens
        needle_tokens = tokenizer.encode(needle)
        
        # Calculate probability of generating needle tokens
        needle_probs = []
        for i, token in enumerate(needle_tokens[:4]):  # Check first 4 tokens of needle
            if input_ids.size(1) > i:
                token_prob = probs[0, i, token].item()
                needle_probs.append(token_prob)
        
        avg_needle_prob = sum(needle_probs) / len(needle_probs) if needle_probs else 0.0
        accuracy = avg_needle_prob  # Simplified accuracy metric
    
    return {
        "needle": needle,
        "avg_probability": avg_needle_prob,
        "accuracy": accuracy,
        "context_length": context_length
    }