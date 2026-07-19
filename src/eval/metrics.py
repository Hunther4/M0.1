"""Quantitative metrics for model evaluation."""
import torch
from torch import Tensor
from torch.nn import Module


def calculate_perplexity(model: Module, input_ids: Tensor, attention_mask: Tensor | None = None) -> float:
    """Calculate perplexity on a sequence.
    
    Args:
        model: The language model to evaluate
        input_ids: Token IDs [seq_len] or [batch, seq_len]
        attention_mask: Optional attention mask (unused, for API compatibility)
        
    Returns:
        Perplexity as a float
    """
    model.eval()
    with torch.no_grad():
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        
        logits = model(input_ids)
        
        # Calculate loss using cross-entropy
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        
        loss_fct = torch.nn.CrossEntropyLoss(reduction="mean")
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        perplexity = torch.exp(loss).item()
    
    return perplexity


def log_loss(model: Module, input_ids: Tensor) -> dict:
    """Calculate loss and return with additional metrics.
    
    Args:
        model: The language model to evaluate
        input_ids: Token IDs
        
    Returns:
        Dictionary with loss, perplexity, and token count
    """
    model.eval()
    with torch.no_grad():
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        
        logits = model(input_ids)
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        
        loss_fct = torch.nn.CrossEntropyLoss(reduction="mean")
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        perplexity = torch.exp(loss).item()
        token_count = shift_labels.numel()
        
        return {
            "loss": loss.item(),
            "perplexity": perplexity,
            "token_count": token_count
        }