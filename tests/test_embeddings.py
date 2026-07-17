import torch
import pytest
from src.transformer.config import M01Config
from src.transformer.embeddings import TokenEmbedding

def test_token_embedding_output_shape():
    """Test that token embedding produces correct shape."""
    config = M01Config()
    embed = TokenEmbedding(config)
    
    # Input: (batch=2, seq_len=5)
    token_ids = torch.randint(0, config.vocab_size, (2, 5))
    
    # Output should be (batch=2, seq_len=5, d_model=640)
    embeddings = embed(token_ids)
    assert embeddings.shape == (2, 5, config.d_model)

def test_token_embedding_scaling():
    """Test that embeddings are scaled by 1/sqrt(d_model)."""
    config = M01Config()
    embed = TokenEmbedding(config)
    
    token_ids = torch.tensor([[0]])  # single token
    embeddings = embed(token_ids)
    
    # The embedding weight for token 0
    raw_weight = embed.embedding.weight[0]
    expected = raw_weight * (1.0 / (config.d_model ** 0.5))
    
    assert torch.allclose(embeddings[0, 0], expected, atol=1e-5)

def test_output_head_tied_weights():
    """Test that output head uses tied weights from embedding."""
    config = M01Config()
    embed = TokenEmbedding(config)
    
    # Create hidden states
    hidden = torch.randn(1, 1, config.d_model)
    
    # Output head should use embedding.weight
    logits = embed.output_head(hidden)
    assert logits.shape == (1, 1, config.vocab_size)
    
    # Verify it's using the same weight matrix
    # output_head computes: hidden @ embedding.weight.T
    expected = torch.nn.functional.linear(hidden, embed.embedding.weight)
    assert torch.allclose(logits, expected, atol=1e-5)

def test_embedding_gradient_flow():
    """Test that gradients flow through tied weights."""
    config = M01Config()
    embed = TokenEmbedding(config)
    
    token_ids = torch.tensor([[0]])
    embeddings = embed(token_ids)
    
    # Forward through output head
    logits = embed.output_head(embeddings)
    
    # Backward pass
    loss = logits.sum()
    loss.backward()
    
    # Gradient should be non-null
    assert embed.embedding.weight.grad is not None, \
        "Gradient should flow through tied weights"
    
    # Gradient should have correct shape
    assert embed.embedding.weight.grad.shape == (config.vocab_size, config.d_model)

def test_embedding_single_parameter_group():
    """Test that there is only one parameter group (tied weights)."""
    config = M01Config()
    embed = TokenEmbedding(config)
    
    # Count parameters
    param_count = sum(p.numel() for p in embed.parameters())
    expected_count = config.vocab_size * config.d_model
    
    assert param_count == expected_count, \
        f"Expected {expected_count} parameters, got {param_count}"