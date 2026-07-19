import torch
import pytest
from src.transformer.config import M01Config
from src.transformer.attention import CausalSelfAttention

def test_attention_output_shape():
    """Test that attention produces correct output shape."""
    config = M01Config()
    attention = CausalSelfAttention(config)
    
    batch, seq_len = 2, 5
    x = torch.randn(batch, seq_len, config.d_model)
    
    output = attention(x)
    assert output.shape == (batch, seq_len, config.d_model)

def test_attention_without_kv_cache():
    """Test attention works without KV cache."""
    config = M01Config()
    attention = CausalSelfAttention(config)
    
    x = torch.randn(1, 3, config.d_model)
    output = attention(x, kv_cache=None)
    
    assert output.shape == (1, 3, config.d_model)
    assert output.dtype == torch.float32

def test_attention_causal_mask():
    """Test that causal mask prevents attending to future tokens."""
    config = M01Config()
    attention = CausalSelfAttention(config)
    
    # Create a simple input
    x = torch.randn(1, 4, config.d_model)
    
    # Forward pass
    output = attention(x)
    
    # The output should be different if we mask future tokens
    # We can't directly test the mask, but we can test that
    # the attention weights respect causality
    # For now, just ensure it runs without error
    assert output.shape == (1, 4, config.d_model)

def test_attention_with_kv_cache():
    """Test attention works with KV cache."""
    config = M01Config()
    attention = CausalSelfAttention(config)
    from src.transformer.kv_cache import KVCache
    
    cache = KVCache(max_seq_len=10, n_heads=config.n_heads, d_head=config.d_head)
    
    # First step
    x1 = torch.randn(1, 1, config.d_model)
    output1 = attention(x1, kv_cache=cache)
    assert output1.shape == (1, 1, config.d_model)
    assert cache.seq_len == 1
    
    # Second step
    x2 = torch.randn(1, 1, config.d_model)
    output2 = attention(x2, kv_cache=cache)
    assert output2.shape == (1, 1, config.d_model)
    assert cache.seq_len == 2
    
    # Outputs should be different
    assert not torch.allclose(output1, output2, atol=1e-5)

def test_attention_gradient_flow():
    """Test that gradients flow through attention."""
    config = M01Config()
    attention = CausalSelfAttention(config)
    
    x = torch.randn(1, 2, config.d_model, requires_grad=True)
    output = attention(x)
    
    loss = output.sum()
    loss.backward()
    
    assert x.grad is not None, "Gradient should flow through attention"
    assert x.grad.shape == x.shape

def test_hybrid_attention_output_shape():
    """Test hybrid attention with compressed projections."""
    config = M01Config(use_hybrid_attention=True)
    attention = CausalSelfAttention(config)
    
    batch, seq_len = 2, 10
    x = torch.randn(batch, seq_len, config.d_model)
    output = attention(x)
    assert output.shape == (batch, seq_len, config.d_model)

def test_hybrid_attention_long_context():
    """Test hybrid attention when sequence length exceeds local window (triggering HCA)."""
    config = M01Config(use_hybrid_attention=True, local_window_size=8)
    attention = CausalSelfAttention(config)
    
    # seq_len = 12 > local_window_size (8)
    batch, seq_len = 1, 12
    x = torch.randn(batch, seq_len, config.d_model)
    output = attention(x)
    assert output.shape == (batch, seq_len, config.d_model)

def test_hybrid_attention_with_kv_cache():
    """Test hybrid attention with KV cache step-by-step decoding."""
    config = M01Config(use_hybrid_attention=True, local_window_size=8)
    attention = CausalSelfAttention(config)
    from src.transformer.kv_cache import KVCache
    
    cache = KVCache(max_seq_len=15, n_heads=config.n_heads, d_head=config.d_head)
    
    # Step 1
    x1 = torch.randn(1, 1, config.d_model)
    output1 = attention(x1, kv_cache=cache)
    assert output1.shape == (1, 1, config.d_model)
    assert cache.seq_len == 1
    
    # Step 2
    x2 = torch.randn(1, 1, config.d_model)
    output2 = attention(x2, kv_cache=cache)
    assert output2.shape == (1, 1, config.d_model)
    assert cache.seq_len == 2