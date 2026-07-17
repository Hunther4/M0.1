import torch
import pytest
from src.transformer.kv_cache import KVCache

def test_kv_cache_initialization():
    """Test that KV cache initializes with correct shape."""
    max_seq_len = 100
    n_heads = 10
    d_head = 64
    
    cache = KVCache(max_seq_len, n_heads, d_head)
    
    # Initial sequence length should be 0
    assert cache.seq_len == 0
    
    # Buffers should be pre-allocated
    assert cache.k.shape == (1, max_seq_len, n_heads, d_head)
    assert cache.v.shape == (1, max_seq_len, n_heads, d_head)

def test_kv_cache_append():
    """Test that append increases sequence length and stores K/V."""
    cache = KVCache(max_seq_len=10, n_heads=2, d_head=4)
    
    # Create K/V tensors: (batch=1, seq_len=1, n_heads=2, d_head=4)
    k = torch.randn(1, 1, 2, 4)
    v = torch.randn(1, 1, 2, 4)
    
    # Append first
    k_full, v_full = cache.append(k, v)
    assert cache.seq_len == 1
    assert k_full.shape == (1, 1, 2, 4)
    assert v_full.shape == (1, 1, 2, 4)
    
    # Append second
    k2 = torch.randn(1, 1, 2, 4)
    v2 = torch.randn(1, 1, 2, 4)
    k_full, v_full = cache.append(k2, v2)
    assert cache.seq_len == 2
    assert k_full.shape == (1, 2, 2, 4)
    assert v_full.shape == (1, 2, 2, 4)
    
    # Check values are stored correctly
    assert torch.allclose(k_full[0, 0], k[0, 0])
    assert torch.allclose(k_full[0, 1], k2[0, 0])

def test_kv_cache_reset():
    """Test that reset clears cache and resets position."""
    cache = KVCache(max_seq_len=10, n_heads=2, d_head=4)
    
    # Append some data
    k = torch.randn(1, 1, 2, 4)
    v = torch.randn(1, 1, 2, 4)
    cache.append(k, v)
    assert cache.seq_len == 1
    
    # Reset
    cache.reset()
    assert cache.seq_len == 0
    
    # After reset, append should work from position 0
    k2 = torch.randn(1, 1, 2, 4)
    v2 = torch.randn(1, 1, 2, 4)
    k_full, v_full = cache.append(k2, v2)
    assert cache.seq_len == 1
    assert k_full.shape == (1, 1, 2, 4)

def test_kv_cache_multiple_appends():
    """Test multiple appends maintain correct state."""
    cache = KVCache(max_seq_len=5, n_heads=1, d_head=2)
    
    values = []
    for i in range(5):
        k = torch.ones(1, 1, 1, 2) * (i + 1)
        v = torch.ones(1, 1, 1, 2) * (i + 10)
        k_full, v_full = cache.append(k, v)
        values.append((k_full.clone(), v_full.clone()))
    
    assert cache.seq_len == 5
    
    # Check final state
    k_final, v_final = values[-1]
    assert k_final.shape == (1, 5, 1, 2)
    assert v_final.shape == (1, 5, 1, 2)
    
    # Check values at each position
    for i in range(5):
        assert torch.allclose(k_final[0, i, 0, 0], torch.tensor(float(i + 1)))
        assert torch.allclose(v_final[0, i, 0, 0], torch.tensor(float(i + 10)))