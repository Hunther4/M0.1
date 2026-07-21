import torch
import pytest
from src.transformer.kv_cache import KVCache

def test_kv_cache_memory_growth():
    # Setup
    max_seq_len = 100
    n_heads = 4
    d_head = 8
    cache = KVCache(max_seq_len, n_heads, d_head)
    
    # Pre-measure
    # This is a bit tricky to measure exactly without a profiler,
    # but we can check if the cache maintains its size.
    
    for i in range(10):
        k = torch.randn(1, 1, n_heads, d_head)
        v = torch.randn(1, 1, n_heads, d_head)
        cache.append(k, v)
        
        # The cache should not be growing, just filling pre-allocated buffer
        assert cache.k.shape == (1, max_seq_len, n_heads, d_head)
        assert cache.v.shape == (1, max_seq_len, n_heads, d_head)

def test_kv_cache_no_leak_on_reset():
    cache = KVCache(10, 4, 8)
    k = torch.randn(1, 1, 4, 8)
    v = torch.randn(1, 1, 4, 8)
    cache.append(k, v)
    
    cache.reset()
    assert cache.seq_len == 0
    assert torch.sum(torch.abs(cache.k)) == 0

def test_kv_cache_view_not_clone():
    cache = KVCache(10, 4, 8)
    k = torch.randn(1, 1, 4, 8)
    v = torch.randn(1, 1, 4, 8)
    full_k, full_v = cache.append(k, v)
    
    # Check if full_k is a view (data_ptr should match)
    # The clone() implementation in KVCache makes it fail
    assert full_k.data_ptr() == cache.k.data_ptr()


def test_kv_cache_rejects_batch_change_after_tokens():
    cache = KVCache(10, 1, 2)
    cache.append(torch.zeros(1, 1, 1, 2), torch.zeros(1, 1, 1, 2))

    with pytest.raises(ValueError, match="batch size"):
        cache.append(torch.zeros(2, 1, 1, 2), torch.zeros(2, 1, 1, 2))
