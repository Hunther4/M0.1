import pytest
from src.transformer.config import M01Config

def test_m01config_defaults():
    """Test that M01Config defaults match architecture.md specifications."""
    config = M01Config()
    
    # Global parameters from architecture.md
    assert config.vocab_size == 32768
    assert config.context_length == 8192
    assert config.d_model == 640
    assert config.n_heads == 10
    assert config.d_ff == 1728
    assert config.n_layers == 12
    assert config.rope_theta == 10000.0
    assert config.num_experts == 4
    assert config.num_shared_experts == 1
    assert config.moe_top_k == 1
    assert config.d_ff_shared == 1024
    assert config.d_ff_routed == 640
    assert config.dropout == 0.0
    
    # Derived parameter
    assert config.d_head == 64  # d_model // n_heads = 640 // 10 = 64

def test_m01config_validation():
    """Test that config validates d_model % n_heads == 0."""
    # Valid config
    config = M01Config(d_model=640, n_heads=10)
    assert config.d_head == 64
    
    # Invalid config should raise ValueError
    with pytest.raises(ValueError):
        M01Config(d_model=640, n_heads=7)  # 640 % 7 != 0

def test_m01config_custom_values():
    """Test config with custom values."""
    config = M01Config(
        vocab_size=1000,
        d_model=512,
        n_heads=8,
        n_layers=6
    )
    assert config.vocab_size == 1000
    assert config.d_model == 512
    assert config.n_heads == 8
    assert config.n_layers == 6
    assert config.d_head == 64  # 512 // 8 = 64