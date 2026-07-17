import torch
import pytest
from src.transformer.config import M01Config
from src.transformer.rope import RotaryPositionalEmbedding

def test_rope_position_0_identity():
    """Test that RoPE at position 0 produces identity rotation (no change)."""
    config = M01Config()
    rope = RotaryPositionalEmbedding(config)
    
    # Create a random tensor: (batch=1, seq=1, n_heads=10, d_head=64)
    x = torch.randn(1, 1, config.n_heads, config.d_head)
    
    # Position 0 should not change the tensor (cos(0)=1, sin(0)=0)
    rotated = rope(x, offset=0)
    assert torch.allclose(x, rotated, atol=1e-5), \
        "RoPE at position 0 should be identity (no rotation)"

def test_rope_position_1_rotation():
    """Test that RoPE at position 1 applies rotation by θ."""
    config = M01Config()
    rope = RotaryPositionalEmbedding(config)
    
    # Create a tensor with known values: (batch=1, seq=1, n_heads=1, d_head=64)
    # We'll use a simple pattern: alternating 1.0 and 0.0
    x = torch.zeros(1, 1, 1, config.d_head)
    x[0, 0, 0, 0::2] = 1.0  # even indices = 1.0
    x[0, 0, 0, 1::2] = 0.0  # odd indices = 0.0
    
    rotated = rope(x, offset=1)
    
    # Expected rotation: for each pair (x_even, x_odd) at position m=1:
    # rotated_even = x_even * cos(m*θ) - x_odd * sin(m*θ)
    # rotated_odd  = x_even * sin(m*θ) + x_odd * cos(m*θ)
    # With x_odd = 0: rotated_even = x_even * cos(θ), rotated_odd = x_even * sin(θ)
    theta = config.rope_theta
    i = 0  # first pair index
    cos_val = torch.cos(torch.tensor(1.0 * theta ** (-2*i/config.d_head)))
    sin_val = torch.sin(torch.tensor(1.0 * theta ** (-2*i/config.d_head)))
    
    expected_even = 1.0 * cos_val.item()
    expected_odd = 1.0 * sin_val.item()
    
    # Check first pair
    assert torch.allclose(rotated[0, 0, 0, 0], torch.tensor(expected_even), atol=1e-5)
    assert torch.allclose(rotated[0, 0, 0, 1], torch.tensor(expected_odd), atol=1e-5)

def test_rope_output_shape():
    """Test that RoPE preserves input shape."""
    config = M01Config()
    rope = RotaryPositionalEmbedding(config)
    
    batch, seq, heads, d_head = 2, 5, config.n_heads, config.d_head
    x = torch.randn(batch, seq, heads, d_head)
    
    rotated = rope(x, offset=3)
    assert rotated.shape == x.shape, f"Shape mismatch: {rotated.shape} != {x.shape}"

def test_rope_different_offsets():
    """Test that different offsets produce different rotations."""
    config = M01Config()
    rope = RotaryPositionalEmbedding(config)
    
    x = torch.randn(1, 1, 1, config.d_head)
    
    rotated_0 = rope(x, offset=0)
    rotated_1 = rope(x, offset=1)
    
    # They should be different (unless x is zero)
    assert not torch.allclose(rotated_0, rotated_1, atol=1e-5), \
        "Different offsets should produce different rotations"