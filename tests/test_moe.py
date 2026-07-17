import torch
import pytest
from src.transformer.config import M01Config
from src.transformer.moe import MoELayer

def test_moe_output_shape():
    """Test that MoE layer produces correct output shape."""
    config = M01Config()
    moe = MoELayer(config)
    
    batch, seq_len = 2, 5
    x = torch.randn(batch, seq_len, config.d_model)
    
    output = moe(x)
    assert output.shape == (batch, seq_len, config.d_model)

def test_moe_placeholder_behavior():
    """Test that MoE placeholder passes through first expert only."""
    config = M01Config(num_experts=1)
    moe = MoELayer(config)
    
    # With 1 expert, it should behave like a single feedforward
    x = torch.randn(1, 1, config.d_model)
    output = moe(x)
    
    # Output should be non-zero (not a no-op)
    assert output.abs().sum() > 0

def test_moe_expert_count():
    """Test that MoE creates correct number of experts."""
    config = M01Config(num_experts=3)
    moe = MoELayer(config)
    
    assert len(moe.experts) == 3
    assert moe.gate.out_features == 3

def test_moe_gradient_flow():
    """Test that gradients flow through MoE layer."""
    config = M01Config()
    moe = MoELayer(config)
    
    x = torch.randn(1, 2, config.d_model, requires_grad=True)
    output = moe(x)
    
    loss = output.sum()
    loss.backward()
    
    assert x.grad is not None, "Gradient should flow through MoE"
    assert x.grad.shape == x.shape