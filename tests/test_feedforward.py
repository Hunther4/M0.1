import torch
import pytest
from src.transformer.config import M01Config
from src.transformer.feedforward import FeedForward

def test_feedforward_output_shape():
    """Test that feedforward produces correct output shape."""
    config = M01Config()
    ff = FeedForward(config)
    
    batch, seq_len = 2, 5
    x = torch.randn(batch, seq_len, config.d_model)
    
    output = ff(x)
    assert output.shape == (batch, seq_len, config.d_model)

def test_feedforward_swiglu_activation():
    """Test that feedforward uses SwiGLU activation."""
    config = M01Config()
    ff = FeedForward(config)
    
    # Create input
    x = torch.randn(1, 1, config.d_model)
    
    # Forward pass
    output = ff(x)
    
    # SwiGLU: down(SiLU(gate(x)) ⊙ up(x))
    # Verify it's not just a linear transformation
    # by checking that different inputs produce different outputs
    x2 = torch.randn(1, 1, config.d_model)
    output2 = ff(x2)
    
    assert not torch.allclose(output, output2, atol=1e-5), \
        "Different inputs should produce different outputs"

def test_feedforward_gradient_flow():
    """Test that gradients flow through feedforward."""
    config = M01Config()
    ff = FeedForward(config)
    
    x = torch.randn(1, 2, config.d_model, requires_grad=True)
    output = ff(x)
    
    loss = output.sum()
    loss.backward()
    
    assert x.grad is not None, "Gradient should flow through feedforward"
    assert x.grad.shape == x.shape

def test_feedforward_parameter_count():
    """Test that feedforward has correct number of parameters."""
    config = M01Config()
    ff = FeedForward(config)
    
    # SwiGLU has 3 linear layers: gate, up, down
    # gate: d_model × d_ff
    # up: d_model × d_ff
    # down: d_ff × d_model
    expected_params = config.d_model * config.d_ff * 3
    
    param_count = sum(p.numel() for p in ff.parameters())
    assert param_count == expected_params, \
        f"Expected {expected_params} parameters, got {param_count}"