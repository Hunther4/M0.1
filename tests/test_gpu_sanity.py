"""GPU Sanity Suite for M0.1 Transformer and MoE Routing Dynamics.

Run via: pytest -m gpu
"""

import pytest
import torch
import torch.nn.functional as F

from src.transformer.config import M01Config
from src.model.lm import TransformerLM
from src.training.engine import TrainingEngine
from src.training.config import TrainingConfig
from torch.utils.data import DataLoader, TensorDataset


@pytest.mark.gpu
def test_tiny_model_config_clamping():
    """Verify tiny configs (d_head <= 16) automatically clamp mla_rope_dim without crashing."""
    config = M01Config(
        vocab_size=1000,
        d_model=64,
        n_heads=4,  # d_head = 16
        n_layers=2,
        use_mla=True,
        mla_rope_dim=16,
    )
    assert config.d_head == 16
    assert config.d_head_rope == 14  # Clamped to even number strictly less than d_head
    assert config.d_head_no_rope == 2

    model = TransformerLM(config)
    x = torch.randint(0, 1000, (2, 32))
    logits = model(x)
    assert logits.shape == (2, 32, 1000)


@pytest.mark.gpu
def test_moe_routing_and_zloss():
    """Verify DeepSeek MoE routing metrics, shared experts, and Router Z-loss stability."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = M01Config(
        vocab_size=500,
        d_model=128,
        n_heads=4,
        n_layers=3,
        num_experts=8,
        num_shared_experts=2,
        moe_top_k=2,
    )
    model = TransformerLM(config).to(device)
    model.train()

    x = torch.randint(0, 500, (4, 64), device=device)
    y = torch.randint(0, 500, (4, 64), device=device)

    logits = model(x)
    ce_loss = F.cross_entropy(logits.view(-1, 500), y.view(-1))
    aux_loss = model.get_aux_loss()

    assert aux_loss > 0.0
    total_loss = ce_loss + aux_loss
    total_loss.backward()

    # Ensure gradients flow to router gate
    for block in model.blocks:
        if hasattr(block.ff, "gate"):
            assert block.ff.gate.weight.grad is not None
            assert torch.isnan(block.ff.gate.weight.grad).sum() == 0


@pytest.mark.gpu
def test_overfit_tiny_batch():
    """Verify tiny model can overfit a single batch to loss < 0.1."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = M01Config(
        vocab_size=256,
        d_model=128,
        n_heads=4,
        n_layers=2,
        num_experts=4,
        num_shared_experts=2,
        moe_top_k=2,
    )
    model = TransformerLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    x = torch.randint(0, 256, (2, 16), device=device)
    y = torch.randint(0, 256, (2, 16), device=device)

    model.train()
    for _ in range(150):
        logits = model(x)
        ce_loss = F.cross_entropy(logits.view(-1, 256), y.view(-1))
        aux_loss = model.get_aux_loss()
        loss = ce_loss + aux_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    assert ce_loss.item() < 0.1
