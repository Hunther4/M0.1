"""DeepSeek-style Mixture of Experts (MoE) Layer.

Implements a Mixture of Experts layer with both shared and routed experts.
- Shared experts capture general common knowledge and are always active.
- Routed experts specialize in different domains and are dynamically gated.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import M01Config
from .feedforward import FeedForward


class MoELayer(nn.Module):
    """DeepSeek-style Mixture of Experts (MoE) Layer.
    
    Combines shared experts (always active) with routed experts (top-k gated).
    """
    
    def __init__(self, config: M01Config) -> None:
        """Initialize MoE layer.
        
        Args:
            config: M01Config with num_experts, num_shared_experts, moe_top_k, d_model
        """
        super().__init__()
        
        self.config = config
        self.num_experts = config.num_experts
        self.num_shared_experts = config.num_shared_experts
        self.moe_top_k = config.moe_top_k
        
        # Determine internal FFN dimensions for shared and routed experts
        # We partition the routed experts' hidden dimension to keep parameters per token constant.
        d_ff_shared = config.d_ff_shared if config.d_ff_shared is not None else config.d_ff
        d_ff_routed = config.d_ff_routed if config.d_ff_routed is not None else max(1, config.d_ff // max(1, config.moe_top_k))
        
        # Shared experts: Always active
        self.shared_experts = nn.ModuleList([
            FeedForward(config, d_ff=d_ff_shared) for _ in range(self.num_shared_experts)
        ])
        
        # Routed experts: Top-k gated
        self.experts = nn.ModuleList([
            FeedForward(config, d_ff=d_ff_routed) for _ in range(self.num_experts)
        ])
        
        # Gate network to route tokens to routed experts
        self.gate = nn.Linear(config.d_model, config.num_experts, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of MoE layer.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)  # (N, d_model)
        
        # 1. Compute shared experts output (always active for all tokens)
        shared_out = torch.zeros_like(x_flat)
        for expert in self.shared_experts:
            shared_out += expert(x_flat)
            
        # 2. Handle routed experts
        if self.num_experts <= 1:
            # Fallback if there is only 1 expert
            routed_out = self.experts[0](x_flat)
            output = shared_out + routed_out
            return output.view(batch_size, seq_len, d_model)
            
        # Compute gate logits and probabilities
        gate_logits = self.gate(x_flat)  # (N, num_experts)
        gate_probs = F.softmax(gate_logits, dim=-1)  # (N, num_experts)
        
        # Select top-k experts
        k = min(self.moe_top_k, self.num_experts)
        topk_probs, topk_indices = torch.topk(gate_probs, k=k, dim=-1)
        
        # Re-normalize top-k probabilities to sum to 1
        topk_probs = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-9)
        
        routed_out = torch.zeros_like(x_flat)
        
        # Route tokens to their selected experts
        for i in range(self.num_experts):
            # Mask identifying which tokens route to expert i
            mask = (topk_indices == i)  # (N, k)
            # Find the token indices and corresponding top-k ranks (0 to k-1)
            token_indices, topk_ranks = torch.where(mask)
            
            if token_indices.numel() == 0:
                continue
            
            # Extract inputs and execute expert i
            expert_inputs = x_flat[token_indices]
            expert_outputs = self.experts[i](expert_inputs)
            
            # Multiply by gate weight
            weights = topk_probs[token_indices, topk_ranks].unsqueeze(-1)
            routed_out[token_indices] += expert_outputs * weights
            
        # Combine shared and routed outputs
        output = shared_out + routed_out
        
        return output.view(batch_size, seq_len, d_model)