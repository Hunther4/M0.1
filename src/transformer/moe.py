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
        self.capacity_factor = config.capacity_factor
        self._target_capacity_factor = config.capacity_factor
        self.capacity_factor_warmup_steps = config.capacity_factor_warmup_steps
        self.capacity_factor_warmup_start = config.capacity_factor_warmup_start
        self.effective_capacity_factor = self._target_capacity_factor
        
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

    def set_step(self, step: int) -> None:
        """Update the capacity factor according to the training-step warmup."""
        if step < self.capacity_factor_warmup_steps:
            ratio = step / max(self.capacity_factor_warmup_steps, 1)
            self.effective_capacity_factor = (
                self.capacity_factor_warmup_start
                + (self._target_capacity_factor - self.capacity_factor_warmup_start) * ratio
            )
        else:
            self.effective_capacity_factor = self._target_capacity_factor
        
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
            
        # Initialize default aux loss
        self.current_aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        # 2. Handle routed experts
        if self.num_experts <= 1:
            # Fallback if there is only 1 expert
            routed_out = self.experts[0](x_flat)
            output = shared_out + routed_out
            return output.view(batch_size, seq_len, d_model)
            
        # Compute gate logits and probabilities
        gate_logits = self.gate(x_flat)  # (N, num_experts) — CLEAN
        if self.training:
            # DeepSeek-style gate noise encourages early exploration
            noise_std = 0.1 * F.softplus(gate_logits)
            routing_logits = gate_logits + torch.randn_like(gate_logits) * noise_std
        else:
            routing_logits = gate_logits

        # Clean probabilities are used for losses; noisy probabilities are used
        # only for routing decisions.
        clean_probs = F.softmax(gate_logits.float(), dim=-1).to(gate_logits.dtype)
        routing_probs = F.softmax(routing_logits.float(), dim=-1).to(routing_logits.dtype)
        
        # Select top-k experts
        k = min(self.moe_top_k, self.num_experts)
        topk_probs, topk_indices = torch.topk(routing_probs, k=k, dim=-1)
        
        # Normalize the selected probabilities before capacity filtering. Accepted
        # assignments are normalized again below because capacity may reject ranks.
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(topk_probs.dtype).eps
        )

        # Store routing intermediates (clean values for losses, noisy values for monitoring)
        self.gate_logits = gate_logits
        self.gate_probs = clean_probs
        self.routing_probs = routing_probs
        self.topk_indices = topk_indices
        self.topk_probs = topk_probs
        num_tokens = x_flat.size(0)

        # Limit each routed expert to its capacity. Every Top-K choice is an
        # independent assignment; rejecting one rank does not prevent another
        # rank for the same token from being accepted.
        self.capacity = max(
            1,
            int(num_tokens * k / self.num_experts * self.effective_capacity_factor),
        )
        expert_counts = torch.zeros(
            self.num_experts,
            device=x.device,
            dtype=torch.long,
        )
        accepted_assignments = torch.zeros(
            (num_tokens, k),
            device=x.device,
            dtype=torch.bool,
        )

        for rank in range(k):
            expert_for_token = topk_indices[:, rank]
            weight_for_token = topk_probs[:, rank]

            for expert_id in range(self.num_experts):
                token_indices = torch.where(expert_for_token.eq(expert_id))[0]
                if token_indices.numel() == 0:
                    continue

                available = self.capacity - expert_counts[expert_id].item()
                take = min(token_indices.numel(), max(0, available))
                if take == 0:
                    continue

                ranked_indices = token_indices[
                    torch.argsort(weight_for_token[token_indices], descending=True)
                ]
                accepted_indices = ranked_indices[:take]
                accepted_assignments[accepted_indices, rank] = True
                expert_counts[expert_id] += take

        assigned_experts = topk_indices.masked_fill(~accepted_assignments, -1)
        accepted_weights = topk_probs * accepted_assignments.to(topk_probs.dtype)
        accepted_weight_sum = accepted_weights.sum(dim=-1, keepdim=True)
        assigned_weights = torch.where(
            accepted_weight_sum > 0,
            accepted_weights
            / accepted_weight_sum.clamp_min(torch.finfo(topk_probs.dtype).eps),
            torch.zeros_like(accepted_weights),
        )

        expert_mask = torch.zeros(
            self.num_experts,
            num_tokens,
            device=x.device,
            dtype=x.dtype,
        )
        for expert_id in range(self.num_experts):
            expert_mask[expert_id] = assigned_experts.eq(expert_id).any(dim=-1).to(x.dtype)

        self.assigned_experts = assigned_experts
        self.assigned_weights = assigned_weights
        self.expert_counts = expert_counts
        self.expert_mask = expert_mask
        self.accepted_assignments_per_rank = accepted_assignments.sum(dim=0)
        self.dropped_tokens = assigned_experts.eq(-1).all(dim=-1).sum()
        
        # 3. Compute auxiliary load balancing loss to prevent routing collapse
        if self.training:
            f = expert_mask.mean(dim=1)  # (num_experts,) post-capacity assignments
            P = clean_probs.mean(dim=0)  # (num_experts,)
            # DeepSeek-style load balancing: num_experts * sum(f * P)
            self.current_aux_loss = self.num_experts * torch.sum(f * P)
            # Z-Loss is computed separately via get_z_loss() to avoid double-counting
            # when LossPipeline's RouterZLossTerm is also used.
        else:
            self.current_aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        routed_out = torch.zeros_like(x_flat)
        
        # Accumulate every capacity-accepted expert output for each token.
        for i in range(self.num_experts):
            token_indices, topk_ranks = torch.where(assigned_experts.eq(i))
            
            if token_indices.numel() == 0:
                continue
            
            # Extract inputs and execute expert i
            expert_inputs = x_flat[token_indices]
            expert_outputs = self.experts[i](expert_inputs)
            
            # Multiply by gate weight
            weights = assigned_weights[token_indices, topk_ranks].unsqueeze(-1)
            routed_out[token_indices] += expert_outputs * weights
            
        # Combine shared and routed outputs
        output = shared_out + routed_out

        # Detach monitoring-only routing intermediates to free autograd graph.
        # Tensors needed for loss (gate_logits, gate_probs, assigned_weights, expert_mask)
        # are intentionally kept in the graph.
        self.routing_probs = self.routing_probs.detach()
        self.topk_indices = self.topk_indices.detach()
        self.topk_probs = self.topk_probs.detach()
        self.assigned_experts = self.assigned_experts.detach()
        self.accepted_assignments_per_rank = self.accepted_assignments_per_rank.detach()
        self.dropped_tokens = self.dropped_tokens.detach()
        
        return output.view(batch_size, seq_len, d_model)

    def get_aux_loss(self) -> torch.Tensor:
        """Return the auxiliary load balancing loss (WITHOUT z-loss).

        Z-loss is computed separately via get_z_loss() to avoid double-counting
        when RouterZLossTerm in LossPipeline is also used in the training loop.
        """
        return getattr(self, "current_aux_loss", torch.tensor(0.0, device=self.gate.weight.device))

    def get_z_loss(self) -> torch.Tensor:
        """Return the Router Z-Loss computed during the last forward pass.

        Z-Loss = mean(logsumexp(gate_logits)²) — penalizes large unnormalized gate logits
        to prevent routing collapse (DeepSeek-style).

        Returns zero tensor when not in training mode or when gate_logits are unavailable.
        """
        gate_logits = getattr(self, "gate_logits", None)
        if gate_logits is not None:
            return torch.mean(torch.logsumexp(gate_logits, dim=-1) ** 2)
        return torch.tensor(0.0, device=self.gate.weight.device)
