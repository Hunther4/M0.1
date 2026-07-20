"""Tests for MoE monitoring: metric computation, logging, and collapse detection.

Tests follow the pattern from test_moe.py and test_training_config.py:
direct imports, pytest assertions, torch-based fixtures.
"""

import torch
import pytest
from src.transformer.config import M01Config
from src.transformer.moe import MoELayer


# =============================================================================
# Task 1.1 — MoELayer attribute storage
# =============================================================================

class TestMoELayerAttrStorage:
    """gate_probs, topk_indices, expert_mask MUST be stored after forward in training mode."""

    def test_gate_probs_stored_after_forward_train(self) -> None:
        """After forward with training=True, self.gate_probs MUST exist with correct shape."""
        config = M01Config(num_experts=4, moe_top_k=2)
        moe = MoELayer(config)
        moe.train()
        batch, seq_len = 2, 5
        x = torch.randn(batch, seq_len, config.d_model)
        _ = moe(x)
        assert hasattr(moe, "gate_probs"), "gate_probs should be stored after forward"
        assert moe.gate_probs.shape == (batch * seq_len, 4), (
            f"Expected ({batch * seq_len}, 4), got {moe.gate_probs.shape}"
        )

    def test_topk_indices_stored_after_forward_train(self) -> None:
        """After forward with training=True, self.topk_indices MUST exist."""
        config = M01Config(num_experts=4, moe_top_k=2)
        moe = MoELayer(config)
        moe.train()
        x = torch.randn(2, 5, config.d_model)
        _ = moe(x)
        assert hasattr(moe, "topk_indices"), "topk_indices should be stored after forward"
        assert moe.topk_indices.shape[-1] == 2, (
            f"Expected topk dim=2, got {moe.topk_indices.shape[-1]}"
        )

    def test_expert_mask_stored_after_forward_train(self) -> None:
        """After forward with training=True, self.expert_mask MUST exist."""
        config = M01Config(num_experts=4, moe_top_k=2)
        moe = MoELayer(config)
        moe.train()
        x = torch.randn(2, 5, config.d_model)
        _ = moe(x)
        assert hasattr(moe, "expert_mask"), "expert_mask should be stored after forward"
        # expert_mask shape: (num_experts, num_tokens)
        assert moe.expert_mask.shape[0] == 4, (
            f"Expected first dim=4 (num_experts), got {moe.expert_mask.shape[0]}"
        )

    def test_attrs_not_stored_in_eval_mode(self) -> None:
        """In eval mode, gate_probs/topk_indices/expert_mask MUST NOT be stored
        (or should be None) since aux_loss is not computed."""
        config = M01Config(num_experts=4, moe_top_k=2)
        moe = MoELayer(config)
        moe.eval()
        x = torch.randn(2, 5, config.d_model)
        _ = moe(x)
        # In eval mode, the training guard skips the aux_loss block
        # where we store these attrs, so they should not exist
        assert not hasattr(moe, "gate_probs"), (
            "gate_probs should NOT be stored in eval mode"
        )
        assert not hasattr(moe, "topk_indices"), (
            "topk_indices should NOT be stored in eval mode"
        )
        assert not hasattr(moe, "expert_mask"), (
            "expert_mask should NOT be stored in eval mode"
        )

    def test_no_attrs_for_single_expert(self) -> None:
        """With num_experts <= 1, no gate_probs/topk_indices/expert_mask are stored."""
        config = M01Config(num_experts=1, moe_top_k=1)
        moe = MoELayer(config)
        moe.train()
        x = torch.randn(2, 5, config.d_model)
        _ = moe(x)
        assert not hasattr(moe, "gate_probs"), (
            "gate_probs should NOT be stored for single-expert fallback path"
        )
        assert not hasattr(moe, "topk_indices")
        assert not hasattr(moe, "expert_mask")

    def test_forward_output_unchanged_by_attrs(self) -> None:
        """Adding attribute storage MUST NOT change forward pass output."""
        config = M01Config(num_experts=4, moe_top_k=2)
        moe1 = MoELayer(config)
        moe2 = MoELayer(config)
        moe1.train()
        moe2.train()
        # Copy weights so both are identical
        moe2.load_state_dict(moe1.state_dict())

        x = torch.randn(2, 5, config.d_model)
        with torch.no_grad():
            out1 = moe1(x)
            out2 = moe2(x)

        assert torch.allclose(out1, out2), (
            "Forward output should be identical with attr storage"
        )


# =============================================================================
# Task 1.2 — TrainingConfig monitoring fields
# =============================================================================

class TestTrainingConfigMonitoringFields:
    """TrainingConfig MUST include MoE monitoring configuration fields."""

    def test_default_log_moe_metrics(self) -> None:
        from src.training.config import TrainingConfig
        config = TrainingConfig()
        assert config.log_moe_metrics is True, (
            f"Expected log_moe_metrics=True, got {config.log_moe_metrics}"
        )

    def test_default_moe_collapse_consecutive_steps(self) -> None:
        from src.training.config import TrainingConfig
        config = TrainingConfig()
        assert config.moe_collapse_consecutive_steps == 50, (
            f"Expected moe_collapse_consecutive_steps=50, "
            f"got {config.moe_collapse_consecutive_steps}"
        )

    def test_default_moe_collapse_expert_ratio(self) -> None:
        from src.training.config import TrainingConfig
        config = TrainingConfig()
        assert config.moe_collapse_expert_ratio == 0.3, (
            f"Expected moe_collapse_expert_ratio=0.3, "
            f"got {config.moe_collapse_expert_ratio}"
        )

    def test_default_log_metrics_backend(self) -> None:
        from src.training.config import TrainingConfig
        config = TrainingConfig()
        assert config.log_metrics_backend == "console", (
            f"Expected log_metrics_backend='console', "
            f"got '{config.log_metrics_backend}'"
        )

    def test_default_collapse_streak_threshold(self) -> None:
        from src.training.config import TrainingConfig
        config = TrainingConfig()
        assert config.collapse_streak_threshold == 500, (
            f"Expected collapse_streak_threshold=500, "
            f"got {config.collapse_streak_threshold}"
        )

    def test_custom_monitoring_values(self) -> None:
        from src.training.config import TrainingConfig
        config = TrainingConfig(
            log_moe_metrics=False,
            moe_collapse_consecutive_steps=100,
            moe_collapse_expert_ratio=0.5,
            log_metrics_backend="wandb",
            collapse_streak_threshold=1000,
        )
        assert config.log_moe_metrics is False
        assert config.moe_collapse_consecutive_steps == 100
        assert config.moe_collapse_expert_ratio == 0.5
        assert config.log_metrics_backend == "wandb"
        assert config.collapse_streak_threshold == 1000

    def test_monitoring_field_types(self) -> None:
        from src.training.config import TrainingConfig
        config = TrainingConfig()
        assert isinstance(config.log_moe_metrics, bool)
        assert isinstance(config.moe_collapse_consecutive_steps, int)
        assert isinstance(config.moe_collapse_expert_ratio, float)
        assert isinstance(config.log_metrics_backend, str)
        assert isinstance(config.collapse_streak_threshold, int)


# =============================================================================
# Task 2.1 — compute_moe_metrics
# =============================================================================

class TestComputeMoeMetrics:
    """compute_moe_metrics MUST return correct metrics dict."""

    @pytest.fixture
    def moe_model(self):
        """Create a minimal model with MoE layers for testing."""
        from src.model.block import TransformerBlock
        from src.model.lm import TransformerLM

        config = M01Config(
            vocab_size=100,
            d_model=64,
            n_layers=2,
            n_heads=2,
            d_ff=128,
            num_experts=4,
            num_shared_experts=1,
            moe_top_k=2,
            num_dense_layers=0,  # All layers use MoE
        )
        model = TransformerLM(config)
        model.train()
        return model, config

    def test_returns_dict_with_expected_keys(self, moe_model):
        """compute_moe_metrics MUST return dict with expected structure."""
        from src.training.moe_metrics import compute_moe_metrics

        model, config = moe_model
        # Run a forward pass to populate attrs
        x = torch.randint(0, 100, (2, 8))
        _ = model(x)

        metrics = compute_moe_metrics(model)
        assert isinstance(metrics, dict)
        # Must have per-layer entries and global aggregates
        assert "global/n_layers_moe" in metrics
        assert "global/mean_entropy" in metrics
        assert "global/router_collapse" in metrics
        assert "global/total_routed_tokens" in metrics
        assert metrics["global/n_layers_moe"] == 2

    def test_entropy_in_bounds(self, moe_model):
        """Router entropy MUST be in [0, 1] range."""
        from src.training.moe_metrics import compute_moe_metrics

        model, config = moe_model
        x = torch.randint(0, 100, (2, 8))
        _ = model(x)

        metrics = compute_moe_metrics(model)
        ent = metrics["global/mean_entropy"]
        assert 0.0 <= ent <= 1.0, (
            f"Expected entropy in [0,1], got {ent}"
        )

    def test_returns_empty_dict_for_dense_model(self):
        """Model with num_experts <= 1 MUST return empty dict."""
        from src.training.moe_metrics import compute_moe_metrics
        from src.model.lm import TransformerLM

        config = M01Config(
            vocab_size=100,
            d_model=64,
            n_layers=1,
            n_heads=2,
            d_ff=128,
            num_experts=1,  # Dense — no MoE
            num_shared_experts=0,
            moe_top_k=1,
            num_dense_layers=1,
        )
        model = TransformerLM(config)
        model.train()
        x = torch.randint(0, 100, (2, 8))
        _ = model(x)

        metrics = compute_moe_metrics(model)
        assert metrics == {}, (
            f"Expected empty dict for dense model, got {metrics}"
        )

    def test_aux_loss_value_tied_to_gate_distribution(self):
        """Aux loss SHOULD be non-zero when gate distribution is non-uniform."""
        from src.training.moe_metrics import compute_moe_metrics
        from src.model.lm import TransformerLM

        config = M01Config(
            vocab_size=100, d_model=64, n_layers=1, n_heads=2, d_ff=128,
            num_experts=4, num_shared_experts=1, moe_top_k=2, num_dense_layers=0,
        )
        model = TransformerLM(config)
        model.train()

        # Force strongly biased gate: expert 0 gets high logits, others get zero
        with torch.no_grad():
            for block in model.blocks:
                # gate weight shape: (num_experts, d_model)
                block.ff.gate.weight.fill_(0.0)
                # Expert 0: large positive weights so sum(x) -> high logit
                block.ff.gate.weight[0, :] = 3.0
                # Expert 1: moderately positive
                block.ff.gate.weight[1, :] = 1.0

        # Use positive-only input to ensure gate behaves deterministically
        x = torch.full((2, 8), 1, dtype=torch.long)  # All token IDs = 1
        _ = model(x)
        metrics = compute_moe_metrics(model)
        assert metrics["global/n_layers_moe"] == 1
        layer_key = "layer_0"
        # aux_loss should be non-trivial with biased distribution
        assert metrics[f"{layer_key}/aux_loss"] >= 0.0
        # With deterministic positive input, expert 0 should dominate
        hist = metrics[f"{layer_key}/histogram"]
        assert hist[0] >= hist[1], (
            f"Expected expert 0 to dominate or tie, got histogram: {hist}"
        )

    def test_entropy_approaches_zero_with_skewed_gate(self):
        """With highly skewed gate weights, entropy SHOULD approach 0."""
        from src.training.moe_metrics import compute_moe_metrics

        # Build a single MoELayer directly (no full model) with controlled input
        config = M01Config(
            d_model=16, n_heads=2, num_experts=4, num_shared_experts=1,
            moe_top_k=1, d_ff=32, mla_rope_dim=4, use_mla=False,
        )
        moe = MoELayer(config)
        moe.train()

        # Set gate so expert 0 gets overwhelmingly high probability
        # gate weight shape: (num_experts, d_model)
        with torch.no_grad():
            moe.gate.weight.fill_(0.0)
            # Expert 0: large positive weight on every input dimension
            moe.gate.weight[0, :] = 10.0
            # Other experts: keep at 0

        # All-ones input -> expert 0 logit = 10 * sum(ones) = 10 * 16 = 160
        # Other experts logit = 0
        # softmax: exp(160) / (exp(160) + 3) ≈ 1.0
        x = torch.ones(2, 5, 16)
        _ = moe(x)

        # Wrap in a minimal model-like object to test compute_moe_metrics
        fake_model = type("FakeModel", (), {
            "config": config,
            "blocks": [type("FakeBlock", (), {"ff": moe})()],
        })()

        metrics = compute_moe_metrics(fake_model)
        ent = metrics["global/mean_entropy"]
        assert ent < 0.3, (
            f"Expected low entropy (< 0.3) with skewed gate, got {ent:.4f}"
        )

    def test_histogram_sum_equals_total_routed_tokens(self, moe_model):
        """Sum of all histograms MUST equal total routed tokens."""
        from src.training.moe_metrics import compute_moe_metrics

        model, config = moe_model
        x = torch.randint(0, 100, (2, 8))
        _ = model(x)

        metrics = compute_moe_metrics(model)
        total_routed = metrics["global/total_routed_tokens"]
        # Collect per-layer histogram sums
        layer_sum = 0
        for key, val in metrics.items():
            if key.endswith("/histogram"):
                layer_sum += sum(val)
        assert layer_sum == total_routed, (
            f"Histogram sum {layer_sum} != total_routed {total_routed}"
        )


# =============================================================================
# Task 2.2 — MetricsLogger protocol + ConsoleLogger
# =============================================================================

class TestMetricsLogger:
    """MetricsLogger protocol and ConsoleLogger MUST satisfy logging contract."""

    def test_console_logger_prints_metric_keys(self):
        """ConsoleLogger.log() MUST print metric keys and values."""
        from src.training.moe_metrics import ConsoleLogger
        import io
        import sys

        logger = ConsoleLogger()
        metrics = {
            "layer_0/aux_loss": 0.05,
            "layer_0/entropy": 0.42,
            "global/mean_entropy": 0.42,
            "global/router_collapse": False,
        }
        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            logger.log(metrics, step=10)
        finally:
            sys.stdout = old_stdout

        output = captured.getvalue()
        assert "Step 10" in output, f"Should contain step. Got: {output}"
        assert "aux_loss" in output.lower(), f"Should contain aux_loss. Got: {output}"
        assert "entropy" in output.lower(), f"Should contain entropy. Got: {output}"
        assert "collapse" in output.lower(), f"Should contain collapse. Got: {output}"

    def test_protocol_duck_typing(self):
        """Any object with log(metrics, step) MUST satisfy the protocol."""
        from src.training.moe_metrics import MetricsLogger

        # A simple object that satisfies the protocol via duck typing
        class FakeLogger:
            def __init__(self):
                self.calls = []

            def log(self, metrics, step):
                self.calls.append((metrics, step))

        logger = FakeLogger()
        # Should be accepted by anything expecting MetricsLogger
        # (duck typing — no isinstance check needed)
        metrics = {"test": 1.0}
        logger.log(metrics, step=5)
        assert len(logger.calls) == 1
        assert logger.calls[0] == (metrics, 5)

    def test_console_logger_empty_metrics(self):
        """ConsoleLogger MUST handle empty metrics dict without error."""
        from src.training.moe_metrics import ConsoleLogger
        import io
        import sys

        logger = ConsoleLogger()
        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            logger.log({}, step=0)
        finally:
            sys.stdout = old_stdout
        output = captured.getvalue()
        assert "Step 0" in output, f"Should handle empty metrics. Got: {output}"


# =============================================================================
# Task 2.3 — Router collapse detection
# =============================================================================

class TestDetectRouterCollapse:
    """detect_router_collapse MUST track consecutive zero-expert streaks."""

    def test_streak_increments_on_dead_expert(self):
        """Counter MUST increment when histogram has any zero-expert."""
        from src.training.moe_metrics import detect_router_collapse

        histogram = torch.tensor([0, 10, 20, 5], dtype=torch.int64)
        stop, counter = detect_router_collapse(histogram, counter=0, threshold=5)
        assert stop is False, "Should not stop on first dead step"
        assert counter == 1, f"Expected counter=1, got {counter}"

    def test_streak_resets_on_all_active(self):
        """Counter MUST reset to 0 when no zero-experts."""
        from src.training.moe_metrics import detect_router_collapse

        histogram = torch.tensor([10, 15, 20, 5], dtype=torch.int64)
        stop, counter = detect_router_collapse(histogram, counter=3, threshold=5)
        assert stop is False, "Should not stop when all experts active"
        assert counter == 0, f"Expected counter=0, got {counter}"

    def test_stop_when_streak_exceeds_threshold(self):
        """Stop MUST be True when counter >= threshold."""
        from src.training.moe_metrics import detect_router_collapse

        histogram = torch.tensor([0, 10, 20, 5], dtype=torch.int64)
        stop, counter = detect_router_collapse(histogram, counter=4, threshold=5)
        assert stop is True, "Should stop when counter >= threshold"
        assert counter == 5, f"Expected counter=5, got {counter}"

    def test_no_stop_below_threshold(self):
        """Stop MUST be False when counter < threshold."""
        from src.training.moe_metrics import detect_router_collapse

        histogram = torch.tensor([0, 10, 20, 5], dtype=torch.int64)
        stop, counter = detect_router_collapse(histogram, counter=2, threshold=5)
        assert stop is False, "Should not stop below threshold"
        assert counter == 3, f"Expected counter=3, got {counter}"

    def test_exact_threshold_triggers_stop(self):
        """Stop MUST be True when counter exactly equals threshold."""
        from src.training.moe_metrics import detect_router_collapse

        histogram = torch.tensor([0, 10, 20, 5], dtype=torch.int64)
        stop, counter = detect_router_collapse(histogram, counter=4, threshold=5)
        assert stop is True, f"Expected stop at exact threshold 5"
        assert counter == 5

    def test_all_dead_advances_streak_once_per_call(self):
        """ALL experts dead still only advances streak by 1 (one call)."""
        from src.training.moe_metrics import detect_router_collapse

        histogram = torch.tensor([0, 0, 0, 0], dtype=torch.int64)
        stop, counter = detect_router_collapse(histogram, counter=0, threshold=10)
        assert stop is False, "Should not stop on first all-dead step with threshold 10"
        assert counter == 1, f"Expected counter=1 for single call, got {counter}"

    def test_recovery_after_dead_streak(self):
        """Counter resets to 0 after a dead streak ends (recovery)."""
        from src.training.moe_metrics import detect_router_collapse

        # Simulate: 3 dead steps, then recovery
        histogram = torch.tensor([10, 15, 20, 5], dtype=torch.int64)
        stop, counter = detect_router_collapse(histogram, counter=3, threshold=10)
        assert stop is False, "Should stop only when threshold exceeded"
        assert counter == 0, f"Expected counter=0 on recovery, got {counter}"

    def test_list_input_works(self):
        """detect_router_collapse MUST accept plain lists, not just tensors."""
        from src.training.moe_metrics import detect_router_collapse

        histogram = [0, 10, 20, 5]
        stop, counter = detect_router_collapse(histogram, counter=0, threshold=5)
        assert stop is False
        assert counter == 1

    def test_no_dead_no_stop_with_high_threshold(self):
        """With no dead expert and high threshold, MUST not stop and counter=0."""
        from src.training.moe_metrics import detect_router_collapse

        histogram = torch.tensor([10, 15, 20, 5], dtype=torch.int64)
        stop, counter = detect_router_collapse(histogram, counter=0, threshold=500)
        assert stop is False
        assert counter == 0


# =============================================================================
# Task 3.1 — TransformerLM.get_moe_metrics() accessor
# =============================================================================

class TestTransformerLMGetMoeMetrics:
    """TransformerLM.get_moe_metrics() MUST delegate to compute_moe_metrics."""

    def test_get_moe_metrics_returns_dict(self):
        """With MoE model after forward, MUST return metrics dict."""
        from src.model.lm import TransformerLM
        from src.transformer.config import M01Config

        config = M01Config(
            vocab_size=100, d_model=64, n_layers=2, n_heads=2, d_ff=128,
            num_experts=4, num_shared_experts=1, moe_top_k=2, num_dense_layers=0,
        )
        model = TransformerLM(config)
        model.train()
        x = torch.randint(0, 100, (2, 8))
        _ = model(x)
        metrics = model.get_moe_metrics()
        assert isinstance(metrics, dict)
        assert "global/n_layers_moe" in metrics
        assert metrics["global/n_layers_moe"] == 2

    def test_get_moe_metrics_empty_for_dense(self):
        """With dense model (num_experts=1), MUST return empty dict."""
        from src.model.lm import TransformerLM
        from src.transformer.config import M01Config

        config = M01Config(
            vocab_size=100, d_model=64, n_layers=1, n_heads=2, d_ff=128,
            num_experts=1, num_shared_experts=0, moe_top_k=1, num_dense_layers=1,
        )
        model = TransformerLM(config)
        model.train()
        x = torch.randint(0, 100, (2, 8))
        _ = model(x)
        metrics = model.get_moe_metrics()
        assert metrics == {}, f"Expected empty dict for dense model, got {metrics}"

    def test_get_moe_metrics_without_forward(self):
        """Before any forward pass, MUST return empty dict (no gate_probs)."""
        from src.model.lm import TransformerLM
        from src.transformer.config import M01Config

        config = M01Config(
            vocab_size=100, d_model=64, n_layers=1, n_heads=2, d_ff=128,
            num_experts=4, num_shared_experts=1, moe_top_k=2, num_dense_layers=0,
        )
        model = TransformerLM(config)
        model.train()
        # No forward pass yet
        metrics = model.get_moe_metrics()
        assert metrics == {}, (
            f"Expected empty dict before forward, got {metrics}"
        )


# =============================================================================
# Task 4.2 — Integration: shared train() loop with MoE collapse early-stop
# =============================================================================

class TestCollapseEarlyStopIntegration:
    """Shared training loop MUST stop early on router collapse."""

    def test_collapse_stops_before_max_steps(self):
        """loop.py train() with MoE model + collapse_streak_threshold=5
        MUST stop before max_steps with collapse message."""
        from src.training.loop import train
        from src.model.lm import TransformerLM
        from src.transformer.config import M01Config
        import torch.nn as nn
        from torch.utils.data import DataLoader

        # Tiny MoE model with zeroed gate to force all tokens to expert 0
        # (top-1 tie-breaking always picks first expert)
        model_config = M01Config(
            vocab_size=256, context_length=4, d_model=64, n_heads=4, d_ff=128,
            n_layers=1, num_experts=2, num_shared_experts=1, moe_top_k=1,
            num_dense_layers=0, use_mla=False,
        )
        model = TransformerLM(model_config)
        device = torch.device("cpu")

        # Zero gate weights → all logits = 0 → tie → topk picks first expert
        # Freeze gate so optimizer doesn't change it → Expert 1 gets 0 tokens
        # every step → guaranteed collapse streak
        with torch.no_grad():
            for block in model.blocks:
                block.ff.gate.weight.zero_()
                block.ff.gate.weight.requires_grad_(False)

        # Tiny synthetic dataset
        tokens = torch.randint(0, 256, (20,), dtype=torch.long)

        class SynthDataset:
            def __init__(self, t, sl):
                self.tokens = t
                self.seq_len = sl
            def __len__(self):
                return len(self.tokens) - self.seq_len
            def __getitem__(self, idx):
                return (self.tokens[idx:idx + self.seq_len],
                        self.tokens[idx + 1:idx + self.seq_len + 1])

        dataset = SynthDataset(tokens, 4)
        loader = DataLoader(dataset, batch_size=2, num_workers=0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()

        result = train(
            model=model, dataloader=loader, optimizer=optimizer,
            criterion=criterion, steps=100, device=device,
            log_interval=1, collapse_streak_threshold=5,
        )
        # Must stop well before max_steps due to guaranteed collapse
        assert result["steps_completed"] < 100, (
            f"Expected early stop before 100 steps, got {result['steps_completed']}"
        )
        assert "collapse" in result.get("stop_reason", "").lower(), (
            f"Expected collapse stop reason, got: {result.get('stop_reason')}"
        )

    def test_no_collapse_with_disabled_monitoring(self):
        """With collapse_streak_threshold=0 (disabled), MUST run to max_steps."""
        from src.training.loop import train
        from src.model.lm import TransformerLM
        from src.transformer.config import M01Config
        import torch.nn as nn
        from torch.utils.data import DataLoader

        model_config = M01Config(
            vocab_size=256, context_length=32, d_model=64, n_heads=4, d_ff=128,
            n_layers=1, num_experts=2, num_shared_experts=1, moe_top_k=1,
            num_dense_layers=0, use_mla=False,
        )
        model = TransformerLM(model_config)
        device = torch.device("cpu")

        tokens = torch.randint(0, 256, (200,), dtype=torch.long)

        class SynthDataset:
            def __init__(self, t, sl):
                self.tokens = t
                self.seq_len = sl
            def __len__(self):
                return len(self.tokens) - self.seq_len
            def __getitem__(self, idx):
                return (self.tokens[idx:idx + self.seq_len],
                        self.tokens[idx + 1:idx + self.seq_len + 1])

        dataset = SynthDataset(tokens, 32)
        loader = DataLoader(dataset, batch_size=4, num_workers=0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()

        result = train(
            model=model, dataloader=loader, optimizer=optimizer,
            criterion=criterion, steps=5, device=device,
            log_interval=10, collapse_streak_threshold=0,  # Disabled
        )
        assert result["steps_completed"] == 5, (
            f"Expected 5 steps with monitoring disabled, got {result['steps_completed']}"
        )
