"""Tests for TrainingConfig dataclass.

TrainingConfig requirements:
- Default values MUST match the spec
- Fields MUST have correct types
- Custom values MUST be accepted
- Instance MUST be a valid dataclass
"""

import pytest
from src.training.config import TrainingConfig


class TestTrainingConfigDefaults:
    """Default values MUST match the design spec."""

    def test_default_batch_size(self) -> None:
        config = TrainingConfig()
        assert config.batch_size == 4, (
            f"Expected batch_size=4, got {config.batch_size}"
        )

    def test_default_seq_len(self) -> None:
        config = TrainingConfig()
        assert config.seq_len == 1024, (
            f"Expected seq_len=1024, got {config.seq_len}"
        )

    def test_default_max_lr(self) -> None:
        config = TrainingConfig()
        assert config.max_lr == 3e-4, (
            f"Expected max_lr=3e-4, got {config.max_lr}"
        )

    def test_default_min_lr_ratio(self) -> None:
        config = TrainingConfig()
        assert config.min_lr_ratio == 0.1, (
            f"Expected min_lr_ratio=0.1, got {config.min_lr_ratio}"
        )

    def test_default_warmup_steps(self) -> None:
        config = TrainingConfig()
        assert config.warmup_steps == 200, (
            f"Expected warmup_steps=200, got {config.warmup_steps}"
        )

    def test_default_max_steps(self) -> None:
        config = TrainingConfig()
        assert config.max_steps == 100_000, (
            f"Expected max_steps=100000, got {config.max_steps}"
        )

    def test_default_weight_decay(self) -> None:
        config = TrainingConfig()
        assert config.weight_decay == 0.1, (
            f"Expected weight_decay=0.1, got {config.weight_decay}"
        )

    def test_default_beta1(self) -> None:
        config = TrainingConfig()
        assert config.beta1 == 0.9, (
            f"Expected beta1=0.9, got {config.beta1}"
        )

    def test_default_beta2(self) -> None:
        config = TrainingConfig()
        assert config.beta2 == 0.95, (
            f"Expected beta2=0.95, got {config.beta2}"
        )

    def test_default_max_norm(self) -> None:
        config = TrainingConfig()
        assert config.max_norm == 1.0, (
            f"Expected max_norm=1.0, got {config.max_norm}"
        )

    def test_default_log_interval(self) -> None:
        config = TrainingConfig()
        assert config.log_interval == 10, (
            f"Expected log_interval=10, got {config.log_interval}"
        )

    def test_default_save_interval(self) -> None:
        config = TrainingConfig()
        assert config.save_interval == 500, (
            f"Expected save_interval=500, got {config.save_interval}"
        )

    def test_default_checkpoint_dir(self) -> None:
        config = TrainingConfig()
        assert config.checkpoint_dir == "checkpoints", (
            f"Expected checkpoint_dir='checkpoints', got '{config.checkpoint_dir}'"
        )

    def test_default_data_dir(self) -> None:
        config = TrainingConfig()
        assert config.data_dir == "data", (
            f"Expected data_dir='data', got '{config.data_dir}'"
        )


class TestTrainingConfigCustom:
    """Custom values MUST override defaults."""

    def test_custom_values(self) -> None:
        config = TrainingConfig(
            batch_size=8,
            seq_len=512,
            max_lr=1e-3,
            checkpoint_dir="/tmp/ckpt",
            data_dir="/tmp/data",
        )
        assert config.batch_size == 8
        assert config.seq_len == 512
        assert config.max_lr == 1e-3
        assert config.checkpoint_dir == "/tmp/ckpt"
        assert config.data_dir == "/tmp/data"
        # Fields not set should keep defaults
        assert config.max_steps == 100_000


class TestTrainingConfigTypes:
    """Field types MUST match the spec."""

    def test_int_fields(self) -> None:
        config = TrainingConfig()
        assert isinstance(config.batch_size, int)
        assert isinstance(config.seq_len, int)
        assert isinstance(config.warmup_steps, int)
        assert isinstance(config.max_steps, int)
        assert isinstance(config.log_interval, int)
        assert isinstance(config.save_interval, int)

    def test_float_fields(self) -> None:
        config = TrainingConfig()
        assert isinstance(config.max_lr, float)
        assert isinstance(config.min_lr_ratio, float)
        assert isinstance(config.weight_decay, float)
        assert isinstance(config.beta1, float)
        assert isinstance(config.beta2, float)
        assert isinstance(config.max_norm, float)

    def test_str_fields(self) -> None:
        config = TrainingConfig()
        assert isinstance(config.checkpoint_dir, str)
        assert isinstance(config.data_dir, str)


class TestTrainingConfigDataclass:
    """TrainingConfig MUST behave as a proper dataclass."""

    def test_is_dataclass(self) -> None:
        from dataclasses import dataclass
        config = TrainingConfig()
        # Can be used with dataclasses functions
        import dataclasses
        fields = dataclasses.fields(config)
        assert len(fields) == 19, (
            f"Expected 19 fields, got {len(fields)}"
        )

    def test_repr_defaults(self) -> None:
        """repr() MUST include field values."""
        config = TrainingConfig()
        r = repr(config)
        assert "batch_size=4" in r
        assert "max_lr=0.0003" in r

    def test_equality(self) -> None:
        """Two instances with same values MUST be equal."""
        a = TrainingConfig()
        b = TrainingConfig()
        assert a == b

    def test_inequality(self) -> None:
        """Two instances with different values MUST NOT be equal."""
        a = TrainingConfig()
        b = TrainingConfig(batch_size=8)
        assert a != b
