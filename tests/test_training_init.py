"""Tests for src.training package init.

Verifies that src.training exports work correctly.
"""


class TestTrainingInit:
    """src.training MUST export expected components."""

    def test_import_training_config(self) -> None:
        """TrainingConfig MUST be importable from src.training."""
        from src.training import TrainingConfig
        assert TrainingConfig is not None

    def test_import_with_all(self) -> None:
        """__all__ MUST include TrainingConfig."""
        import src.training
        assert hasattr(src.training, "__all__")
        assert "TrainingConfig" in src.training.__all__
