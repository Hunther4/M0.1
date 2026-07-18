"""Tests for src.model package init.

Verifies that src.model exports work correctly.
"""


class TestModelInit:
    """src.model MUST export expected components."""

    def test_import_rms_norm(self) -> None:
        """RMSNorm MUST be importable from src.model."""
        from src.model import RMSNorm
        assert RMSNorm is not None

    def test_import_with_all(self) -> None:
        """__all__ MUST include RMSNorm."""
        import src.model
        assert hasattr(src.model, "__all__")
        assert "RMSNorm" in src.model.__all__
