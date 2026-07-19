"""Tests for train_text_only_gpu.py script.

Approval tests that capture current behavior before refactoring.
"""

import importlib
import sys
import pytest


class TestScriptImport:
    """Verify script can be imported and has expected functions."""

    def test_import_module(self) -> None:
        """Module MUST be importable."""
        module = importlib.import_module("scripts.training.train_text_only_gpu")
        assert module is not None

    def test_evaluate_val_loss_exists(self) -> None:
        """evaluate_val_loss function MUST exist."""
        module = importlib.import_module("scripts.training.train_text_only_gpu")
        assert hasattr(module, "evaluate_val_loss")
        assert callable(module.evaluate_val_loss)

    def test_run_story_test_exists(self) -> None:
        """run_story_test function MUST exist."""
        module = importlib.import_module("scripts.training.train_text_only_gpu")
        assert hasattr(module, "run_story_test")
        assert callable(module.run_story_test)

    def test_main_exists(self) -> None:
        """main function MUST exist."""
        module = importlib.import_module("scripts.training.train_text_only_gpu")
        assert hasattr(module, "main")
        assert callable(module.main)

    def test_imports_from_expected_modules(self) -> None:
        """Script MUST import from expected modules."""
        module = importlib.import_module("scripts.training.train_text_only_gpu")
        # Check that the module has imported expected names
        assert hasattr(module, "M01Config")
        assert hasattr(module, "TransformerLM")
        assert hasattr(module, "Tokenizer")
        assert hasattr(module, "TinyShakespeareDataset")
        assert hasattr(module, "TrainingConfig")
        assert hasattr(module, "DataLoader")
        assert hasattr(module, "AdamW")