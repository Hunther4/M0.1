"""Unit tests for QA benchmarks."""
import torch
import pytest
from unittest.mock import MagicMock, patch

from src.eval.qa import coherence_test, niah_test


class MockTokenizer:
    """Mock tokenizer for testing."""
    
    def __init__(self):
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = {}
    
    def encode(self, text: str):
        """Simple character-based encoding."""
        return [ord(c) % 256 for c in text[:100]]  # Limit to 100 chars


class TestCoherenceTest:
    """Tests for coherence_test function."""
    
    def test_coherence_returns_dict(self):
        """Test that coherence_test returns expected dictionary."""
        mock_model = MagicMock()
        mock_model.eval = MagicMock()
        mock_model.return_value = torch.zeros(1, 10, 100)
        
        tokenizer = MockTokenizer()
        prompt = "The quick brown fox jumps over the lazy dog."
        
        result = coherence_test(mock_model, prompt, tokenizer)
        
        assert isinstance(result, dict)
        assert "interval_perplexities" in result
        assert "average_coherence" in result
        assert "interval" in result
    
    def test_coherence_default_interval(self):
        """Test that default interval is 128."""
        mock_model = MagicMock()
        mock_model.eval = MagicMock()
        mock_model.return_value = torch.zeros(1, 10, 100)
        
        tokenizer = MockTokenizer()
        prompt = "Hello world"
        
        result = coherence_test(mock_model, prompt, tokenizer)
        
        assert result["interval"] == 128
    
    def test_coherence_custom_interval(self):
        """Test that custom interval is respected."""
        mock_model = MagicMock()
        mock_model.eval = MagicMock()
        mock_model.return_value = torch.zeros(1, 10, 100)
        
        tokenizer = MockTokenizer()
        prompt = "Hello world"
        
        result = coherence_test(mock_model, prompt, tokenizer, interval=64)
        
        assert result["interval"] == 64
    
    def test_coherence_with_empty_prompt(self):
        """Test coherence with empty prompt."""
        mock_model = MagicMock()
        mock_model.eval = MagicMock()
        mock_model.return_value = torch.zeros(1, 10, 100)
        
        tokenizer = MockTokenizer()
        
        # Empty prompt should not crash
        result = coherence_test(mock_model, "", tokenizer)
        
        assert isinstance(result, dict)


class TestNiahTest:
    """Tests for niah_test function."""
    
    def test_niah_returns_dict(self):
        """Test that niah_test returns expected dictionary."""
        mock_model = MagicMock()
        mock_model.eval = MagicMock()
        mock_model.return_value = torch.zeros(1, 10, 100)
        
        tokenizer = MockTokenizer()
        prompt = "The capital of France is Paris."
        needle = "Paris"
        
        result = niah_test(mock_model, prompt, needle, tokenizer)
        
        assert isinstance(result, dict)
        assert "needle" in result
        assert "avg_probability" in result
        assert "accuracy" in result
        assert "context_length" in result
    
    def test_niah_default_context_length(self):
        """Test that default context length is 512."""
        mock_model = MagicMock()
        mock_model.eval = MagicMock()
        mock_model.return_value = torch.zeros(1, 10, 100)
        
        tokenizer = MockTokenizer()
        prompt = "The secret code is 42."
        needle = "42"
        
        result = niah_test(mock_model, prompt, needle, tokenizer)
        
        assert result["context_length"] == 512
    
    def test_niah_needle_preserved(self):
        """Test that the needle is preserved in result."""
        mock_model = MagicMock()
        mock_model.eval = MagicMock()
        mock_model.return_value = torch.zeros(1, 10, 100)
        
        tokenizer = MockTokenizer()
        prompt = "Some context"
        needle = "secret123"
        
        result = niah_test(mock_model, prompt, needle, tokenizer)
        
        assert result["needle"] == needle
    
    def test_niah_accuracy_bounds(self):
        """Test that accuracy is a valid probability."""
        mock_model = MagicMock()
        mock_model.eval = MagicMock()
        # Return probs that sum to 1 per position
        probs = torch.ones(1, 10, 100) / 100
        mock_model.return_value = probs
        
        tokenizer = MockTokenizer()
        prompt = "Test prompt " * 10
        needle = "42"
        
        result = niah_test(mock_model, prompt, needle, tokenizer)
        
        assert 0 <= result["accuracy"] <= 1