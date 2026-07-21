"""Unit tests for QA benchmarks."""
import torch
import pytest
from unittest.mock import MagicMock, patch

from src.eval.qa import coherence_test, niah_test


class MockTokenizer:
    """Mock tokenizer for testing."""
    
    def __init__(self, vocab_size: int = 100):
        self.vocab = {i: bytes([i]) for i in range(vocab_size)}
        self.merges = {}
        self.vocab_size = vocab_size
    
    def encode(self, text: str):
        """Simple character-based encoding with tokens in range [0, vocab_size)."""
        return [ord(c) % self.vocab_size for c in text[:100]]  # Limit to 100 chars


class ExactTokenizer:
    vocab_size = 256

    def encode(self, text: str):
        return [ord(c) for c in text]


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
        tokenizer = MockTokenizer()
        prompt = "The capital of France is Paris."
        needle = "Paris"
        haystack = f"{prompt} {needle}"
        encoded = tokenizer.encode(haystack)
        seq_len = len(encoded)
        
        mock_model = MagicMock()
        mock_model.eval = MagicMock()
        mock_model.return_value = torch.zeros(1, seq_len, 100)
        result = niah_test(mock_model, prompt, needle, tokenizer)
        
        assert isinstance(result, dict)
        assert "needle" in result
        assert "avg_probability" in result
        assert "accuracy" in result
        assert "context_length" in result
    
    def test_niah_default_context_length(self):
        """Test that default context length is 512."""
        tokenizer = MockTokenizer()
        prompt = "The secret code is 42."
        needle = "42"
        haystack = f"{prompt} {needle}"
        encoded = tokenizer.encode(haystack)
        seq_len = len(encoded)
        
        mock_model = MagicMock()
        mock_model.eval = MagicMock()
        mock_model.return_value = torch.zeros(1, seq_len, 100)
        result = niah_test(mock_model, prompt, needle, tokenizer)
        
        assert result["context_length"] == 512
    
    def test_niah_needle_preserved(self):
        """Test that the needle is preserved in result."""
        tokenizer = MockTokenizer()
        prompt = "Some context"
        needle = "secret123"
        haystack = f"{prompt} {needle}"
        encoded = tokenizer.encode(haystack)
        seq_len = len(encoded)
        
        mock_model = MagicMock()
        mock_model.eval = MagicMock()
        mock_model.return_value = torch.zeros(1, seq_len, 100)
        result = niah_test(mock_model, prompt, needle, tokenizer)
        
        assert result["needle"] == needle
    
    def test_niah_accuracy_bounds(self):
        """Test that accuracy is a valid probability."""
        tokenizer = MockTokenizer()
        prompt = "Test prompt " * 10
        needle = "42"
        haystack = f"{prompt} {needle}"
        encoded = tokenizer.encode(haystack)
        
        probs = torch.ones(1, len(encoded), 100) / 100
        mock_model = MagicMock()
        mock_model.eval = MagicMock()
        mock_model.return_value = probs
        
        result = niah_test(mock_model, prompt, needle, tokenizer)
        
        assert 0 <= result["accuracy"] <= 1

    def test_niah_appended_needle_survives_long_haystack(self):
        tokenizer = ExactTokenizer()
        prompt = "h" * 40
        needle = "OK"
        needle_tokens = tokenizer.encode(needle)
        context_length = 16

        class NeedleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.seen = None

            def forward(self, input_ids):
                self.seen = input_ids.detach().clone()
                logits = torch.full((1, input_ids.shape[1], 256), -20.0)
                # The causal prediction at pos-1 must score the token at pos.
                for pos, token in enumerate(needle_tokens, start=input_ids.shape[1] - len(needle_tokens)):
                    logits[0, pos - 1, token] = 20.0
                return logits

        model = NeedleModel()
        result = niah_test(model, prompt, needle, tokenizer, context_length=context_length)

        assert model.seen[0, -len(needle_tokens):].tolist() == needle_tokens
        assert result["accuracy"] == 1.0
        assert result["avg_probability"] > 0.99
