"""Unit tests for evaluation metrics."""
import torch
import pytest
from unittest.mock import MagicMock

from src.eval.metrics import calculate_perplexity, log_loss


class TestCalculatePerplexity:
    """Tests for calculate_perplexity function."""
    
    def test_perplexity_with_mock_model(self):
        """Test perplexity calculation with a mock model."""
        # Create a mock model that returns fixed logits
        mock_model = MagicMock()
        
        # Create dummy input
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])  # 5 tokens
        
        # Mock logits: shape [batch, seq_len, vocab_size]
        # Model outputs logits for all input tokens (seq_len = input length)
        vocab_size = 100
        seq_len = 5  # 5 tokens -> 5 output positions
        mock_logits = torch.zeros(1, seq_len, vocab_size)
        # Make one token much more likely to get non-trivial loss
        mock_logits[0, 0, 10] = 5.0  # High logit for token 10
        
        mock_model.return_value = mock_logits
        
        # Should not raise
        perplexity = calculate_perplexity(mock_model, input_ids)
        
        assert isinstance(perplexity, float)
        assert perplexity > 0
    
    def test_perplexity_handles_1d_input(self):
        """Test that 1D input is properly expanded to 2D."""
        mock_model = MagicMock()
        
        input_ids_1d = torch.tensor([1, 2, 3, 4, 5])
        
        # Mock logits: seq_len = 5 for 5 tokens
        mock_logits = torch.zeros(1, 5, 100)
        mock_model.return_value = mock_logits
        
        perplexity = calculate_perplexity(mock_model, input_ids_1d)
        
        assert isinstance(perplexity, float)
        assert perplexity > 0
    
    def test_perplexity_with_attention_mask(self):
        """Test perplexity calculation with attention mask."""
        mock_model = MagicMock()
        
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        attention_mask = torch.tensor([[1, 1, 1, 1, 1]])
        
        # seq_len = 5 for 5 tokens
        mock_logits = torch.zeros(1, 5, 100)
        mock_model.return_value = mock_logits
        
        perplexity = calculate_perplexity(mock_model, input_ids, attention_mask)
        
        assert isinstance(perplexity, float)


class TestLogLoss:
    """Tests for log_loss function."""
    
    def test_log_loss_returns_dict(self):
        """Test that log_loss returns expected dictionary structure."""
        mock_model = MagicMock()
        
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])  # 5 tokens
        # seq_len = 5 for 5 tokens
        mock_logits = torch.zeros(1, 5, 100)
        mock_model.return_value = mock_logits
        
        result = log_loss(mock_model, input_ids)
        
        assert isinstance(result, dict)
        assert "loss" in result
        assert "perplexity" in result
        assert "token_count" in result
    
    def test_log_loss_token_count(self):
        """Test that token_count is correct."""
        mock_model = MagicMock()
        
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])  # 5 tokens
        # shift gives 4 predictions
        mock_logits = torch.zeros(1, 5, 100)
        mock_model.return_value = mock_logits
        
        result = log_loss(mock_model, input_ids)
        
        assert result["token_count"] == 4  # shift_labels has 4 elements
    
    def test_log_loss_perplexity_matches_calculate(self):
        """Test that log_loss perplexity matches calculate_perplexity."""
        mock_model = MagicMock()
        
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        mock_logits = torch.zeros(1, 5, 100)
        mock_model.return_value = mock_logits
        
        result = log_loss(mock_model, input_ids)
        perplexity = calculate_perplexity(mock_model, input_ids)
        
        assert abs(result["perplexity"] - perplexity) < 1e-6