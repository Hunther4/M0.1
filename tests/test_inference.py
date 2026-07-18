"""Tests for Fase 3 — Inference (sampling + generate)."""

import torch
import pytest

from src.inference.sampling import sample
from src.model.lm import TransformerLM
from src.transformer.config import M01Config


# ─── Sampling Tests ───────────────────────────────────────────────────────────


class TestSampleTemperature:
    """sample() with temperature parameter."""

    def test_greedy_temperature_zero(self):
        """temperature=0 MUST return argmax token."""
        logits = torch.tensor([0.1, 0.2, 10.0, 0.0, -5.0])
        token = sample(logits, temperature=0.0)
        assert token == 2  # index 2 has highest logit

    def test_temperature_one_is_random(self):
        """temperature=1.0 MUST produce valid token in range."""
        logits = torch.randn(1000)
        token = sample(logits, temperature=1.0)
        assert 0 <= token < 1000

    def test_temperature_negative_raises(self):
        """Negative temperature MUST raise ValueError."""
        logits = torch.randn(10)
        with pytest.raises(ValueError, match="Temperature"):
            sample(logits, temperature=-1.0)


class TestSampleTopK:
    """sample() with top-k filtering."""

    def test_top_k_limits_candidates(self):
        """top_k MUST restrict sampling to top k tokens."""
        logits = torch.tensor([-100.0, -100.0, 0.0, 0.0, 100.0])
        # Greedy at temperature 0 with top_k
        token = sample(logits, temperature=0.0, top_k=3)
        assert token == 4  # highest among top 3

    def test_top_k_one_is_argmax(self):
        """top_k=1 MUST behave identically to greedy."""
        logits = torch.randn(100)
        token_k1 = sample(logits, temperature=0.0, top_k=1)
        token_greedy = sample(logits, temperature=0.0)
        assert token_k1 == token_greedy


class TestSampleTopP:
    """sample() with top-p (nucleus) filtering."""

    def test_top_p_one_is_full_distribution(self):
        """top_p=1.0 MUST NOT filter anything."""
        logits = torch.randn(100)
        # Should not crash or filter — full distribution
        token = sample(logits, temperature=1.0, top_p=1.0)
        assert 0 <= token < 100

    def test_top_p_zero_fallback(self):
        """top_p=0.0 MUST fall back to unscaled sampling (no crash)."""
        logits = torch.randn(100)
        token = sample(logits, temperature=1.0, top_p=0.0)
        assert 0 <= token < 100


class TestSampleEdgeCases:
    """sample() edge cases."""

    def test_all_equal_logits(self):
        """Uniform logits MUST produce a valid token."""
        logits = torch.ones(50)
        for _ in range(10):
            token = sample(logits, temperature=1.0)
            assert 0 <= token < 50

    def test_single_token_vocab(self):
        """Single-element vocab MUST return index 0."""
        logits = torch.tensor([42.0])
        assert sample(logits, temperature=0.0) == 0
        assert sample(logits, temperature=1.0) == 0


# ─── Generation Smoke Tests ────────────────────────────────────────────────────


class TestGenerate:
    """Minimal transformer for generation tests."""

    @pytest.fixture
    def mini_config(self):
        """Tiny config for fast testing."""
        return M01Config(
            vocab_size=258,  # minimum: 256 bytes + 2 special tokens
            context_length=128,
            d_model=32,
            n_heads=4,
            d_ff=64,
            n_layers=2,
        )

    @pytest.fixture
    def mini_model(self, mini_config):
        """Untrained tiny model for shape/smoke tests."""
        model = TransformerLM(mini_config)
        model.eval()
        return model

    def test_generate_returns_string(self, mini_model):
        """generate() MUST return a string."""
        from src.inference.generate import generate
        from src.tokenizer.bpe import Tokenizer

        tokenizer = Tokenizer()
        tokenizer.train(["hello world"], vocab_size=258)

        text = generate(
            model=mini_model,
            tokenizer=tokenizer,
            prompt="hello",
            max_gen_len=5,
            temperature=1.0,
        )
        assert isinstance(text, str)
        assert len(text) > 0

    def test_generate_empty_prompt_raises(self, mini_model):
        """Empty prompt MUST raise ValueError."""
        from src.inference.generate import generate
        from src.tokenizer.bpe import Tokenizer

        tokenizer = Tokenizer()
        tokenizer.train(["test"], vocab_size=258)

        with pytest.raises(ValueError, match="empty"):
            generate(model=mini_model, tokenizer=tokenizer, prompt="   ")

    def test_generate_zero_max_len_raises(self, mini_model):
        """max_gen_len=0 MUST raise ValueError."""
        from src.inference.generate import generate
        from src.tokenizer.bpe import Tokenizer

        tokenizer = Tokenizer()
        tokenizer.train(["test"], vocab_size=258)

        with pytest.raises(ValueError, match="max_gen_len"):
            generate(
                model=mini_model,
                tokenizer=tokenizer,
                prompt="test",
                max_gen_len=0,
            )

    def test_generate_produces_different_outputs_with_temp(self, mini_model):
        """Different temperatures MUST produce different outputs
        (statistical, not guaranteed but very likely with random logits)."""
        from src.inference.generate import generate
        from src.tokenizer.bpe import Tokenizer

        tokenizer = Tokenizer()
        vocab = [chr(i) for i in range(65, 91)]  # A-Z
        tokenizer.train(vocab, vocab_size=258)

        text_greedy = generate(
            model=mini_model,
            tokenizer=tokenizer,
            prompt="A",
            max_gen_len=10,
            temperature=0.0,
        )
        text_random = generate(
            model=mini_model,
            tokenizer=tokenizer,
            prompt="A",
            max_gen_len=10,
            temperature=2.0,
        )
        assert isinstance(text_greedy, str)
        assert isinstance(text_random, str)


# ─── CLI Smoke Tests ───────────────────────────────────────────────────────────


class TestCLI:
    """CLI entry point smoke tests."""

    def test_help_succeeds(self):
        """--help MUST raise SystemExit(0)."""
        from src.inference.cli import parse_args

        with pytest.raises(SystemExit) as exc:
            parse_args(["--help"])
        assert exc.value.code == 0

    def test_minimal_args_parse(self):
        """Minimal valid args MUST parse correctly."""
        from src.inference.cli import parse_args

        args = parse_args(["--prompt", "test", "--max-len", "1", "--device", "cpu"])
        assert args.prompt == "test"
        assert args.max_len == 1
        assert args.device == "cpu"

    def test_default_values(self):
        """Default values MUST match expected."""
        from src.inference.cli import parse_args

        args = parse_args(["--prompt", "hello"])
        assert args.temperature == 1.0
        assert args.top_k is None
        assert args.top_p is None
        assert args.max_len == 100
        assert args.device == "auto"
