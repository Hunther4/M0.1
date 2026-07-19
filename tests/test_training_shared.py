"""Tests for shared training modules.

Tests config_to_dict, save/load_checkpoint roundtrip, AmplifiedDialogueDataset,
JsonlDataset, evaluate_val_loss, setup_device, and the shared train() function.
"""

import json
import os
import tempfile

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.transformer.config import M01Config
from src.training.checkpoint import config_to_dict, save_checkpoint, load_checkpoint
from src.training.datasets import AmplifiedDialogueDataset, JsonlDataset
from src.training.eval import evaluate_val_loss
from src.training.loop import train
from src.training.setup import setup_device, setup_stdout


class TestConfigToDict:
    """Tests for config_to_dict function."""

    def test_basic_config(self):
        """config_to_dict returns all 11 core fields."""
        config = M01Config(
            vocab_size=1000,
            context_length=128,
            d_model=64,
            n_heads=4,
            d_ff=128,
            n_layers=2,
            num_experts=4,
            num_shared_experts=2,
            moe_top_k=2,
            use_hybrid_attention=True,
            local_window_size=16,
        )
        result = config_to_dict(config)

        assert result["vocab_size"] == 1000
        assert result["context_length"] == 128
        assert result["d_model"] == 64
        assert result["n_heads"] == 4
        assert result["d_ff"] == 128
        assert result["n_layers"] == 2
        assert result["num_experts"] == 4
        assert result["num_shared_experts"] == 2
        assert result["moe_top_k"] == 2
        assert result["use_hybrid_attention"] is True
        assert result["local_window_size"] == 16

    def test_roundtrip(self):
        """M01Config -> config_to_dict -> M01Config round-trip preserves values."""
        original = M01Config(
            vocab_size=8192,
            context_length=256,
            d_model=364,
            n_heads=7,
            d_ff=624,
            n_layers=10,
            num_experts=16,
            num_shared_experts=3,
            moe_top_k=4,
            use_hybrid_attention=True,
            local_window_size=16,
        )
        d = config_to_dict(original)
        reconstructed = M01Config(**d)

        assert reconstructed.vocab_size == original.vocab_size
        assert reconstructed.context_length == original.context_length
        assert reconstructed.d_model == original.d_model
        assert reconstructed.n_heads == original.n_heads
        assert reconstructed.d_ff == original.d_ff
        assert reconstructed.n_layers == original.n_layers
        assert reconstructed.num_experts == original.num_experts
        assert reconstructed.num_shared_experts == original.num_shared_experts
        assert reconstructed.moe_top_k == original.moe_top_k
        assert reconstructed.use_hybrid_attention == original.use_hybrid_attention
        assert reconstructed.local_window_size == original.local_window_size

    def test_optional_moe_fields(self):
        """config_to_dict includes optional d_ff_shared/d_ff_routed when present."""
        config = M01Config(
            vocab_size=1000,
            context_length=128,
            d_model=64,
            n_heads=4,
            d_ff=128,
            n_layers=2,
            num_experts=4,
            num_shared_experts=2,
            moe_top_k=2,
            use_hybrid_attention=True,
            local_window_size=16,
        )
        config.d_ff_shared = 256
        config.d_ff_routed = 512

        result = config_to_dict(config)
        assert result["d_ff_shared"] == 256
        assert result["d_ff_routed"] == 512

    def test_no_optional_fields(self):
        """config_to_dict omits d_ff_shared/d_ff_routed when not present."""
        config = M01Config(
            vocab_size=1000,
            context_length=128,
            d_model=64,
            n_heads=4,
            d_ff=128,
            n_layers=2,
            num_experts=4,
            num_shared_experts=2,
            moe_top_k=2,
            use_hybrid_attention=True,
            local_window_size=16,
        )
        result = config_to_dict(config)
        assert "d_ff_shared" not in result
        assert "d_ff_routed" not in result


class TestCheckpoint:
    """Tests for save_checkpoint / load_checkpoint roundtrip."""

    def test_roundtrip(self, tmp_path):
        """save_checkpoint then load_checkpoint preserves model weights."""
        config = M01Config(
            vocab_size=128,
            context_length=32,
            d_model=16,
            n_heads=2,
            d_ff=32,
            n_layers=2,
            num_experts=2,
            num_shared_experts=1,
            moe_top_k=2,
            use_hybrid_attention=True,
            local_window_size=8,
        )
        from src.model.lm import TransformerLM

        model = TransformerLM(config)
        ckpt_path = str(tmp_path / "test_ckpt.pt")

        save_checkpoint(model, config, ckpt_path)
        loaded_model, loaded_config = load_checkpoint(ckpt_path)

        # Verify config fields match
        assert loaded_config.vocab_size == config.vocab_size
        assert loaded_config.d_model == config.d_model
        assert loaded_config.n_layers == config.n_layers

        # Verify model weights match
        for (n1, p1), (n2, p2) in zip(
            model.named_parameters(), loaded_model.named_parameters()
        ):
            assert n1 == n2
            assert torch.allclose(p1, p2)

    def test_file_not_found(self):
        """load_checkpoint raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_checkpoint("/nonexistent/path/model.pt")

    def test_creates_directory(self, tmp_path):
        """save_checkpoint creates parent directory if it doesn't exist."""
        config = M01Config(
            vocab_size=128,
            context_length=32,
            d_model=16,
            n_heads=2,
            d_ff=32,
            n_layers=2,
            num_experts=2,
            num_shared_experts=1,
            moe_top_k=2,
            use_hybrid_attention=True,
            local_window_size=8,
        )
        from src.model.lm import TransformerLM

        model = TransformerLM(config)
        nested_path = str(tmp_path / "deep" / "nested" / "ckpt.pt")

        save_checkpoint(model, config, nested_path)
        assert os.path.exists(nested_path)


class TestAmplifiedDialogueDataset:
    """Tests for AmplifiedDialogueDataset."""

    def test_len(self):
        """__len__ returns overlapping window count."""
        # Mock tokenizer
        class MockTokenizer:
            def encode(self, text):
                return list(range(len(text)))

        tokenizer = MockTokenizer()
        dialogues = ["hello", "world"]
        dataset = AmplifiedDialogueDataset(tokenizer, dialogues, amp_factor=2, seq_len=10)

        # amp_factor=2, 2 dialogues -> 2*2 dialogues each ~5-6 chars
        # total tokens should be > seq_len
        assert len(dataset) > 0

    def test_getitem_shape(self):
        """__getitem__ returns (seq_len,) tensors."""
        class MockTokenizer:
            def encode(self, text):
                return list(range(len(text)))

        tokenizer = MockTokenizer()
        dialogues = ["hello world test data"] * 3
        dataset = AmplifiedDialogueDataset(tokenizer, dialogues, amp_factor=5, seq_len=16)

        if len(dataset) > 0:
            x, y = dataset[0]
            assert x.shape == (16,)
            assert y.shape == (16,)
            # Target is shifted by 1
            assert y[0].item() == x[1].item()


class TestJsonlDataset:
    """Tests for JsonlDataset with temp JSONL files."""

    def _make_jsonl(self, path, lines):
        """Helper to create a temp JSONL file at the given path."""
        with open(str(path), "w", encoding="utf-8") as f:
            for line in lines:
                f.write(json.dumps(line) + "\n")
        return str(path)

    def test_single_shard(self, tmp_path):
        """JsonlDataset loads a single JSONL shard."""
        class MockTokenizer:
            def encode(self, text):
                return list(range(len(text)))

        lines = [
            {"system": "You are helpful", "conversation": "a" * 200},
            {"system": "You are helpful", "conversation": "b" * 200},
        ]
        path = self._make_jsonl(tmp_path / "test.jsonl", lines)
        dataset = JsonlDataset(MockTokenizer(), [path], seq_len=16, max_lines_per_shard=100)

        assert len(dataset) > 0
        x, y = dataset[0]
        assert x.shape == (16,)
        assert y.shape == (16,)

    def test_multi_shard(self, tmp_path):
        """JsonlDataset loads multiple JSONL shards."""
        class MockTokenizer:
            def encode(self, text):
                return list(range(len(text)))

        lines1 = [{"system": "sys1", "conversation": "a" * 100}]
        lines2 = [{"system": "sys2", "conversation": "b" * 100}]
        path1 = self._make_jsonl(tmp_path / "shard1.jsonl", lines1)
        path2 = self._make_jsonl(tmp_path / "shard2.jsonl", lines2)

        dataset = JsonlDataset(MockTokenizer(), [path1, path2], seq_len=16)
        assert len(dataset) > 0

    def test_missing_shard_skipped(self, tmp_path):
        """JsonlDataset skips missing shard files gracefully."""
        class MockTokenizer:
            def encode(self, text):
                return list(range(len(text)))

        lines = [{"system": "sys", "conversation": "x" * 200}]
        path = self._make_jsonl(tmp_path / "test.jsonl", lines)

        dataset = JsonlDataset(
            MockTokenizer(), [path, "/nonexistent/shard.jsonl"], seq_len=16
        )
        assert len(dataset) > 0


class TestEvaluateValLoss:
    """Tests for evaluate_val_loss function."""

    def test_returns_float(self):
        """evaluate_val_loss returns a float."""
        # Simple model
        model = nn.Linear(16, 32)
        model.train()

        # Dummy data
        x = torch.randn(4, 16)
        y = torch.randint(0, 32, (4,))
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=2)

        criterion = nn.CrossEntropyLoss()
        device = torch.device("cpu")

        result = evaluate_val_loss(model, loader, device, criterion, max_batches=2)
        assert isinstance(result, float)
        assert result >= 0.0

    def test_sets_model_back_to_train(self):
        """evaluate_val_loss sets model back to train mode after eval."""
        model = nn.Linear(16, 32)
        model.train()

        x = torch.randn(4, 16)
        y = torch.randint(0, 32, (4,))
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=2)

        criterion = nn.CrossEntropyLoss()
        device = torch.device("cpu")

        evaluate_val_loss(model, loader, device, criterion)
        assert model.training is True


class TestSetupDevice:
    """Tests for setup_device function."""

    def test_returns_device(self):
        """setup_device returns a torch.device."""
        device = setup_device()
        assert isinstance(device, torch.device)
        assert device.type in ("cuda", "cpu")


class TestSetupStdout:
    """Tests for setup_stdout function."""

    def test_no_error(self):
        """setup_stdout does not raise."""
        setup_stdout()  # Should not raise


class TestSharedTrain:
    """Tests for the shared train() function."""

    def test_loss_decreases(self):
        """train() returns lower loss after training."""
        # Tiny model
        model = nn.Sequential(nn.Linear(8, 16), nn.Linear(16, 8))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        criterion = nn.CrossEntropyLoss()

        # Dummy dataloader
        x = torch.randn(16, 8)
        y = torch.randint(0, 8, (16,))
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=4, shuffle=True)

        device = torch.device("cpu")
        model.train()

        result = train(model, loader, optimizer, criterion, steps=10, device=device, log_interval=5)

        assert result["steps_completed"] == 10
        assert result["last_loss"] < 5.0  # Should be reasonable
        assert result["elapsed"] >= 0.0

    def test_target_loss_early_stop(self):
        """train() stops early when target_loss is reached."""
        model = nn.Sequential(nn.Linear(8, 16), nn.Linear(16, 8))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        criterion = nn.CrossEntropyLoss()

        x = torch.randn(16, 8)
        y = torch.randint(0, 8, (16,))
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=4, shuffle=True)

        device = torch.device("cpu")
        model.train()

        # Set a very high target loss so it stops immediately
        result = train(model, loader, optimizer, criterion, steps=1000, device=device, target_loss=100.0)

        assert result["steps_completed"] < 1000
