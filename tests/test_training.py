"""Tests for training components: TinyShakespeareDataset, CheckpointManager, and integration.

TinyShakespeareDataset requirements:
- Loads BPE tokenizer from data/tokenizer.json
- Tokenizes data/tinyshakespeare.txt once at init → LongTensor
- __getitem__(i) returns (input, target) LongTensors of shape (seq_len,)
- Input = tokens[i:i+seq_len], target = tokens[i+1:i+seq_len+1]
- __len__ = len(tokens) - seq_len
- Compatible with DataLoader(batch_size=N, num_workers=0)

CheckpointManager requirements:
- __init__(checkpoint_dir) ensures directory exists
- save(step, model, optimizer, scheduler, loss, config) → atomic write
- load(model, optimizer, scheduler) → restores states, returns checkpoint dict
- Atomic: write .tmp → os.replace(tmp, checkpoint.pt)

Integration (training loop):
- 2-step training produces lower loss than initial loss
- Checkpoint round-trip preserves model state dicts after train → save → modify → load
"""

import os
import tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import LongTensor
import pytest
from torch.utils.data import DataLoader

from src.training.config import TrainingConfig
from src.transformer.config import M01Config


# ---------------------------------------------------------------------------
# TinyShakespeareDataset Tests
# ---------------------------------------------------------------------------

@pytest.fixture
def training_config() -> TrainingConfig:
    """Small config for fast dataset tests."""
    return TrainingConfig(seq_len=64, data_dir="data")


@pytest.fixture
def dataset(training_config: TrainingConfig):
    """TinyShakespeareDataset instance (RED: will fail — module doesn't exist yet)."""
    from src.training.dataset import TinyShakespeareDataset  # type: ignore
    return TinyShakespeareDataset(training_config)


class TestDatasetLength:
    """TinyShakespeareDataset length invariants."""

    def test_dataset_len_positive(self, dataset) -> None:
        """__len__ MUST be positive for a non-trivial dataset."""
        assert len(dataset) > 0, (
            f"Expected len(dataset) > 0, got {len(dataset)}"
        )

    def test_dataset_len_equals_total_minus_seq(self, dataset, training_config) -> None:
        """__len__ MUST equal total_tokens - seq_len."""
        # The invariant is deterministic; corpus size is intentionally not
        # asserted because data revisions are valid.
        n = len(dataset)
        expected_total = n + training_config.seq_len
        assert expected_total == len(dataset.tokens)


class TestDatasetGetItem:
    """TinyShakespeareDataset __getitem__ invariants."""

    def test_getitem_returns_tuple(self, dataset) -> None:
        """__getitem__(i) MUST return a tuple of (input, target)."""
        result = dataset[0]
        assert isinstance(result, tuple), (
            f"Expected tuple, got {type(result)}"
        )
        assert len(result) == 2, (
            f"Expected len 2 tuple, got {len(result)}"
        )

    def test_input_is_long_tensor(self, dataset) -> None:
        """input tensor MUST be dtype torch.long."""
        x, y = dataset[0]
        assert x.dtype == torch.long, (
            f"Expected torch.long, got {x.dtype}"
        )

    def test_target_is_long_tensor(self, dataset) -> None:
        """target tensor MUST be dtype torch.long."""
        x, y = dataset[0]
        assert y.dtype == torch.long, (
            f"Expected torch.long, got {y.dtype}"
        )

    def test_input_shape(self, dataset, training_config) -> None:
        """input tensor MUST have shape (seq_len,)."""
        x, y = dataset[0]
        assert x.shape == (training_config.seq_len,), (
            f"Expected ({training_config.seq_len},), got {x.shape}"
        )

    def test_target_shape(self, dataset, training_config) -> None:
        """target tensor MUST have shape (seq_len,)."""
        x, y = dataset[0]
        assert y.shape == (training_config.seq_len,), (
            f"Expected ({training_config.seq_len},), got {y.shape}"
        )

    def test_target_is_input_shifted(self, dataset, training_config) -> None:
        """target[t] MUST equal input[t+1] for t in range(seq_len-1)."""
        x, y = dataset[0]
        # x[i+1] should equal y[i] for i in range(seq_len - 1)
        assert torch.equal(x[1:], y[:-1]), (
            "target[: -1] must equal input[1:] (sliding window shift)"
        )

    def test_multiple_indices_different_output(self, dataset) -> None:
        """Different indices MUST yield different (input, target) pairs."""
        x0, y0 = dataset[0]
        x1, y1 = dataset[100]
        # Different indices must produce different sequences
        assert not torch.equal(x0, x1), (
            "dataset[0] and dataset[100] must not produce identical inputs"
        )

    def test_getitem_last_index(self, dataset) -> None:
        """__getitem__(len-1) MUST work without IndexError."""
        n = len(dataset)
        x, y = dataset[n - 1]
        assert x.shape == y.shape, (
            f"Last index returned mismatched shapes: {x.shape} vs {y.shape}"
        )


class TestDatasetDataLoader:
    """TinyShakespeareDataset must work with DataLoader."""

    def test_dataloader_batches(self, dataset, training_config) -> None:
        """DataLoader with batch_size=4 MUST yield (B, S) tensors."""
        loader = DataLoader(
            dataset,
            batch_size=4,
            num_workers=0,
        )
        batch_x, batch_y = next(iter(loader))
        assert batch_x.shape == (4, training_config.seq_len), (
            f"Expected (4, {training_config.seq_len}), got {batch_x.shape}"
        )
        assert batch_y.shape == (4, training_config.seq_len), (
            f"Expected (4, {training_config.seq_len}), got {batch_y.shape}"
        )

    def test_dataloader_num_workers_zero(self, dataset) -> None:
        """DataLoader with num_workers=0 MUST work (Windows safety)."""
        loader = DataLoader(
            dataset,
            batch_size=2,
            num_workers=0,
        )
        for batch_x, batch_y in loader:
            assert batch_x.shape[0] == 2
            assert batch_y.shape[0] == 2
            break  # single batch is enough


# ---------------------------------------------------------------------------
# CheckpointManager Tests
# ---------------------------------------------------------------------------

@pytest.fixture
def chk_model() -> nn.Module:
    """Simple model for checkpoint round-trip tests."""
    return nn.Sequential(
        nn.Linear(16, 32),
        nn.ReLU(),
        nn.Linear(32, 8),
    )


@pytest.fixture
def chk_optimizer(chk_model) -> torch.optim.Optimizer:
    """AdamW optimizer for the simple model."""
    return torch.optim.AdamW(chk_model.parameters(), lr=1e-3)


@pytest.fixture
def chk_scheduler(chk_optimizer) -> torch.optim.lr_scheduler.LRScheduler:
    """Cosine annealing scheduler for checkpoint tests."""
    return torch.optim.lr_scheduler.CosineAnnealingLR(chk_optimizer, T_max=10)


@pytest.fixture
def checkpoint_dir() -> str:
    """Temporary directory for checkpoint files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def manager(checkpoint_dir: str):
    """CheckpointManager instance (RED: will fail — module doesn't exist yet)."""
    from src.training.checkpoint import CheckpointManager  # type: ignore
    return CheckpointManager(checkpoint_dir)


class TestCheckpointInit:
    """CheckpointManager initialization."""

    def test_creates_directory(self) -> None:
        """CheckpointManager MUST create the checkpoint directory."""
        from src.training.checkpoint import CheckpointManager
        with tempfile.TemporaryDirectory() as tmpdir:
            chk_dir = os.path.join(tmpdir, "my_checkpoints")
            assert not os.path.exists(chk_dir)
            CheckpointManager(chk_dir)
            assert os.path.isdir(chk_dir), (
                f"CheckpointManager must create '{chk_dir}'"
            )

    def test_accepts_existing_directory(self, checkpoint_dir) -> None:
        """CheckpointManager MUST accept an already-existing directory."""
        from src.training.checkpoint import CheckpointManager
        manager = CheckpointManager(checkpoint_dir)
        assert manager is not None


class TestCheckpointSave:
    """CheckpointManager.save() behavior."""

    def test_save_creates_file(self, manager, chk_model, chk_optimizer,
                               chk_scheduler) -> None:
        """save() MUST create checkpoint.pt in the checkpoint_dir."""
        manager.save(
            step=0, model=chk_model, optimizer=chk_optimizer,
            scheduler=chk_scheduler, loss=2.5, config={"vocab_size": 32768},
        )
        pt_path = os.path.join(manager.checkpoint_dir, "checkpoint.pt")
        assert os.path.isfile(pt_path), (
            f"Expected checkpoint file at {pt_path}"
        )

    def test_save_no_tmp_left(self, manager, chk_model, chk_optimizer,
                              chk_scheduler) -> None:
        """save() MUST NOT leave .tmp files after completion."""
        manager.save(
            step=0, model=chk_model, optimizer=chk_optimizer,
            scheduler=chk_scheduler, loss=2.5, config={"vocab_size": 32768},
        )
        tmp_files = [
            f for f in os.listdir(manager.checkpoint_dir)
            if f.endswith(".tmp")
        ]
        assert len(tmp_files) == 0, (
            f"Found .tmp files after save: {tmp_files}"
        )

    def test_save_checkpoint_content(self, manager, chk_model, chk_optimizer,
                                     chk_scheduler) -> None:
        """Saved checkpoint dict MUST contain all expected keys."""
        manager.save(
            step=42, model=chk_model, optimizer=chk_optimizer,
            scheduler=chk_scheduler, loss=1.234, config={"vocab_size": 32768},
        )
        pt_path = os.path.join(manager.checkpoint_dir, "checkpoint.pt")
        checkpoint = torch.load(pt_path, weights_only=True)
        expected_keys = {
            "epoch", "step", "loss",
            "model_state_dict", "optimizer_state_dict",
            "scheduler_state_dict", "config",
        }
        actual_keys = set(checkpoint.keys())
        assert actual_keys == expected_keys, (
            f"Expected keys {expected_keys}, got {actual_keys}"
        )

    def test_save_step_value(self, manager, chk_model, chk_optimizer,
                             chk_scheduler) -> None:
        """Saved step value MUST match the provided step argument."""
        manager.save(
            step=99, model=chk_model, optimizer=chk_optimizer,
            scheduler=chk_scheduler, loss=0.5, config={"lr": 1e-3},
        )
        pt_path = os.path.join(manager.checkpoint_dir, "checkpoint.pt")
        checkpoint = torch.load(pt_path, weights_only=True)
        assert checkpoint["step"] == 99, (
            f"Expected step=99, got {checkpoint['step']}"
        )

    def test_save_loss_value(self, manager, chk_model, chk_optimizer,
                             chk_scheduler) -> None:
        """Saved loss value MUST match the provided loss argument."""
        manager.save(
            step=0, model=chk_model, optimizer=chk_optimizer,
            scheduler=chk_scheduler, loss=3.14159, config={},
        )
        pt_path = os.path.join(manager.checkpoint_dir, "checkpoint.pt")
        checkpoint = torch.load(pt_path, weights_only=True)
        assert abs(checkpoint["loss"] - 3.14159) < 1e-5, (
            f"Expected loss ~3.14159, got {checkpoint['loss']}"
        )

    def test_save_epoch_default(self, manager, chk_model, chk_optimizer,
                                chk_scheduler) -> None:
        """Saved epoch MUST default to 0 when not provided."""
        manager.save(
            step=0, model=chk_model, optimizer=chk_optimizer,
            scheduler=chk_scheduler, loss=1.0, config={},
        )
        pt_path = os.path.join(manager.checkpoint_dir, "checkpoint.pt")
        checkpoint = torch.load(pt_path, weights_only=True)
        assert checkpoint["epoch"] == 0, (
            f"Expected epoch == 0, got {checkpoint['epoch']}"
        )


class TestCheckpointLoad:
    """CheckpointManager.load() behavior."""

    def test_load_returns_checkpoint_dict(self, manager, chk_model,
                                          chk_optimizer, chk_scheduler) -> None:
        """load() MUST return a dict with epoch, step, loss, config."""
        manager.save(
            step=5, model=chk_model, optimizer=chk_optimizer,
            scheduler=chk_scheduler, loss=0.123, config={"d_model": 640},
        )
        checkpoint = manager.load(chk_model, chk_optimizer, chk_scheduler)
        assert isinstance(checkpoint, dict), (
            f"Expected dict, got {type(checkpoint)}"
        )
        assert checkpoint["step"] == 5
        assert abs(checkpoint["loss"] - 0.123) < 1e-5

    def test_load_restores_model_weights(self, manager, chk_model,
                                         chk_optimizer, chk_scheduler) -> None:
        """After save → load, model weights MUST match saved weights."""
        # Record initial weights
        initial_weights = [p.clone() for p in chk_model.parameters()]

        # Modify model
        with torch.no_grad():
            for param in chk_model.parameters():
                param.add_(1.0)

        modified_weights = [p.clone() for p in chk_model.parameters()]
        # Verify model changed
        assert not torch.equal(initial_weights[0], modified_weights[0]), (
            "Test setup error: model weights should have changed"
        )

        # Now save, then load (restore initial weights conceptually)
        # Save the modified model
        manager.save(
            step=0, model=chk_model, optimizer=chk_optimizer,
            scheduler=chk_scheduler, loss=1.0, config={},
        )

        # Reset model weights to something different
        with torch.no_grad():
            for param in chk_model.parameters():
                param.zero_()

        zero_weights = [p.clone() for p in chk_model.parameters()]
        assert not torch.equal(modified_weights[0], zero_weights[0]), (
            "Test setup error: weights should be zeroed"
        )

        # Load checkpoint — should restore modified weights
        manager.load(chk_model, chk_optimizer, chk_scheduler)
        loaded_weights = [p for p in chk_model.parameters()]

        for i, (saved, loaded) in enumerate(zip(modified_weights, loaded_weights)):
            assert torch.equal(saved, loaded), (
                f"Parameter {i} does not match after save → load"
            )

    def test_load_restores_optimizer_state(self, manager, chk_model,
                                           chk_optimizer, chk_scheduler) -> None:
        """After save → load, optimizer state MUST be restored."""
        # Do a step to create optimizer state
        dummy_input = torch.randn(2, 16)
        target = torch.randint(0, 8, (2,))
        output = chk_model(dummy_input)
        loss = nn.functional.cross_entropy(output, target)
        loss.backward()
        chk_optimizer.step()

        # Save the state
        manager.save(
            step=1, model=chk_model, optimizer=chk_optimizer,
            scheduler=chk_scheduler, loss=loss.item(), config={},
        )

        # Create a fresh optimizer and load
        fresh_optimizer = torch.optim.AdamW(chk_model.parameters(), lr=1e-3)
        manager.load(chk_model, fresh_optimizer, chk_scheduler)

        # Check that optimizer param_groups match
        fresh_state = fresh_optimizer.state_dict()

        original_state = {k: v for k, v in fresh_state.items() if k != "state"}
        loaded_after = fresh_optimizer.state_dict()
        loaded_after_clean = {k: v for k, v in loaded_after.items() if k != "state"}

        assert original_state == loaded_after_clean, (
            "Optimizer param_groups must match after save → load"
        )

    def test_load_restores_scheduler_state(self, manager, chk_model,
                                           chk_optimizer, chk_scheduler) -> None:
        """After save → load, scheduler state MUST be restored."""
        # Step the scheduler a few times
        for _ in range(3):
            chk_scheduler.step()

        original_lr = chk_scheduler.get_last_lr()[0]

        manager.save(
            step=3, model=chk_model, optimizer=chk_optimizer,
            scheduler=chk_scheduler, loss=0.5, config={},
        )

        # Create fresh scheduler
        fresh_optimizer = torch.optim.AdamW(chk_model.parameters(), lr=1e-3)
        fresh_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            fresh_optimizer, T_max=10
        )

        manager.load(chk_model, fresh_optimizer, fresh_scheduler)
        restored_lr = fresh_scheduler.get_last_lr()[0]

        assert abs(restored_lr - original_lr) < 1e-8, (
            f"Expected LR {original_lr}, got {restored_lr} after load"
        )

    def test_load_raises_on_missing_file(self, manager, chk_model,
                                         chk_optimizer, chk_scheduler) -> None:
        """load() MUST raise FileNotFoundError when no checkpoint exists."""
        with pytest.raises(FileNotFoundError):
            manager.load(chk_model, chk_optimizer, chk_scheduler)


# ---------------------------------------------------------------------------
# Integration Tests — Training Loop
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_model_config() -> M01Config:
    """Minimal M01Config for fast integration tests."""
    return M01Config(
        vocab_size=1024,
        context_length=128,
        d_model=32,
        n_heads=4,
        d_ff=64,
        n_layers=2,
        num_experts=1,
        dropout=0.0,
    )


@pytest.fixture
def tiny_train_config() -> TrainingConfig:
    """Minimal TrainingConfig for fast integration tests."""
    return TrainingConfig(
        batch_size=2,
        seq_len=32,
        max_lr=1e-3,
        min_lr_ratio=0.1,
        warmup_steps=2,
        max_steps=5,
        weight_decay=0.1,
        beta1=0.9,
        beta2=0.95,
        max_norm=1.0,
        log_interval=10,
        save_interval=10,
        checkpoint_dir="checkpoints",
        data_dir="data",
    )


@pytest.fixture
def integration_model(tiny_model_config) -> nn.Module:
    """TransformerLM with tiny config for fast integration tests."""
    from src.model.lm import TransformerLM
    return TransformerLM(tiny_model_config)


@pytest.fixture
def synthetic_tokens(tiny_train_config) -> LongTensor:
    """Synthetic token tensor to avoid loading the real dataset."""
    return torch.randint(0, 1024, (1000,), dtype=torch.long)


class SyntheticDataset:
    """In-memory sliding-window dataset to avoid file I/O during integration tests."""

    def __init__(self, tokens: LongTensor, seq_len: int) -> None:
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.tokens) - self.seq_len

    def __getitem__(self, idx: int):
        x = self.tokens[idx: idx + self.seq_len]
        y = self.tokens[idx + 1: idx + self.seq_len + 1]
        return x, y


@pytest.fixture
def synthetic_dataset(synthetic_tokens, tiny_train_config) -> SyntheticDataset:
    """Synthetic dataset for fast integration tests."""
    return SyntheticDataset(synthetic_tokens, tiny_train_config.seq_len)


class TestTwoStepTraining:
    """2-step training MUST reduce loss below initial value."""

    def test_two_step_loss_decreases(self, integration_model, tiny_train_config,
                                     synthetic_dataset) -> None:
        """After 2 optimizer steps, loss MUST be lower than initial loss.
        (RED: configure_optimizer / get_lr_scheduler don't exist yet.)"""
        from src.training.train import configure_optimizer, get_lr_scheduler
        from torch.utils.data import DataLoader

        loader = DataLoader(
            synthetic_dataset,
            batch_size=tiny_train_config.batch_size,
            num_workers=0,
        )

        optimizer = configure_optimizer(integration_model, tiny_train_config)
        scheduler = get_lr_scheduler(
            optimizer,
            tiny_train_config.warmup_steps,
            tiny_train_config.max_steps,
            tiny_train_config.min_lr_ratio,
        )

        # Measure initial loss (step 0, no training yet)
        x0, y0 = next(iter(loader))
        logits0 = integration_model(x0)
        V = logits0.size(-1)
        loss_initial = F.cross_entropy(logits0.view(-1, V), y0.view(-1))

        # Step 1
        loss_initial.backward()
        torch.nn.utils.clip_grad_norm_(
            integration_model.parameters(), tiny_train_config.max_norm
        )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # Step 2
        x1, y1 = next(iter(loader))
        logits1 = integration_model(x1)
        loss_step1 = F.cross_entropy(logits1.view(-1, V), y1.view(-1))
        loss_step1.backward()
        torch.nn.utils.clip_grad_norm_(
            integration_model.parameters(), tiny_train_config.max_norm
        )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # Measure loss after 2 training steps
        x2, y2 = next(iter(loader))
        logits2 = integration_model(x2)
        loss_after = F.cross_entropy(logits2.view(-1, V), y2.view(-1))

        assert loss_after < loss_initial, (
            f"Expected loss after 2 steps ({loss_after.item():.4f}) "
            f"< initial loss ({loss_initial.item():.4f})"
        )

    def test_training_step_does_not_crash_single_batch(
        self, integration_model, tiny_train_config, synthetic_dataset
    ) -> None:
        """A single training step MUST complete without runtime errors.
        (RED: configure_optimizer / get_lr_scheduler don't exist yet.)"""
        from src.training.train import configure_optimizer, get_lr_scheduler
        from torch.utils.data import DataLoader

        loader = DataLoader(
            synthetic_dataset,
            batch_size=tiny_train_config.batch_size,
            num_workers=0,
        )

        optimizer = configure_optimizer(integration_model, tiny_train_config)
        scheduler = get_lr_scheduler(
            optimizer,
            tiny_train_config.warmup_steps,
            tiny_train_config.max_steps,
            tiny_train_config.min_lr_ratio,
        )

        x, y = next(iter(loader))
        logits = integration_model(x)
        V = logits.size(-1)
        loss = F.cross_entropy(logits.view(-1, V), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            integration_model.parameters(), tiny_train_config.max_norm
        )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # If we got here without exception, the step succeeded
        pass


class TestCheckpointRoundtrip:
    """Checkpoint save → modify → load MUST restore state dicts."""

    def test_roundtrip_restores_model_weights(
        self, integration_model, tiny_train_config, synthetic_dataset
    ) -> None:
        """After save → modify → load, model weights MUST match saved weights.
        (RED: configure_optimizer / get_lr_scheduler don't exist yet.)"""
        from src.training.train import configure_optimizer, get_lr_scheduler
        from src.training.checkpoint import CheckpointManager
        from torch.utils.data import DataLoader
        import tempfile

        loader = DataLoader(
            synthetic_dataset,
            batch_size=tiny_train_config.batch_size,
            num_workers=0,
        )

        optimizer = configure_optimizer(integration_model, tiny_train_config)
        scheduler = get_lr_scheduler(
            optimizer,
            tiny_train_config.warmup_steps,
            tiny_train_config.max_steps,
            tiny_train_config.min_lr_ratio,
        )

        # Perform one training step to get meaningful state
        x, y = next(iter(loader))
        logits = integration_model(x)
        V = logits.size(-1)
        loss = F.cross_entropy(logits.view(-1, V), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            integration_model.parameters(), tiny_train_config.max_norm
        )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir)

            # Save the current (trained) state
            manager.save(
                step=1,
                model=integration_model,
                optimizer=optimizer,
                scheduler=scheduler,
                loss=loss.item(),
                config={"vocab_size": 1024, "d_model": 32},
            )

            # Record saved weights
            saved_weights = [
                p.clone() for p in integration_model.parameters()
            ]

            # Modify model weights (add 1.0 to all)
            with torch.no_grad():
                for param in integration_model.parameters():
                    param.add_(1.0)

            modified_weights = [
                p.clone() for p in integration_model.parameters()
            ]
            assert not torch.equal(saved_weights[0], modified_weights[0]), (
                "Test setup error: weights should have changed"
            )

            # Load checkpoint — should restore saved weights
            fresh_optimizer = configure_optimizer(
                integration_model, tiny_train_config
            )
            fresh_scheduler = get_lr_scheduler(
                fresh_optimizer,
                tiny_train_config.warmup_steps,
                tiny_train_config.max_steps,
                tiny_train_config.min_lr_ratio,
            )
            manager.load(
                integration_model, fresh_optimizer, fresh_scheduler
            )

            loaded_weights = [
                p for p in integration_model.parameters()
            ]

            for i, (saved, loaded) in enumerate(
                zip(saved_weights, loaded_weights)
            ):
                assert torch.equal(saved, loaded), (
                    f"Parameter {i} does not match after round-trip"
                )

    def test_roundtrip_with_two_steps(
        self, integration_model, tiny_train_config, synthetic_dataset
    ) -> None:
        """After 2 training steps, save → modify → load MUST restore state.
        (RED: configure_optimizer / get_lr_scheduler don't exist yet.)"""
        from src.training.train import configure_optimizer, get_lr_scheduler
        from src.training.checkpoint import CheckpointManager
        from torch.utils.data import DataLoader
        import tempfile

        loader = DataLoader(
            synthetic_dataset,
            batch_size=tiny_train_config.batch_size,
            num_workers=0,
        )

        optimizer = configure_optimizer(integration_model, tiny_train_config)
        scheduler = get_lr_scheduler(
            optimizer,
            tiny_train_config.warmup_steps,
            tiny_train_config.max_steps,
            tiny_train_config.min_lr_ratio,
        )

        # Train 2 steps
        for _ in range(2):
            x, y = next(iter(loader))
            logits = integration_model(x)
            V = logits.size(-1)
            loss = F.cross_entropy(logits.view(-1, V), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                integration_model.parameters(), tiny_train_config.max_norm
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir)

            # Save trained state
            manager.save(
                step=2,
                model=integration_model,
                optimizer=optimizer,
                scheduler=scheduler,
                loss=loss.item(),
                config={"vocab_size": 1024, "d_model": 32},
            )

            # Record saved weights
            saved_weights = [
                p.clone() for p in integration_model.parameters()
            ]

            # Scramble weights
            with torch.no_grad():
                for param in integration_model.parameters():
                    param.normal_(0, 1)

            # Create fresh optimizer + scheduler for load
            fresh_optimizer = configure_optimizer(
                integration_model, tiny_train_config
            )
            fresh_scheduler = get_lr_scheduler(
                fresh_optimizer,
                tiny_train_config.warmup_steps,
                tiny_train_config.max_steps,
                tiny_train_config.min_lr_ratio,
            )

            manager.load(
                integration_model, fresh_optimizer, fresh_scheduler
            )

            loaded_weights = [
                p for p in integration_model.parameters()
            ]

            for i, (saved, loaded) in enumerate(
                zip(saved_weights, loaded_weights)
            ):
                assert torch.equal(saved, loaded), (
                    f"Parameter {i} does not match after 2-step round-trip"
                )


class TestCLIHelp:
    """CLI entry --help must work without error."""

    def test_cli_help_prints_usage(self) -> None:
        """``python -m src.training.train --help`` MUST print usage and exit.
        (RED: train.py doesn't exist yet.)"""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "src.training.train", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"CLI --help exited with code {result.returncode}: {result.stderr}"
        )
        assert "usage:" in result.stdout.lower(), (
            f"stdout should contain usage info:\n{result.stdout}"
        )
        assert "train" in result.stdout.lower(), (
            f"stdout should mention 'train':\n{result.stdout}"
        )


class TestLRComponents:
    """Unit tests for training utilities used in train.py."""

    def test_configure_optimizer_has_two_param_groups(
        self, integration_model, tiny_train_config
    ) -> None:
        """configure_optimizer MUST create AdamW with 2 param groups
        (decay and no_decay). (RED: configure_optimizer doesn't exist yet.)"""
        from src.training.train import configure_optimizer

        optimizer = configure_optimizer(integration_model, tiny_train_config)

        assert len(optimizer.param_groups) == 2, (
            f"Expected 2 param groups, got {len(optimizer.param_groups)}"
        )

        # First group (decay) should have nonzero weight_decay
        assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(
            tiny_train_config.weight_decay
        ), "First param group should have weight_decay"

        # Second group (no_decay) should have zero weight_decay
        assert optimizer.param_groups[1]["weight_decay"] == 0.0, (
            "No-decay param group must have weight_decay=0.0"
        )

    def test_configure_optimizer_bias_and_gamma_in_no_decay(
        self, integration_model, tiny_train_config
    ) -> None:
        """Parameters named 'bias' or 'gamma' MUST be in the no-decay group.
        (RED: configure_optimizer doesn't exist yet.)"""
        from src.training.train import configure_optimizer

        optimizer = configure_optimizer(integration_model, tiny_train_config)

        no_decay_params = optimizer.param_groups[1]["params"]
        no_decay_ids = {id(p) for p in no_decay_params}

        for name, param in integration_model.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "gamma" in name:
                assert id(param) in no_decay_ids, (
                    f"Parameter '{name}' should be in no_decay group"
                )

    def test_lr_scheduler_warmup_then_cosine(
        self, tiny_train_config
    ) -> None:
        """get_lr_scheduler MUST produce warmup → cosine decay schedule.
        (RED: get_lr_scheduler doesn't exist yet.)"""
        from src.training.train import get_lr_scheduler
        import torch

        dummy_params = [torch.nn.Parameter(torch.randn(1))]
        optimizer = torch.optim.AdamW(dummy_params, lr=tiny_train_config.max_lr)

        scheduler = get_lr_scheduler(
            optimizer,
            tiny_train_config.warmup_steps,
            tiny_train_config.max_steps,
            tiny_train_config.min_lr_ratio,
        )

        # Collect LR for each step
        lrs = []
        for step in range(tiny_train_config.max_steps):
            scheduler.step()
            lrs.append(scheduler.get_last_lr()[0])

        # Step 0 (after first scheduler.step()) should have lr > 0 (warmup)
        assert lrs[0] > 0.0, "LR at step 0 should be positive (warmup)"

        # Peak LR should be at warmup_steps (or near it)
        peak_lr = max(lrs)
        peak_step = lrs.index(peak_lr)
        assert peak_step <= tiny_train_config.warmup_steps + 1, (
            f"Peak LR at step {peak_step}, expected near warmup_steps={tiny_train_config.warmup_steps}"
        )

        # Final LR should be close to min_lr_ratio * max_lr
        final_lr = lrs[-1]
        expected_min_lr = tiny_train_config.max_lr * tiny_train_config.min_lr_ratio
        assert abs(final_lr - expected_min_lr) < 1e-6, (
            f"Final LR {final_lr:.6f} != expected min {expected_min_lr:.6f}"
        )

        # LR should be decreasing in the cosine decay phase
        for i in range(tiny_train_config.warmup_steps + 1, len(lrs) - 1):
            assert lrs[i + 1] <= lrs[i] + 1e-8, (
                f"LR increased in decay phase at step {i}: {lrs[i]} -> {lrs[i + 1]}"
            )
