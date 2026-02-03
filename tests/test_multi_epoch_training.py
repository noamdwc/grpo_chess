"""Test that training runs for multiple epochs without stopping.

This test verifies that the self-play training loop correctly continues
past the first epoch. The bug being tested is that training stops after
1 epoch when using the ChessStartStatesDataset with multiple workers.
"""
import sys
import os
import json
from unittest.mock import MagicMock
print(f"os.getcwd(): {os.getcwd()}")
print(f"sys.path: {sys.path}")
print(f"ls: {os.listdir()}")
# Mock the missing searchless_chess imports before importing anything that uses them
with open('tests/resources/ACTION_TO_MOVE.json', 'r') as f:
    ACTION_TO_MOVE = json.load(f)
with open('tests/resources/MOVE_TO_ACTION.json', 'r') as f:
    MOVE_TO_ACTION = json.load(f)
mock_utils = MagicMock()
mock_utils.ACTION_TO_MOVE = ACTION_TO_MOVE
mock_utils.MOVE_TO_ACTION = MOVE_TO_ACTION

mock_tokenizer = MagicMock()
mock_tokenizer.tokenize = lambda x: [1, 2, 3]  # Dummy tokenization
mock_tokenizer.SEQUENCE_LENGTH = 128

# Create mock modules
sys.modules['src.searchless_chess_model'] = MagicMock()
sys.modules['src.searchless_chess_model.searchless_chess_code'] = MagicMock()
sys.modules['src.searchless_chess_model.searchless_chess_code.utils'] = mock_utils
sys.modules['src.searchless_chess_model.searchless_chess_code.tokenizer'] = mock_tokenizer

import pytest
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, IterableDataset

from src.grpo_self_play.chess.boards_dataset import ChessStartStatesDataset, ChessDatasetConfig


class EpochCounterCallback(pl.Callback):
    """Callback to track the number of completed epochs."""

    def __init__(self):
        self.epochs_completed = 0
        self.batches_per_epoch = []
        self.current_epoch_batches = 0

    def on_train_epoch_start(self, trainer, pl_module):
        self.current_epoch_batches = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.current_epoch_batches += 1

    def on_train_epoch_end(self, trainer, pl_module):
        self.epochs_completed += 1
        self.batches_per_epoch.append(self.current_epoch_batches)


class SimpleIterableDataset(IterableDataset):
    """Minimal iterable dataset that mimics ChessStartStatesDataset behavior."""

    def __init__(self, max_steps: int = 10):
        self.max_steps = max_steps

    def __iter__(self):
        for i in range(self.max_steps):
            yield f"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 {i+1}"


class DummyGRPOModel(pl.LightningModule):
    """Minimal model for testing the training loop without actual GRPO logic."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 1)
        self.epochs_seen = []

    def forward(self, x):
        return self.linear(x)

    def training_step(self, batch, batch_idx):
        # Track which epoch we're in
        if self.current_epoch not in self.epochs_seen:
            self.epochs_seen.append(self.current_epoch)
        # Return dummy loss
        return torch.tensor(0.1, requires_grad=True)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


class TestMultiEpochTraining:
    """Tests for verifying training continues past the first epoch."""

    def test_simple_iterable_dataset_multi_epoch(self):
        """Test that a simple iterable dataset allows multiple epochs."""
        dataset = SimpleIterableDataset(max_steps=8)
        dataloader = DataLoader(dataset, batch_size=2, num_workers=0)

        model = DummyGRPOModel()
        epoch_counter = EpochCounterCallback()

        trainer = pl.Trainer(
            max_epochs=3,
            accelerator="cpu",
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            callbacks=[epoch_counter],
        )

        trainer.fit(model, dataloader)

        assert epoch_counter.epochs_completed == 3, (
            f"Expected 3 epochs completed, but got {epoch_counter.epochs_completed}. "
            f"Batches per epoch: {epoch_counter.batches_per_epoch}"
        )
        assert len(model.epochs_seen) == 3, (
            f"Model only saw epochs: {model.epochs_seen}"
        )

    def test_chess_dataset_multi_epoch_single_worker(self):
        """Test ChessStartStatesDataset runs multiple epochs with single worker.

        This test uses the actual ChessStartStatesDataset but with quality_filter=False
        to avoid Stockfish dependency and make it fast.
        """
        config = ChessDatasetConfig(
            max_steps=8,  # Small for fast test
            quality_filter=False,  # Don't use Stockfish
        )
        dataset = ChessStartStatesDataset(config)
        dataloader = DataLoader(dataset, batch_size=2, num_workers=0)

        model = DummyGRPOModel()
        epoch_counter = EpochCounterCallback()

        trainer = pl.Trainer(
            max_epochs=3,
            accelerator="cpu",
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            callbacks=[epoch_counter],
        )

        trainer.fit(model, dataloader)

        assert epoch_counter.epochs_completed == 3, (
            f"Expected 3 epochs completed, but got {epoch_counter.epochs_completed}. "
            f"Batches per epoch: {epoch_counter.batches_per_epoch}"
        )

    def test_chess_dataset_multi_epoch_multi_worker(self):
        """Test ChessStartStatesDataset runs multiple epochs with multiple workers.

        This is the key test - the bug manifests when using num_workers > 0.
        The training should continue past epoch 1 when using multiple workers.
        """
        config = ChessDatasetConfig(
            max_steps=8,  # Small for fast test
            quality_filter=False,  # Don't use Stockfish
        )
        dataset = ChessStartStatesDataset(config)
        # Use num_workers=2 like in the notebook to reproduce the bug
        dataloader = DataLoader(dataset, batch_size=2, num_workers=2)

        model = DummyGRPOModel()
        epoch_counter = EpochCounterCallback()

        trainer = pl.Trainer(
            max_epochs=3,
            accelerator="cpu",
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            callbacks=[epoch_counter],
        )

        trainer.fit(model, dataloader)

        # This assertion should FAIL with the current bug
        # Training stops after 1 epoch instead of completing 3
        assert epoch_counter.epochs_completed == 3, (
            f"BUG DETECTED: Expected 3 epochs completed, but got {epoch_counter.epochs_completed}. "
            f"Training stopped after epoch {epoch_counter.epochs_completed}. "
            f"Batches per epoch: {epoch_counter.batches_per_epoch}"
        )

    def test_chess_dataset_with_phases_multi_worker(self):
        """Test ChessStartStatesDataset with phase_distribution runs multiple epochs.

        Tests the exact configuration used in the notebook.
        """
        config = ChessDatasetConfig(
            max_steps=8,
            phase_distribution={
                "opening": 0.33,
                "middlegame": 0.34,
                "endgame": 0.33,
            },
            quality_filter=False,  # Don't use Stockfish for fast test
        )
        dataset = ChessStartStatesDataset(config)
        dataloader = DataLoader(dataset, batch_size=2, num_workers=2)

        model = DummyGRPOModel()
        epoch_counter = EpochCounterCallback()

        trainer = pl.Trainer(
            max_epochs=3,
            accelerator="cpu",
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            callbacks=[epoch_counter],
        )

        trainer.fit(model, dataloader)

        assert epoch_counter.epochs_completed == 3, (
            f"BUG DETECTED: Expected 3 epochs completed, but got {epoch_counter.epochs_completed}. "
            f"Training stopped prematurely with phase distribution config."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
