"""Test real GRPO training to reproduce the stuck-after-1-epoch issue.

This test uses the actual GRPOChessTransformer with Stockfish evaluation,
unlike test_multi_epoch_training.py which uses a DummyGRPOModel.

The goal is to reproduce the issue where training gets stuck after 1 epoch
when running in Colab with num_workers > 0.
"""
import sys
import os
import time
import shutil

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.grpo_self_play.chess.boards_dataset import ChessStartStatesDataset, ChessDatasetConfig
from src.grpo_self_play.grpo_logic.model import GRPOChessTransformer, GRPOConfig
from src.grpo_self_play.models import ChessTransformerConfig
from src.grpo_self_play.chess.stockfish import StockfishConfig, StockfishManager


class DetailedLoggingCallback(pl.Callback):
    """Callback with detailed logging to track exactly where training gets stuck."""

    def __init__(self):
        self.epochs_started = []
        self.epochs_ended = []
        self.batch_times = []
        self.current_batch_start = None

    def on_train_epoch_start(self, trainer, pl_module):
        epoch = trainer.current_epoch
        self.epochs_started.append(epoch)
        print(f"\n{'='*60}", flush=True)
        print(f"EPOCH {epoch} STARTED at {time.strftime('%H:%M:%S')}", flush=True)
        print(f"{'='*60}", flush=True)

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        self.epochs_ended.append(epoch)
        print(f"\n{'='*60}", flush=True)
        print(f"EPOCH {epoch} ENDED at {time.strftime('%H:%M:%S')}", flush=True)
        print(f"{'='*60}\n", flush=True)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.current_batch_start = time.time()
        print(f"  Batch {batch_idx} started...", end="", flush=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        elapsed = time.time() - self.current_batch_start
        self.batch_times.append(elapsed)
        loss = outputs.get("loss", outputs) if isinstance(outputs, dict) else outputs
        print(f" done in {elapsed:.2f}s (loss={loss:.4f})", flush=True)


class EpochCounterCallback(pl.Callback):
    """Simple callback to count epochs."""

    def __init__(self):
        self.epochs_completed = 0

    def on_train_epoch_end(self, trainer, pl_module):
        self.epochs_completed += 1


def get_stockfish_path():
    """Find Stockfish binary path."""
    possible_paths = [
        "/opt/homebrew/bin/stockfish",    # macOS Apple Silicon (brew)
        "/usr/local/bin/stockfish",       # macOS (brew) or Linux manual
        "/usr/games/stockfish",           # Linux (apt install)
        "stockfish",                       # In PATH
    ]
    for path in possible_paths:
        if shutil.which(path):
            return path
    # Try just "stockfish" and hope it's in PATH
    return "stockfish"


def setup_stockfish():
    """Initialize Stockfish with the correct path before tests."""
    path = get_stockfish_path()
    if not shutil.which(path):
        return None

    # Pre-initialize the reward engine with correct path
    cfg = StockfishConfig(path=path)
    StockfishManager.get_engine("reward_engine", cfg)
    return path


# Use smaller configs for faster testing
SMALL_GRPO_CONFIG = GRPOConfig(
    lr=1e-5,
    num_trajectories=2,      # Reduced from 8
    trajectory_depth=4,      # Reduced from 32
    clip_ratio=0.2,
    kl_coef=0.01,
)

SMALL_TRANSFORMER_CONFIG = ChessTransformerConfig()


class TestRealGRPOTraining:
    """Tests using the real GRPOChessTransformer with Stockfish."""

    @pytest.fixture(autouse=True)
    def stockfish_available(self):
        """Check if Stockfish is available and initialize it."""
        path = setup_stockfish()
        if path is None:
            pytest.skip("Stockfish not available")
        yield path
        # Cleanup after each test - reset the global engine in rewards.py
        from src.grpo_self_play.chess import rewards
        rewards._engine = None
        StockfishManager.close_all()

    def test_real_grpo_single_worker_multi_epoch(self, stockfish_available):
        """Test real GRPO training with single worker for multiple epochs.

        This should work - it's the baseline to compare against multi-worker.
        """
        # Very small config for fast testing
        dataset_config = ChessDatasetConfig(
            max_steps=4,           # Very small
            quality_filter=False,  # Skip quality filtering (no Stockfish in dataset)
        )
        dataset = ChessStartStatesDataset(config=dataset_config)
        dataloader = DataLoader(dataset, batch_size=2, num_workers=0)

        model = GRPOChessTransformer(SMALL_TRANSFORMER_CONFIG, SMALL_GRPO_CONFIG)
        epoch_counter = EpochCounterCallback()
        logging_cb = DetailedLoggingCallback()

        trainer = pl.Trainer(
            max_epochs=2,
            accelerator="cpu",
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            callbacks=[epoch_counter, logging_cb],
        )

        print("\n" + "="*60)
        print("TEST: Real GRPO, single worker, 2 epochs")
        print("="*60)

        trainer.fit(model, dataloader)

        assert epoch_counter.epochs_completed == 2, (
            f"Expected 2 epochs, got {epoch_counter.epochs_completed}. "
            f"Epochs started: {logging_cb.epochs_started}, "
            f"Epochs ended: {logging_cb.epochs_ended}"
        )

    def test_real_grpo_multi_worker_multi_epoch(self, stockfish_available):
        """Test real GRPO training with multiple workers for multiple epochs.

        This is the key test - it should reproduce the stuck-after-1-epoch bug.
        """
        dataset_config = ChessDatasetConfig(
            max_steps=4,           # Very small
            quality_filter=False,  # Skip quality filtering
        )
        dataset = ChessStartStatesDataset(config=dataset_config)
        # Use num_workers=2 to match the notebook configuration
        dataloader = DataLoader(dataset, batch_size=2, num_workers=2)

        model = GRPOChessTransformer(SMALL_TRANSFORMER_CONFIG, SMALL_GRPO_CONFIG)
        epoch_counter = EpochCounterCallback()
        logging_cb = DetailedLoggingCallback()

        trainer = pl.Trainer(
            max_epochs=2,
            accelerator="cpu",
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            callbacks=[epoch_counter, logging_cb],
        )

        print("\n" + "="*60)
        print("TEST: Real GRPO, multi worker (num_workers=2), 2 epochs")
        print("="*60)

        trainer.fit(model, dataloader)

        assert epoch_counter.epochs_completed == 2, (
            f"BUG REPRODUCED: Expected 2 epochs, got {epoch_counter.epochs_completed}. "
            f"Epochs started: {logging_cb.epochs_started}, "
            f"Epochs ended: {logging_cb.epochs_ended}"
        )

    def test_real_grpo_with_quality_filter_single_worker(self, stockfish_available):
        """Test with quality_filter=True (Stockfish in dataset generation).

        This tests if Stockfish usage in dataset generation causes issues.
        """
        dataset_config = ChessDatasetConfig(
            max_steps=4,
            quality_filter=True,   # Use Stockfish for filtering
            stockfish_filter_depth=1,  # Very shallow for speed
        )
        dataset = ChessStartStatesDataset(config=dataset_config)
        dataloader = DataLoader(dataset, batch_size=2, num_workers=0)

        model = GRPOChessTransformer(SMALL_TRANSFORMER_CONFIG, SMALL_GRPO_CONFIG)
        epoch_counter = EpochCounterCallback()
        logging_cb = DetailedLoggingCallback()

        trainer = pl.Trainer(
            max_epochs=2,
            accelerator="cpu",
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            callbacks=[epoch_counter, logging_cb],
        )

        print("\n" + "="*60)
        print("TEST: Real GRPO with quality_filter=True, single worker")
        print("="*60)

        trainer.fit(model, dataloader)

        assert epoch_counter.epochs_completed == 2, (
            f"Expected 2 epochs, got {epoch_counter.epochs_completed}"
        )

    def test_real_grpo_with_quality_filter_multi_worker(self, stockfish_available):
        """Test with quality_filter=True and multiple workers.

        This is the most likely scenario to reproduce issues:
        - Multiple workers each starting Stockfish processes
        - Potential for resource contention and deadlocks
        """
        dataset_config = ChessDatasetConfig(
            max_steps=4,
            quality_filter=True,   # Use Stockfish for filtering
            stockfish_filter_depth=1,
        )
        dataset = ChessStartStatesDataset(config=dataset_config)
        dataloader = DataLoader(dataset, batch_size=2, num_workers=2)

        model = GRPOChessTransformer(SMALL_TRANSFORMER_CONFIG, SMALL_GRPO_CONFIG)
        epoch_counter = EpochCounterCallback()
        logging_cb = DetailedLoggingCallback()

        trainer = pl.Trainer(
            max_epochs=2,
            accelerator="cpu",
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            callbacks=[epoch_counter, logging_cb],
        )

        print("\n" + "="*60)
        print("TEST: Real GRPO with quality_filter=True, multi worker")
        print("="*60)

        trainer.fit(model, dataloader)

        assert epoch_counter.epochs_completed == 2, (
            f"BUG REPRODUCED: Expected 2 epochs, got {epoch_counter.epochs_completed}. "
            f"This likely indicates Stockfish + multi-worker issues."
        )

    def test_real_grpo_longer_training(self, stockfish_available):
        """Test with more batches to see if issue appears over time."""
        dataset_config = ChessDatasetConfig(
            max_steps=8,           # More steps
            quality_filter=False,
        )
        dataset = ChessStartStatesDataset(config=dataset_config)
        dataloader = DataLoader(dataset, batch_size=2, num_workers=2)

        model = GRPOChessTransformer(SMALL_TRANSFORMER_CONFIG, SMALL_GRPO_CONFIG)
        epoch_counter = EpochCounterCallback()
        logging_cb = DetailedLoggingCallback()

        trainer = pl.Trainer(
            max_epochs=3,
            accelerator="cpu",
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            callbacks=[epoch_counter, logging_cb],
        )

        print("\n" + "="*60)
        print("TEST: Real GRPO, longer training (8 steps, 3 epochs)")
        print("="*60)

        trainer.fit(model, dataloader)

        assert epoch_counter.epochs_completed == 3, (
            f"Expected 3 epochs, got {epoch_counter.epochs_completed}"
        )

        # Print timing stats
        if logging_cb.batch_times:
            avg_time = sum(logging_cb.batch_times) / len(logging_cb.batch_times)
            print(f"\nAverage batch time: {avg_time:.2f}s")
            print(f"Total batches: {len(logging_cb.batch_times)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
