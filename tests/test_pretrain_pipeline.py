"""Tests for the pretraining pipeline.

This module tests the pretraining components:
- ChessPretrainDataset: Streaming dataset from HuggingFace
- PretrainChessTransformer: PyTorch Lightning module for supervised learning
- Hash-based train/eval splitting
- Collate function and data loading
"""
import sys
import os
import json
from unittest.mock import MagicMock, patch

# Mock the searchless_chess imports before importing anything that uses them
with open('tests/resources/ACTION_TO_MOVE.json', 'r') as f:
    ACTION_TO_MOVE = {int(k): v for k, v in json.load(f).items()}
with open('tests/resources/MOVE_TO_ACTION.json', 'r') as f:
    MOVE_TO_ACTION = json.load(f)

mock_utils = MagicMock()
mock_utils.ACTION_TO_MOVE = ACTION_TO_MOVE
mock_utils.MOVE_TO_ACTION = MOVE_TO_ACTION

# Create a proper tokenize function that returns the right format
def mock_tokenize(fen: str):
    """Mock tokenize that returns 77 tokens like the real one."""
    import numpy as np
    # Return 77 random tokens in range 0-29 (like real tokenizer)
    return np.array([i % 30 for i in range(77)], dtype=np.uint8)

mock_tokenizer = MagicMock()
mock_tokenizer.tokenize = mock_tokenize
mock_tokenizer.SEQUENCE_LENGTH = 77

# Create mock modules
sys.modules['src.searchless_chess_model'] = MagicMock()
sys.modules['src.searchless_chess_model.searchless_chess_code'] = MagicMock()
sys.modules['src.searchless_chess_model.searchless_chess_code.utils'] = mock_utils
sys.modules['src.searchless_chess_model.searchless_chess_code.tokenizer'] = mock_tokenizer

import pytest
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader


class TestPretrainDatasetConfig:
    """Tests for PretrainDatasetConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        from src.pretrain import PretrainDatasetConfig

        config = PretrainDatasetConfig()

        assert config.min_elo == 1500
        assert config.max_samples is None
        assert config.skip_first_n_moves == 5
        assert config.skip_last_n_moves == 5
        assert config.sample_positions_per_game == 3
        assert config.buffer_size == 10000
        assert config.is_eval is False
        assert config.eval_fraction == 0.05

    def test_custom_config(self):
        """Test custom configuration values."""
        from src.pretrain import PretrainDatasetConfig

        config = PretrainDatasetConfig(
            min_elo=1800,
            max_samples=1000,
            is_eval=True,
            eval_fraction=0.10
        )

        assert config.min_elo == 1800
        assert config.max_samples == 1000
        assert config.is_eval is True
        assert config.eval_fraction == 0.10


class TestPGNParsing:
    """Tests for PGN move parsing."""

    def test_parse_simple_pgn(self):
        """Test parsing simple PGN movetext."""
        from src.pretrain.pretrain_dataset import parse_pgn_moves

        movetext = "1. e4 e5 2. Nf3 Nc6"
        moves = parse_pgn_moves(movetext)

        assert len(moves) == 4
        assert moves[0] == "e2e4"
        assert moves[1] == "e7e5"
        assert moves[2] == "g1f3"
        assert moves[3] == "b8c6"

    def test_parse_pgn_with_clock(self):
        """Test parsing PGN with clock annotations."""
        from src.pretrain.pretrain_dataset import parse_pgn_moves

        movetext = "1. e4 {[%clk 0:10:00]} e5 {[%clk 0:10:00]} 2. Nf3"
        moves = parse_pgn_moves(movetext)

        assert len(moves) == 3
        assert moves[0] == "e2e4"
        assert moves[1] == "e7e5"
        assert moves[2] == "g1f3"

    def test_parse_pgn_with_result(self):
        """Test parsing PGN with game result."""
        from src.pretrain.pretrain_dataset import parse_pgn_moves

        movetext = "1. e4 e5 1-0"
        moves = parse_pgn_moves(movetext)

        assert len(moves) == 2
        assert moves[0] == "e2e4"
        assert moves[1] == "e7e5"

    def test_parse_empty_pgn(self):
        """Test parsing empty PGN."""
        from src.pretrain.pretrain_dataset import parse_pgn_moves

        assert parse_pgn_moves("") == []
        assert parse_pgn_moves(None) == []


class TestPositionExtraction:
    """Tests for extracting positions from games."""

    def test_get_positions_basic(self):
        """Test basic position extraction."""
        from src.pretrain.pretrain_dataset import get_positions_from_game

        # A game with 10 moves
        moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6", "e1g1", "f8e7"]

        positions = get_positions_from_game(
            moves,
            skip_first_n=2,
            skip_last_n=2,
            sample_n=3
        )

        # Should get positions from moves 2-7 (indices), sampled down to 3
        assert len(positions) <= 3
        for fen, uci_move, move_num in positions:
            assert isinstance(fen, str)
            assert isinstance(uci_move, str)
            assert 2 <= move_num < 8

    def test_get_positions_short_game(self):
        """Test position extraction from game too short."""
        from src.pretrain.pretrain_dataset import get_positions_from_game

        moves = ["e2e4", "e7e5"]  # Only 2 moves

        positions = get_positions_from_game(
            moves,
            skip_first_n=5,
            skip_last_n=5,
            sample_n=3
        )

        # Game is too short, should return empty
        assert len(positions) == 0


class TestUCIToAction:
    """Tests for UCI move to action conversion."""

    def test_valid_move(self):
        """Test conversion of valid UCI move."""
        from src.pretrain.pretrain_dataset import uci_to_action

        # e2e4 should be in the action space
        action = uci_to_action("e2e4")
        assert action is not None
        assert isinstance(action, int)

    def test_invalid_move(self):
        """Test conversion of invalid UCI move."""
        from src.pretrain.pretrain_dataset import uci_to_action

        action = uci_to_action("invalid")
        assert action is None


class TestCollateFn:
    """Tests for the collate function."""

    def test_collate_batch(self):
        """Test collating a batch of samples."""
        from src.pretrain import collate_pretrain_batch

        # Create dummy samples
        batch = [
            (torch.randint(0, 30, (77,)), 100, torch.ones(1968, dtype=torch.bool)),
            (torch.randint(0, 30, (77,)), 200, torch.ones(1968, dtype=torch.bool)),
            (torch.randint(0, 30, (77,)), 300, torch.ones(1968, dtype=torch.bool)),
        ]

        boards, actions, masks = collate_pretrain_batch(batch)

        assert boards.shape == (3, 77)
        assert actions.shape == (3,)
        assert masks.shape == (3, 1968)
        assert actions.tolist() == [100, 200, 300]


class TestPretrainChessTransformer:
    """Tests for the PretrainChessTransformer Lightning module."""

    def test_model_creation(self):
        """Test creating the pretraining model."""
        from src.pretrain.pretrain import PretrainChessTransformer, PretrainConfig
        from src.models import ChessTransformerConfig

        transformer_config = ChessTransformerConfig(
            embed_dim=64,
            num_layers=2,
            num_heads=2
        )
        pretrain_config = PretrainConfig(
            lr=1e-4,
            batch_size=4,
            use_wandb=False
        )

        model = PretrainChessTransformer(transformer_config, pretrain_config)

        assert model is not None
        param_count = sum(p.numel() for p in model.parameters())
        assert param_count > 0

    def test_forward_pass(self):
        """Test forward pass through the model."""
        from src.pretrain.pretrain import PretrainChessTransformer, PretrainConfig
        from src.models import ChessTransformerConfig

        transformer_config = ChessTransformerConfig(
            embed_dim=64,
            num_layers=2,
            num_heads=2
        )
        pretrain_config = PretrainConfig(use_wandb=False)

        model = PretrainChessTransformer(transformer_config, pretrain_config)

        # Create dummy input
        batch_size = 4
        seq_len = 77
        x = torch.randint(0, 30, (batch_size, seq_len))

        # Forward pass
        logits = model(x)

        assert logits.shape == (batch_size, 1968)

    def test_training_step(self):
        """Test a single training step."""
        from src.pretrain.pretrain import PretrainChessTransformer, PretrainConfig
        from src.models import ChessTransformerConfig

        transformer_config = ChessTransformerConfig(
            embed_dim=64,
            num_layers=2,
            num_heads=2
        )
        pretrain_config = PretrainConfig(use_wandb=False)

        model = PretrainChessTransformer(transformer_config, pretrain_config)

        # Create dummy batch
        batch_size = 4
        boards = torch.randint(0, 30, (batch_size, 77))
        actions = torch.randint(0, 1968, (batch_size,))
        masks = torch.ones(batch_size, 1968, dtype=torch.bool)

        # Training step
        loss = model.training_step((boards, actions, masks), 0)

        assert loss is not None
        assert loss.requires_grad
        assert not torch.isnan(loss)

    def test_validation_step(self):
        """Test a single validation step."""
        from src.pretrain.pretrain import PretrainChessTransformer, PretrainConfig
        from src.models import ChessTransformerConfig

        transformer_config = ChessTransformerConfig(
            embed_dim=64,
            num_layers=2,
            num_heads=2
        )
        pretrain_config = PretrainConfig(use_wandb=False)

        model = PretrainChessTransformer(transformer_config, pretrain_config)

        # Create dummy batch
        batch_size = 4
        boards = torch.randint(0, 30, (batch_size, 77))
        actions = torch.randint(0, 1968, (batch_size,))
        masks = torch.ones(batch_size, 1968, dtype=torch.bool)

        # Validation step
        loss = model.validation_step((boards, actions, masks), 0)

        assert loss is not None
        assert not torch.isnan(loss)


class TestHashBasedSplit:
    """Tests for the hash-based train/eval split."""

    def test_split_is_deterministic(self):
        """Test that the same site URL always goes to the same split."""
        # The split is based on hash(site) % 10000 < threshold
        site1 = "https://lichess.org/abc123"
        site2 = "https://lichess.org/def456"

        # Hash should be consistent
        hash1_a = hash(site1) % 10000
        hash1_b = hash(site1) % 10000

        assert hash1_a == hash1_b

    def test_split_separates_data(self):
        """Test that train and eval get different portions."""
        eval_fraction = 0.05
        threshold = int(eval_fraction * 10000)

        # Generate some test sites
        train_count = 0
        eval_count = 0

        for i in range(1000):
            site = f"https://lichess.org/game{i}"
            hash_val = hash(site) % 10000

            if hash_val < threshold:
                eval_count += 1
            else:
                train_count += 1

        # Should be roughly 5% eval, 95% train
        eval_ratio = eval_count / 1000
        assert 0.02 < eval_ratio < 0.10  # Allow some variance


class TestPretrainLoadConfig:
    """Tests for PretrainLoadConfig."""

    def test_default_config(self):
        """Test default configuration."""
        from src.pretrain import PretrainLoadConfig

        config = PretrainLoadConfig()

        assert config.checkpoint_path is None
        assert config.freeze_layers == 0

    def test_custom_config(self):
        """Test custom configuration."""
        from src.pretrain import PretrainLoadConfig

        config = PretrainLoadConfig(
            checkpoint_path="/path/to/checkpoint.pt",
            freeze_layers=2
        )

        assert config.checkpoint_path == "/path/to/checkpoint.pt"
        assert config.freeze_layers == 2


class TestIntegration:
    """Integration tests for the full pretraining pipeline."""

    def test_dataloader_iteration(self):
        """Test that DataLoader can iterate over the dataset."""
        from src.pretrain import collate_pretrain_batch
        from torch.utils.data import IterableDataset

        # Create a mock dataset that yields proper samples
        class MockPretrainDataset(IterableDataset):
            def __init__(self, num_samples=10):
                self.num_samples = num_samples

            def __iter__(self):
                for i in range(self.num_samples):
                    board = torch.randint(0, 30, (77,))
                    action = i % 1968
                    mask = torch.ones(1968, dtype=torch.bool)
                    yield board, action, mask

        dataset = MockPretrainDataset(num_samples=10)
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            collate_fn=collate_pretrain_batch
        )

        batches = list(dataloader)

        assert len(batches) == 3  # 10 samples / 4 batch_size = 3 batches

        for boards, actions, masks in batches:
            assert boards.dim() == 2
            assert boards.shape[1] == 77
            assert actions.dim() == 1
            assert masks.dim() == 2

    def test_training_loop_mock(self):
        """Test a mock training loop with Lightning."""
        from src.pretrain.pretrain import PretrainChessTransformer, PretrainConfig
        from src.pretrain import collate_pretrain_batch
        from src.models import ChessTransformerConfig
        from torch.utils.data import IterableDataset

        # Create mock dataset
        class MockPretrainDataset(IterableDataset):
            def __init__(self, num_samples=20):
                self.num_samples = num_samples

            def __iter__(self):
                for i in range(self.num_samples):
                    board = torch.randint(0, 30, (77,))
                    action = i % 1968
                    mask = torch.ones(1968, dtype=torch.bool)
                    yield board, action, mask

        # Create model
        transformer_config = ChessTransformerConfig(
            embed_dim=32,
            num_layers=1,
            num_heads=2
        )
        pretrain_config = PretrainConfig(
            lr=1e-3,
            batch_size=4,
            num_epochs=2,
            use_wandb=False
        )

        model = PretrainChessTransformer(transformer_config, pretrain_config)

        # Create dataloaders
        train_dataset = MockPretrainDataset(num_samples=20)
        train_loader = DataLoader(
            train_dataset,
            batch_size=4,
            collate_fn=collate_pretrain_batch
        )

        # Create trainer
        trainer = pl.Trainer(
            max_epochs=2,
            accelerator="cpu",
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
        )

        # Fit model
        trainer.fit(model, train_loader)

        # Should complete without errors
        assert trainer.current_epoch == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
