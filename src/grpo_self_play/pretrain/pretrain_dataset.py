"""Dataset for pretraining on chess games from HuggingFace.

Uses angeluriot/chess_games: 14M high-ELO games (7.3GB download).
Mean ELO ~2355, moves already in UCI format - no parsing needed.
"""

import os
import chess
import torch
import random
from typing import Optional
from dataclasses import dataclass
from torch.utils.data import Dataset
from datasets import load_dataset
from tqdm import tqdm

from src.grpo_self_play.searchless_chess_imports import MOVE_TO_ACTION, tokenize


@dataclass
class PretrainDatasetConfig:
    """Configuration for the pretraining dataset.

    Uses angeluriot/chess_games: 14M high-ELO games (7.3GB download).
    Mean ELO ~2355, moves already in UCI format.

    Attributes:
        min_elo: Minimum player ELO to include games
        max_samples: Maximum number of samples per epoch (None for unlimited)
        skip_first_n_moves: Skip the first N moves (avoid memorizing openings)
        skip_last_n_moves: Skip the last N moves (avoid noisy endgame positions)
        sample_positions_per_game: Number of positions to sample from each game
        is_eval: If True, use eval portion of hash-based split.
        eval_fraction: Fraction of data to use for evaluation (default 0.05 = 5%)
    """
    min_elo: int = 2000
    max_samples: Optional[int] = None
    skip_first_n_moves: int = 5
    skip_last_n_moves: int = 5
    sample_positions_per_game: int = 3
    is_eval: bool = False
    eval_fraction: float = 0.05


def uci_to_action(uci_move: str) -> Optional[int]:
    """Convert UCI move string to action index."""
    return MOVE_TO_ACTION.get(uci_move)


def get_positions_from_game(
    moves: list[str],
    skip_first_n: int = 5,
    skip_last_n: int = 5,
    sample_n: int = 3,
) -> list[tuple[str, str, int]]:
    """Extract (FEN, move_played, move_number) tuples from a game.

    Args:
        moves: List of UCI moves
        skip_first_n: Skip first N moves (opening book territory)
        skip_last_n: Skip last N moves (endgame/resignation noise)
        sample_n: Number of positions to randomly sample

    Returns:
        List of (fen, uci_move, move_number) tuples
    """
    if len(moves) <= skip_first_n + skip_last_n:
        return []

    board = chess.Board()
    positions = []

    for i, uci_move in enumerate(moves):
        if i < skip_first_n:
            try:
                board.push_uci(uci_move)
            except (ValueError, chess.InvalidMoveError):
                return positions
            continue

        if i >= len(moves) - skip_last_n:
            break

        fen = board.fen()
        positions.append((fen, uci_move, i))

        try:
            board.push_uci(uci_move)
        except (ValueError, chess.InvalidMoveError):
            break

    if len(positions) > sample_n:
        positions = random.sample(positions, sample_n)

    return positions


class ChessPretrainDataset(Dataset):
    """Dataset for chess pretraining from angeluriot/chess_games.

    Downloads the full dataset (7.3GB) and processes games into
    (board_tensor, target_action, legal_moves_mask) tuples.

    Example:
        >>> config = PretrainDatasetConfig(min_elo=2000)
        >>> dataset = ChessPretrainDataset(config)
        >>> dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    """

    def __init__(self, config: PretrainDatasetConfig = PretrainDatasetConfig()):
        """Initialize the dataset - downloads and processes all games."""
        self.config = config
        self._action_space_size = max(MOVE_TO_ACTION.values()) + 1
        self._samples: list[tuple[torch.Tensor, int, torch.Tensor]] = []

        self._load_and_process()

    def _load_and_process(self):
        """Download dataset and process all games into samples."""
        print("Downloading angeluriot/chess_games (7.3GB)...")
        dataset = load_dataset("angeluriot/chess_games", split="train")
        print(f"Loaded {len(dataset):,} games")

        # Fast batched filtering
        print(f"Filtering games (min_elo={self.config.min_elo})...")
        min_elo = self.config.min_elo
        eval_frac = self.config.eval_fraction
        is_eval = self.config.is_eval

        def batch_filter(batch):
            """Filter a batch of games - much faster than per-example."""
            keep = []
            for i in range(len(batch['white_elo'])):
                white_elo = batch['white_elo'][i]
                black_elo = batch['black_elo'][i]

                # Skip if ELO is missing
                if white_elo is None or black_elo is None:
                    keep.append(False)
                    continue
                # ELO filter
                if white_elo < min_elo or black_elo < min_elo:
                    keep.append(False)
                    continue
                # Moves filter
                if len(batch['moves_uci'][i]) < 10:
                    keep.append(False)
                    continue
                # Hash-based train/eval split
                game_id = f"{batch['date'][i]}-{white_elo}-{black_elo}"
                hash_val = hash(game_id) % 10000
                is_eval_game = hash_val < (eval_frac * 10000)
                if is_eval_game != is_eval:
                    keep.append(False)
                    continue
                keep.append(True)
            return keep

        dataset = dataset.filter(batch_filter, batched=True, batch_size=10000, desc="Filtering")
        print(f"After filtering: {len(dataset):,} games")

        # Limit dataset size if max_samples is set
        if self.config.max_samples:
            max_games = self.config.max_samples // self.config.sample_positions_per_game + 1000
            if len(dataset) > max_games:
                dataset = dataset.select(range(max_games))
                print(f"Limited to {len(dataset):,} games")

        # Process games into samples (sequential - requires board state)
        print("Processing games into samples...")
        for game in tqdm(dataset, desc="Processing"):
            for sample in self._process_game(game):
                self._samples.append(sample)
                if self.config.max_samples and len(self._samples) >= self.config.max_samples:
                    break

            if self.config.max_samples and len(self._samples) >= self.config.max_samples:
                break

        print(f"Done: {len(self._samples):,} samples")

    def _filter_game(self, game: dict) -> bool:
        """Check if a game meets the quality criteria."""
        # Hash-based train/eval split using date (unique identifier)
        game_id = f"{game.get('date', '')}-{game.get('white_elo', '')}-{game.get('black_elo', '')}"
        hash_val = hash(game_id) % 10000
        threshold = int(self.config.eval_fraction * 10000)
        is_eval_game = hash_val < threshold

        if self.config.is_eval and not is_eval_game:
            return False
        if not self.config.is_eval and is_eval_game:
            return False

        # Filter by ELO - both players must meet minimum
        white_elo = game.get('white_elo')
        black_elo = game.get('black_elo')

        if white_elo is None or black_elo is None:
            return False

        if min(white_elo, black_elo) < self.config.min_elo:
            return False

        # Must have enough moves
        moves = game.get('moves_uci', [])
        if not moves or len(moves) < 10:
            return False

        return True

    def _process_game(self, game: dict):
        """Process a single game and yield training samples."""
        moves = game.get('moves_uci', [])

        positions = get_positions_from_game(
            moves,
            skip_first_n=self.config.skip_first_n_moves,
            skip_last_n=self.config.skip_last_n_moves,
            sample_n=self.config.sample_positions_per_game,
        )

        for fen, uci_move, move_num in positions:
            action_idx = uci_to_action(uci_move)
            if action_idx is None:
                continue

            try:
                token_ids = list(tokenize(fen))
                board_tensor = torch.tensor(token_ids, dtype=torch.long)
            except Exception:
                continue

            try:
                board = chess.Board(fen)
                legal_mask = torch.zeros(self._action_space_size, dtype=torch.bool)
                for move in board.legal_moves:
                    move_idx = MOVE_TO_ACTION.get(move.uci())
                    if move_idx is not None:
                        legal_mask[move_idx] = True
            except Exception:
                continue

            if not legal_mask[action_idx]:
                continue

            yield board_tensor, action_idx, legal_mask

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, torch.Tensor]:
        return self._samples[idx]


def collate_pretrain_batch(
    batch: list[tuple[torch.Tensor, int, torch.Tensor]]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate function for DataLoader.

    Returns:
        Tuple of (boards [B, 77], actions [B], legal_masks [B, num_actions])
    """
    boards, actions, masks = zip(*batch)

    boards = torch.stack(boards)
    actions = torch.tensor(actions, dtype=torch.long)
    masks = torch.stack(masks)

    return boards, actions, masks
