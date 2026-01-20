"""Dataset for pretraining on Lichess games from HuggingFace.

This module provides a streaming dataset that loads chess games from the
Icannos/lichess_games HuggingFace dataset and yields (board, action, mask)
tuples for supervised pretraining.

The dataset is process-safe and supports multi-worker DataLoader through
HuggingFace's built-in sharding mechanism.
"""

import chess
import torch
import random
from typing import Optional, Iterator
from dataclasses import dataclass
from torch.utils.data import IterableDataset

from src.grpo_self_play.searchless_chess_imports import MOVE_TO_ACTION, tokenize


@dataclass
class PretrainDatasetConfig:
    """Configuration for the pretraining dataset.

    Attributes:
        min_elo: Minimum player ELO to include games (filters low-quality games)
        max_samples: Maximum number of samples per epoch (None for unlimited)
        skip_first_n_moves: Skip the first N moves (avoid memorizing openings)
        skip_last_n_moves: Skip the last N moves (avoid noisy endgame positions)
        sample_positions_per_game: Number of positions to sample from each game
        buffer_size: Size of shuffle buffer for streaming randomization
        filter_abandoned: Whether to filter out abandoned/timeout games
        dataset_name: HuggingFace dataset name to use
        split: Dataset split to use (default "train")
        is_eval: If True, only include games where hash(Site) % 100 < 5 (5% of data).
                 If False, exclude those games. This creates deterministic train/eval split.
        eval_fraction: Fraction of data to use for evaluation (default 0.05 = 5%)
    """
    min_elo: int = 1500
    max_samples: Optional[int] = None
    skip_first_n_moves: int = 5
    skip_last_n_moves: int = 5
    sample_positions_per_game: int = 3
    buffer_size: int = 10000
    filter_abandoned: bool = True
    dataset_name: str = "Lichess/standard-chess-games"
    split: str = "train"
    is_eval: bool = False  # True for evaluation set, False for training set
    eval_fraction: float = 0.05  # 5% of games go to eval set


def uci_to_action(uci_move: str) -> Optional[int]:
    """Convert UCI move string to action index.

    Args:
        uci_move: Move in UCI format (e.g., "e2e4", "a7a8q")

    Returns:
        Action index or None if move not in action space
    """
    return MOVE_TO_ACTION.get(uci_move)


def parse_pgn_moves(movetext: str) -> list[str]:
    """Parse PGN movetext into list of UCI moves.

    The Lichess dataset stores moves in PGN format (e.g., "1. e4 e5 2. Nf3 Nc6").
    This function converts them to UCI format (e.g., ["e2e4", "e7e5", "g1f3", "b8c6"]).

    Args:
        movetext: PGN movetext string from Lichess dataset

    Returns:
        List of UCI move strings
    """
    if not movetext:
        return []

    import re

    # Remove comments like {[%clk 0:10:00]} and {[%eval 0.17]}
    movetext = re.sub(r'\{[^}]*\}', '', movetext)
    # Remove move numbers like "1." or "1..."
    movetext = re.sub(r'\d+\.+\s*', '', movetext)
    # Remove result at end
    movetext = re.sub(r'\s*(1-0|0-1|1/2-1/2|\*)\s*$', '', movetext)

    # Parse the SAN moves using python-chess
    board = chess.Board()
    uci_moves = []

    for san_move in movetext.split():
        san_move = san_move.strip()
        if not san_move:
            continue
        try:
            move = board.parse_san(san_move)
            uci_moves.append(move.uci())
            board.push(move)
        except (ValueError, chess.InvalidMoveError, chess.AmbiguousMoveError):
            # Stop at first invalid move
            break

    return uci_moves


def get_positions_from_game(
    moves: list[str],
    skip_first_n: int = 5,
    skip_last_n: int = 5,
    sample_n: int = 3,
) -> list[tuple[str, str, int]]:
    """Extract (FEN, move_played, move_number) tuples from a game.

    Replays the game and samples positions, skipping early opening moves
    and late endgame moves to focus on the most instructive positions.

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

    # Play through the game and collect positions
    for i, uci_move in enumerate(moves):
        # Skip early moves (opening book)
        if i < skip_first_n:
            try:
                board.push_uci(uci_move)
            except (ValueError, chess.InvalidMoveError):
                return positions  # Invalid move, stop processing
            continue

        # Skip late moves (endgame noise)
        if i >= len(moves) - skip_last_n:
            break

        # Record position before the move is made
        fen = board.fen()
        positions.append((fen, uci_move, i))

        # Make the move on the board
        try:
            board.push_uci(uci_move)
        except (ValueError, chess.InvalidMoveError):
            break  # Invalid move, stop processing

    # Randomly sample if we have more positions than requested
    if len(positions) > sample_n:
        positions = random.sample(positions, sample_n)

    return positions


class ChessPretrainDataset(IterableDataset):
    """Streaming dataset for chess pretraining from HuggingFace.

    Streams games from the Icannos/lichess_games dataset and yields
    (board_tensor, target_action, legal_moves_mask) tuples for supervised learning.

    This dataset is process-safe: when used with multiple DataLoader workers,
    each worker processes a disjoint shard of the data using HuggingFace's
    built-in sharding mechanism.

    Example:
        >>> config = PretrainDatasetConfig(min_elo=1800, max_samples=100000)
        >>> dataset = ChessPretrainDataset(config)
        >>> dataloader = DataLoader(
        ...     dataset,
        ...     batch_size=256,
        ...     num_workers=4,
        ...     collate_fn=collate_pretrain_batch
        ... )
        >>> for boards, actions, masks in dataloader:
        ...     # boards: [B, 77], actions: [B], masks: [B, 1968]
        ...     pass
    """

    def __init__(self, config: PretrainDatasetConfig = PretrainDatasetConfig()):
        """Initialize the pretraining dataset.

        Args:
            config: Dataset configuration
        """
        self.config = config
        self._dataset = None
        self._action_space_size = max(MOVE_TO_ACTION.values()) + 1

    def _load_dataset(self):
        """Lazily load the HuggingFace dataset."""
        if self._dataset is None:
            from datasets import load_dataset

            # Load Lichess games dataset in streaming mode
            # The split can be "train" for all data, or a specific year-month like "2024-01"
            self._dataset = load_dataset(
                self.config.dataset_name,
                split=self.config.split,
                streaming=True,
            )

    def _filter_game(self, game: dict) -> bool:
        """Check if a game meets the quality criteria.

        Args:
            game: Game record from the dataset

        Returns:
            True if game should be included
        """
        # Hash-based train/eval split using game Site URL (unique per game)
        # This ensures deterministic, non-overlapping train and eval sets
        site = game.get('Site', '')
        if site:
            # Use hash to deterministically assign games to train or eval
            hash_val = hash(site) % 10000
            threshold = int(self.config.eval_fraction * 10000)
            is_eval_game = hash_val < threshold

            if self.config.is_eval and not is_eval_game:
                return False  # Eval set only wants eval games
            if not self.config.is_eval and is_eval_game:
                return False  # Train set excludes eval games

        # Filter by ELO - both players must meet minimum
        white_elo = game.get('WhiteElo')
        black_elo = game.get('BlackElo')

        if white_elo and black_elo:
            try:
                min_game_elo = min(int(white_elo), int(black_elo))
                if min_game_elo < self.config.min_elo:
                    return False
            except (ValueError, TypeError):
                return False
        else:
            # Skip games without ELO info
            return False

        # Must have a reasonable number of moves
        movetext = game.get('movetext', '')
        if not movetext or len(movetext.split()) < 10:
            return False

        # Filter abandoned/timeout games if configured
        if self.config.filter_abandoned:
            termination = game.get('Termination', '').lower()
            if 'abandoned' in termination:
                return False

        return True

    def _process_game(self, game: dict) -> Iterator[tuple[torch.Tensor, int, torch.Tensor]]:
        """Process a single game and yield training samples.

        Args:
            game: Game record from the dataset

        Yields:
            (board_tensor, target_action, legal_moves_mask) tuples
        """
        movetext = game.get('movetext', '')
        moves = parse_pgn_moves(movetext)

        # Extract positions from this game
        positions = get_positions_from_game(
            moves,
            skip_first_n=self.config.skip_first_n_moves,
            skip_last_n=self.config.skip_last_n_moves,
            sample_n=self.config.sample_positions_per_game,
        )

        for fen, uci_move, move_num in positions:
            # Convert move to action index
            action_idx = uci_to_action(uci_move)
            if action_idx is None:
                continue

            # Tokenize the FEN string
            try:
                token_ids = list(tokenize(fen))
                board_tensor = torch.tensor(token_ids, dtype=torch.long)
            except Exception:
                continue

            # Build legal moves mask
            try:
                board = chess.Board(fen)
                legal_mask = torch.zeros(self._action_space_size, dtype=torch.bool)
                for move in board.legal_moves:
                    move_idx = MOVE_TO_ACTION.get(move.uci())
                    if move_idx is not None:
                        legal_mask[move_idx] = True
            except Exception:
                continue

            # Verify the target move is actually legal
            if not legal_mask[action_idx]:
                continue

            yield board_tensor, action_idx, legal_mask

    def __iter__(self) -> Iterator[tuple[torch.Tensor, int, torch.Tensor]]:
        """Iterate over the dataset, yielding training samples.

        When used with multiple DataLoader workers, each worker processes
        a disjoint shard of the data for process safety.
        """
        self._load_dataset()

        worker_info = torch.utils.data.get_worker_info()

        if worker_info is not None:
            # Multi-worker mode: shard the dataset across workers
            num_workers = worker_info.num_workers
            worker_id = worker_info.id

            # Use HuggingFace's built-in sharding for process safety
            dataset_shard = self._dataset.shard(
                num_shards=num_workers,
                index=worker_id
            )

            # Set unique random seed per worker for shuffle buffer
            random.seed(42 + worker_id * 1000)
        else:
            # Single-worker mode: use full dataset
            dataset_shard = self._dataset
            random.seed(42)

        samples_yielded = 0

        # Shuffle buffer for better randomization in streaming mode
        buffer: list[tuple[torch.Tensor, int, torch.Tensor]] = []

        for game in dataset_shard:
            # Filter by quality criteria
            if not self._filter_game(game):
                continue

            # Process game and add samples to buffer
            for sample in self._process_game(game):
                buffer.append(sample)

                # Yield from buffer when it's large enough
                if len(buffer) >= self.config.buffer_size:
                    random.shuffle(buffer)
                    # Yield half the buffer, keeping some for mixing
                    while len(buffer) > self.config.buffer_size // 2:
                        yield buffer.pop()
                        samples_yielded += 1

                        # Check max samples limit
                        if self.config.max_samples and samples_yielded >= self.config.max_samples:
                            return

        # Yield remaining samples in buffer
        random.shuffle(buffer)
        for sample in buffer:
            yield sample
            samples_yielded += 1
            if self.config.max_samples and samples_yielded >= self.config.max_samples:
                return


def collate_pretrain_batch(
    batch: list[tuple[torch.Tensor, int, torch.Tensor]]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate function for DataLoader.

    Args:
        batch: List of (board_tensor, action, legal_mask) tuples

    Returns:
        Tuple of:
            - boards: [B, 77] token IDs
            - actions: [B] target action indices
            - legal_masks: [B, num_actions] boolean masks
    """
    boards, actions, masks = zip(*batch)

    boards = torch.stack(boards)  # [B, 77]
    actions = torch.tensor(actions, dtype=torch.long)  # [B]
    masks = torch.stack(masks)  # [B, num_actions]

    return boards, actions, masks
