"""Dataset of random chess boards."""

import chess
import random
import torch
from collections import deque

from typing import Any, Optional, Dict
from dataclasses import dataclass
from torch.utils.data import IterableDataset
from src.chess.rewards import evaluate_fen


def generate_random_board(step_num=30):
    """Generate a random board position by making random moves from starting position.
    
    Args:
        step_num: Maximum number of random moves to make
        
    Returns:
        Chess board after random moves
    """
    board = chess.Board()
    random_steps = random.randint(0, step_num)
    for _ in range(random_steps):
        if board.is_game_over(): break
        board.push(random.choice(list(board.legal_moves)))
    return board


def get_game_phase(board: chess.Board) -> str:
    """Determine the game phase (opening, middlegame, or endgame).
    
    Args:
        board: Chess board position
        
    Returns:
        "opening", "middlegame", or "endgame"
    """
    move_count = board.fullmove_number * 2 - (1 if board.turn == chess.BLACK else 0)
    
    # Count material (excluding kings)
    material_count = sum(
        len(board.pieces(pt, color))
        for pt in [chess.PAWN, chess.ROOK, chess.KNIGHT, chess.BISHOP, chess.QUEEN]
        for color in [chess.WHITE, chess.BLACK]
    )
    
    # Endgame: few pieces remaining (typically < 12-14 pieces)
    if material_count <= 12:
        return "endgame"
    # Opening: early moves (typically first 15 moves)
    elif move_count <= 15:
        return "opening"
    # Middlegame: everything else
    else:
        return "middlegame"

def evaluate_position_quality(board: chess.Board, depth: int = 2) -> Optional[float]:
    """Quick Stockfish evaluation to check position quality.
    
    Args:
        board: Chess board position
        depth: Stockfish search depth (shallow for speed)
        
    Returns:
        Centipawn evaluation from White's perspective, or None if evaluation fails
    """
    try:
        fen = board.fen()
        pov_is_white = board.turn == chess.WHITE
        eval_cp = evaluate_fen(fen, pov_is_white, movetime_ms=0, depth=depth)
        return eval_cp
    except Exception:
        return None

def generate_opening_position(max_moves: int = 15) -> chess.Board:
    """Generate a realistic opening position using common opening moves.
    
    Args:
        max_moves: Maximum number of opening moves to make
        
    Returns:
        Chess board in opening phase
    """
    board = chess.Board()
    
    # Common first moves for White
    first_moves = [
        chess.Move.from_uci("e2e4"),  # King's pawn
        chess.Move.from_uci("d2d4"),  # Queen's pawn
        chess.Move.from_uci("g1f3"),  # King's knight
        chess.Move.from_uci("c2c4"),  # English opening
    ]
    
    # Make first move
    if first_moves:
        first_move = random.choice(first_moves)
        if first_move in board.legal_moves:
            board.push(first_move)
    
    # Continue with semi-random play (preferring development moves)
    moves_made = 1
    while moves_made < max_moves and not board.is_game_over():
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            break
        
        # Prefer piece development over pawn moves in opening
        piece_moves = [m for m in legal_moves if board.piece_at(m.from_square) and 
                      board.piece_at(m.from_square).piece_type != chess.PAWN]
        
        if piece_moves and random.random() < 0.6:  # 60% chance to prefer piece moves
            move = random.choice(piece_moves)
        else:
            move = random.choice(legal_moves)
        
        board.push(move)
        moves_made += 1
    
    return board


def generate_middlegame_position(min_moves: int = 15, max_moves: int = 40) -> chess.Board:
    """Generate a middlegame position from a reasonable starting point.
    
    Args:
        min_moves: Minimum moves to reach middlegame
        max_moves: Maximum moves for middlegame
        
    Returns:
        Chess board in middlegame phase
    """
    # Start from an opening position
    board = generate_opening_position(max_moves=min_moves)
    
    # Continue with random play to reach middlegame
    target_moves = random.randint(min_moves, max_moves)
    moves_made = len(board.move_stack)
    
    while moves_made < target_moves and not board.is_game_over():
        legal_moves = list[Any](board.legal_moves)
        if not legal_moves:
            break
        board.push(random.choice(legal_moves))
        moves_made += 1
    
    return board



def generate_endgame_position() -> chess.Board: # TODO: This is  not working as expected, it should be a function that generates a random endgame position.
    """Generate an endgame position by removing pieces from a middlegame position.
    
    Returns:
        Chess board in endgame phase
    """
    # Start with a middlegame position
    board = generate_middlegame_position(min_moves=20, max_moves=35)
    
    # Remove pieces to create endgame (keep kings, remove other pieces randomly)
    pieces_to_remove = []
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.piece_type != chess.KING:
            pieces_to_remove.append(square)
    
    # Remove random pieces until we have endgame material (<= 12 pieces total)
    target_pieces = random.randint(6, 12)  # Endgame typically has 6-12 pieces
    current_pieces = len([p for p in pieces_to_remove if board.piece_at(p)])
    
    # We need to remove pieces, but we can't directly remove them from python-chess Board
    # Instead, we'll generate a new position by making moves that trade pieces
    # For simplicity, we'll just continue playing until we naturally reach endgame material
    
    # Count material
    def count_material(b: chess.Board) -> int:
        return sum(
            len(b.pieces(pt, color))
            for pt in [chess.PAWN, chess.ROOK, chess.KNIGHT, chess.BISHOP, chess.QUEEN]
            for color in [chess.WHITE, chess.BLACK]
        )
    
    # Play random moves until we reach endgame material
    max_attempts = 100
    attempts = 0
    while count_material(board) > 12 and attempts < max_attempts and not board.is_game_over():
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            break
        
        # Prefer captures to reduce material
        captures = [m for m in legal_moves if board.is_capture(m)]
        if captures:
            move = random.choice(captures)
        else:
            move = random.choice(legal_moves)
        
        board.push(move)
        attempts += 1
    
    return board



def generate_position_by_phase(phase: str) -> chess.Board:
    """Generate a position for a specific game phase.
    
    Args:
        phase: "opening", "middlegame", or "endgame"
        
    Returns:
        Chess board in the specified phase
    """
    if phase == "opening":
        return generate_opening_position()
    elif phase == "middlegame":
        return generate_middlegame_position()
    elif phase == "endgame":
        return generate_endgame_position()
    else:
        raise ValueError(f"Unknown phase: {phase}. Must be 'opening', 'middlegame', or 'endgame'")


def generate_quality_filtered_board(
    step_num: int = 30,
    min_eval_cp: int = -200,
    max_eval_cp: int = 200,
    filter_depth: int = 2,
    max_attempts: int = 50,
    phase: Optional[str] = None
) -> Optional[chess.Board]:
    """Generate a random board position filtered by Stockfish evaluation quality.
    
    Args:
        step_num: Maximum number of random moves (if phase is None)
        min_eval_cp: Minimum centipawn evaluation to accept
        max_eval_cp: Maximum centipawn evaluation to accept
        filter_depth: Stockfish depth for filtering (shallow for speed)
        max_attempts: Maximum attempts to generate a valid position
        phase: Optional phase to generate ("opening", "middlegame", "endgame")
        
    Returns:
        Chess board within evaluation range, or None if no valid position found
    """
    for attempt in range(max_attempts):
        # Generate position
        if phase:
            board = generate_position_by_phase(phase)
        else:
            board = generate_random_board(step_num)
        
        # Skip if game over or no legal moves
        if board.is_game_over() or not list(board.legal_moves):
            continue
        
        # Evaluate position quality
        eval_cp = evaluate_position_quality(board, depth=filter_depth)
        if eval_cp is None:
            continue
        
        # Check if evaluation is within acceptable range
        if min_eval_cp <= eval_cp <= max_eval_cp:
            return board
    
    # If we couldn't find a good position, return a random one anyway
    if phase:
        return generate_position_by_phase(phase)
    else:
        return generate_random_board(step_num)


@dataclass
class ChessDatasetConfig:
    """Configuration for the Chess Start States Dataset.
    
    Attributes:
        max_steps: Maximum number of positions to generate per epoch
        random_walk_gen_steps: Maximum random moves (legacy, used if phase_distribution is None)
        phase_distribution: Dict mapping phase names to weights, e.g. {"opening": 0.3, "middlegame": 0.5, "endgame": 0.2}
        min_eval_cp: Minimum centipawn evaluation to accept (-200)
        max_eval_cp: Maximum centipawn evaluation to accept (+200)
        use_opening_book: Whether to use opening book moves for opening positions
        stockfish_filter_depth: Stockfish depth for quality filtering (2-4 for speed)
        cache_positions: Whether to cache and reuse high-quality positions
        cache_size: Maximum number of positions to cache
        quality_filter: Whether to filter positions by Stockfish evaluation
    """
    max_steps: int = 10000
    random_walk_gen_steps: int = 30
    phase_distribution: Optional[Dict[str, float]] = None
    min_eval_cp: int = -200
    max_eval_cp: int = 200
    use_opening_book: bool = True
    stockfish_filter_depth: int = 2
    cache_positions: bool = False
    cache_size: int = 1000
    quality_filter: bool = True


class ChessStartStatesDataset(IterableDataset):
  """
  Infinite dataset that yields high-quality FEN strings from diverse game phases.
  
  Supports quality filtering, phase-aware generation, and position caching.
  """
  def __init__(
      self,
      config: ChessDatasetConfig = ChessDatasetConfig()
  ):
      """
      Initialize dataset with quality filtering and phase diversity options.
      
      Args:
          config: ChessDatasetConfig object with all configuration parameters. 
          Defaults to ChessDatasetConfig() if no config is provided.
          Parameters in the config are:
            max_steps: Maximum number of positions to generate per epoch
            random_walk_gen_steps: Maximum random moves (legacy, used if phase_distribution is None)
            phase_distribution: Dict mapping phase names to weights, e.g. {"opening": 0.3, "middlegame": 0.5, "endgame": 0.2}
            min_eval_cp: Minimum centipawn evaluation to accept (-200)
            max_eval_cp: Maximum centipawn evaluation to accept (+200)
            use_opening_book: Whether to use opening book moves for opening positions
            stockfish_filter_depth: Stockfish depth for quality filtering (2-4 for speed)
            cache_positions: Whether to cache and reuse high-quality positions
            cache_size: Maximum number of positions to cache
            quality_filter: Whether to filter positions by Stockfish evaluation
      """
      # Use config if provided, otherwise use individual parameters or defaults
      
      self.max_steps = config.max_steps
      self.random_walk_gen_steps = config.random_walk_gen_steps
      self.phase_distribution = config.phase_distribution
      self.min_eval_cp = config.min_eval_cp
      self.max_eval_cp = config.max_eval_cp
      self.use_opening_book = config.use_opening_book
      self.stockfish_filter_depth = config.stockfish_filter_depth
      self.cache_positions = config.cache_positions
      self.cache_size = config.cache_size
      self.quality_filter = config.quality_filter
     
      # Normalize phase distribution (only if not None)
      if self.phase_distribution is not None:
          total_weight = sum(self.phase_distribution.values())
          if total_weight > 0:
              self.phase_distribution = {k: v / total_weight for k, v in self.phase_distribution.items()}
      
      # Position cache
      self._position_cache: deque = deque[Any](maxlen=self.cache_size)
      self._cache_stats = {"hits": 0, "misses": 0, "generated": 0}
      
      # Statistics tracking
      self._stats = {
          "opening": 0,
          "middlegame": 0,
          "endgame": 0,
          "filtered_out": 0,
          "total_generated": 0,
      }

  def _sample_phase(self) -> str:
      """Sample a game phase according to phase_distribution weights.
      
      Returns:
          Phase name: "opening", "middlegame", or "endgame"
      """
      rand = random.random()
      cumulative = 0.0
      for phase, weight in self.phase_distribution.items():
          cumulative += weight
          if rand <= cumulative:
              return phase
      # Fallback to middlegame
      return "middlegame"

  def _generate_position(self) -> Optional[chess.Board]:
      """Generate a single position according to configuration.
      
      Returns:
          Chess board or None if generation fails
      """
      # Check cache first
      if self.cache_positions and self._position_cache:
          if random.random() < 0.3:  # 30% chance to use cached position
              cached_pos = random.choice(self._position_cache)
              self._cache_stats["hits"] += 1
              return chess.Board(cached_pos)
          self._cache_stats["misses"] += 1
      
      # Determine phase
      if self.phase_distribution:
          phase = self._sample_phase()
      else:
          phase = None
      
      # Generate position
      if self.quality_filter:
          board = generate_quality_filtered_board(
              step_num=self.random_walk_gen_steps,
              min_eval_cp=self.min_eval_cp,
              max_eval_cp=self.max_eval_cp,
              filter_depth=self.stockfish_filter_depth,
              phase=phase
          )
      else:
          if phase:
              board = generate_position_by_phase(phase)
          else:
              board = generate_random_board(self.random_walk_gen_steps)
      
      if board is None:
          return None
      
      # Update statistics
      if not board.is_game_over():
          actual_phase = get_game_phase(board)
          self._stats[actual_phase] = self._stats.get(actual_phase, 0) + 1
          self._stats["total_generated"] += 1
          
          # Cache position if enabled
          if self.cache_positions:
              self._position_cache.append(board.fen())
              self._cache_stats["generated"] += 1
      
      return board

  def get_stats(self) -> Dict:
      """Get statistics about generated positions.
      
      Returns:
          Dictionary with statistics
      """
      stats = self._stats.copy()
      if self.cache_positions:
          stats["cache"] = self._cache_stats.copy()
          stats["cache"]["size"] = len(self._position_cache)
      return stats

  def __iter__(self):
      worker_info = torch.utils.data.get_worker_info()

      # Determine how many steps this worker should generate
      if worker_info is not None:
          # Split work among workers
          num_workers = worker_info.num_workers
          worker_id = worker_info.id
          per_worker = self.max_steps // num_workers
          # Give remainder to the last worker
          if worker_id == num_workers - 1:
              per_worker += self.max_steps % num_workers

          # Set deterministic seed per worker for reproducibility and isolation
          worker_seed = 42 + worker_id * 1000
          random.seed(worker_seed)
          torch.manual_seed(worker_seed)
          steps_to_generate = per_worker
      else:
          # Single process mode
          steps_to_generate = self.max_steps

      # Generate positions for this worker's share
      for step in range(steps_to_generate):
          board = self._generate_position()
          if board is not None and not board.is_game_over():
              yield board.fen()