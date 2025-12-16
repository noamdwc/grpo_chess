import chess
import chess.engine
import math

from functools import lru_cache
from src.grpo_self_play.chess.stockfish import StockfishManager, StockfishConfig

_engine: chess.engine.SimpleEngine | None = None
def get_engine(cfg: StockfishConfig | None = None) -> chess.engine.SimpleEngine:
    global _engine
    if _engine is None:
        _engine = StockfishManager.get_engine("reward_engine")
    return _engine


@lru_cache(maxsize=200_000)
def _eval_fen_cached(fen: str, pov_is_white: bool, movetime_ms: int, depth: int):
    """
    Cached Stockfish eval for a given FEN and settings.
    Returns a normalized reward in [-1, 1].
    """
    board = chess.Board(fen)
    engine = get_engine()

    if depth and depth > 0:
      limit = chess.engine.Limit(depth=depth)
    else:
      limit = chess.engine.Limit(time=movetime_ms / 1000.0)

    info = engine.analyse(board, limit)
    pov_color = chess.WHITE if pov_is_white else chess.BLACK
    score = info["score"].pov(pov_color)

    if score.is_mate():
      m = score.mate()
      if m is None:
        return 0.0
      return 1.0 if m > 0 else -1.0

    cp = score.score() # centi-pawn score
    if cp is None:
      return 0.0

    # Normalize centipawns to [-1,1] smoothly
    # Scale chosen so ~400cp is already a strong signal
    return float(math.tanh(cp / 400.0))

def reward_board(env: chess.Board, board_start: chess.Board, movetime_ms: int = 10, depth: int = 0):
    """
    Stockfish-based reward from the perspective of board_start.turn,
    matching your original intent.

    env: current board (python-chess Board)
    board_start: board at trajectory start (used for POV)
    """
    pov_is_white = (board_start.turn == chess.WHITE)
    fen = env.fen()
    return _eval_fen_cached(fen, pov_is_white, movetime_ms, depth)
    

def old_reward_board(env, board_start):
  score = 0
  piece_vals = {chess.PAWN:1, chess.KNIGHT:3, chess.BISHOP:3, chess.ROOK:5, chess.QUEEN:9, chess.KING:0}
  for sq, piece in env.piece_map().items():
      val = piece_vals.get(piece.piece_type, 0)
      if piece.color == board_start.turn: # Friendly piece
          score += val
      else:
          score -= val
  return score # Normalize to [-1, 1] (max material difference is 39 points)