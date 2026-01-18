import os
import chess
import chess.engine

from functools import lru_cache
from src.grpo_self_play.chess.stockfish import StockfishManager, StockfishConfig

_engine: chess.engine.SimpleEngine | None = None
_engine_pid: int | None = None
def get_engine(cfg: StockfishConfig | None = None) -> chess.engine.SimpleEngine:
    global _engine, _engine_pid
    pid = os.getpid()
    if pid != _engine_pid:
        _engine = None
        _engine_pid = pid
    if _engine is None:
        _engine = StockfishManager.get_engine(f"reward_engine_{_engine_pid}", cfg)
    return _engine


def _raw_white_reward(fen: str,movetime_ms: int, depth: int) -> float:
  if depth and depth > 0:
      limit = chess.engine.Limit(depth=depth)
  else:
      limit = chess.engine.Limit(time=movetime_ms / 1000.0)
  engine = get_engine()
  info = engine.analyse(chess.Board(fen), limit)
  score = info["score"].pov(chess.WHITE)
  if score.is_mate():
    return 10000.0 if score.mate() > 0 else -10000.0
  return float(score.score())


@lru_cache(maxsize=200_000)
def cached_raw_reward_white(fen: str, depth: int) -> float:
    """
    Cached Stockfish raw eval for a given FEN from White's POV.
    Returns centipawn score (positive = White is better).
    Only caches by depth, not movetime since movetime is not deterministic.
    """
    return _raw_white_reward(fen, movetime_ms=10, depth=depth)


def normalize_cp(raw_cp: float) -> float:
    """Normalize raw centipawn score to [-2, 2] using linear clipping."""
    return float(max(-2.0, min(2.0, raw_cp / 1000.0)))


def evaluate_fen(fen: str, pov_is_white: bool, movetime_ms: int, depth: int, normalize: bool = True):
    """
    Cached Stockfish eval for a given FEN and settings.
    Returns a normalized reward in [-1, 1].
    """
    if depth and depth > 0:
      raw_score = cached_raw_reward_white(fen, depth)
    else:
      raw_score = _raw_white_reward(fen, movetime_ms, depth)

    if not pov_is_white: # Flip sign for black POV
        raw_score = -raw_score
    # Normalize raw score using linear clipping instead of tanh
    # Linear clipping preserves gradient signal regardless of position evaluation
    # tanh was compressing differentials at higher absolute values
    if normalize:
      return normalize_cp(raw_score)
    else:
      return raw_score


def evaluate_board(board: chess.Board, pov_is_white: bool, depth: int = 16, normalize: bool = True) -> float:
    """
    Evaluate a board position from a given POV.
    Returns normalized reward in [-2, 2] or raw centipawns if normalize=False.
    """
    if board.is_game_over(claim_draw=True):
        if board.is_checkmate():
            pov_loses = (board.turn == (chess.WHITE if pov_is_white else chess.BLACK))
            raw = -10000.0 if pov_loses else 10000.0
        else:
            raw = 0.0  # Draw
        return normalize_cp(raw) if normalize else raw
    else:
        return evaluate_fen(board.fen(), pov_is_white, movetime_ms=0, depth=depth, normalize=normalize)


def reward_board(env: chess.Board, board_start: chess.Board, movetime_ms: int = 0, depth: int = 16) -> float:
    """
    Stockfish-based reward from the perspective of board_start.turn,
    matching your original intent.

    env: current board (python-chess Board)
    board_start: board at trajectory start (used for POV)
    """
    pov_is_white = (board_start.turn == chess.WHITE)
    if env.is_game_over(claim_draw=True): # Terminal state
        if env.is_checkmate():
          pov_loses = (env.turn == (chess.WHITE if pov_is_white else chess.BLACK))
          r_t = -1.0 if pov_loses else 1.0
        else:
          r_t = 0.0 # Draw
    else:
      fen_t = env.fen()
      r_t = evaluate_fen(fen_t, pov_is_white, movetime_ms, depth)

    fen_0 = board_start.fen()
    r_0 = evaluate_fen(fen_0, pov_is_white, movetime_ms, depth)
    return r_t - r_0 # Reward is the change in eval
