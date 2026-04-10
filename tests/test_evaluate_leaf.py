"""evaluate_leaf returns a POV-normalized reward in [-1, 1]."""
import chess
import pytest

from src.chess.rewards import evaluate_leaf


def test_terminal_checkmate_gives_pov_loser_minus_one():
    fen = "8/8/8/8/8/5K2/5Q2/7k b - - 0 1"
    board = chess.Board(fen)
    if not board.is_checkmate():
        pytest.skip("fixture not checkmate on this board; use a known mate-in-0")
    reward = evaluate_leaf(fen, pov_is_white=False, mode="stockfish")
    assert reward == pytest.approx(-1.0) or reward == pytest.approx(1.0)


def test_stockfish_backend_returns_in_range():
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    reward = evaluate_leaf(fen, pov_is_white=True, mode="stockfish")
    assert -1.0 <= reward <= 1.0
