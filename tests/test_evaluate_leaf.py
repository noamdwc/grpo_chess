"""evaluate_leaf returns a POV-normalized reward in [-1, 1]."""
import chess
import pytest
from types import SimpleNamespace

from src.chess.rewards import evaluate_leaf


def test_terminal_checkmate_gives_pov_loser_minus_one():
    fen = "7k/6Q1/6K1/8/8/8/8/8 b - - 0 1"
    board = chess.Board(fen)
    assert board.is_checkmate()
    reward = evaluate_leaf(fen, pov_is_white=False, mode="stockfish")
    assert reward == pytest.approx(-1.0) or reward == pytest.approx(1.0)


def test_stockfish_backend_returns_in_range():
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    reward = evaluate_leaf(fen, pov_is_white=True, mode="stockfish")
    assert -1.0 <= reward <= 1.0


def test_dm_leaf_oracle_rejects_non_action_value_checkpoint(monkeypatch):
    from src.chess import dm_leaf_oracle

    dm_leaf_oracle._ENGINE = None
    dm_leaf_oracle._BUCKET_VALUES = None

    monkeypatch.setattr(
        dm_leaf_oracle,
        "inspect_checkpoint",
        lambda path: SimpleNamespace(family="dm_behavioral_cloning", path=path),
        raising=False,
    )

    build_calls = []

    def _unexpected_build_teacher_engine(**kwargs):
        build_calls.append(kwargs)
        raise AssertionError("build_teacher_engine should not be called for non-action-value checkpoints")

    monkeypatch.setattr(
        "src.distill.teacher.build_teacher_engine",
        _unexpected_build_teacher_engine,
    )

    with pytest.raises(ValueError, match="dm_action_value"):
        dm_leaf_oracle._ensure_loaded()

    assert build_calls == []
