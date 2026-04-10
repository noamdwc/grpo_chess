"""real_play.play_final_and_respond plays m_final on the real root, runs the rival, and returns the post-response FEN."""
from pathlib import Path

import chess
import pytest

from src.searchless_chess_imports import MOVE_TO_ACTION

CKPT = Path("checkpoints/dm_port/9M.pt")
pytestmark = pytest.mark.skipif(not CKPT.exists(), reason="Phase 0 not done")


def test_push_and_rival_respond():
    from src.reasoning.model import ReasoningModel, ReasoningModelConfig
    from src.reasoning.real_play import play_final_and_respond

    cfg = ReasoningModelConfig(dm_checkpoint=str(CKPT), max_seq_len=120)
    model = ReasoningModel(cfg).eval()

    root = chess.Board()
    m_final = MOVE_TO_ACTION["e2e4"]
    post_fen, terminal = play_final_and_respond(
        model=model,
        root_fen=root.fen(),
        m_final_token=m_final,
    )
    assert not terminal
    assert chess.Board(post_fen).turn == chess.WHITE
