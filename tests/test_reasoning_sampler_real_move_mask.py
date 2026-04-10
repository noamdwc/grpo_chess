"""m_final must be legal on the REAL root, not on the imagined leaf."""
from pathlib import Path

import chess
import pytest
import torch

from src.searchless_chess_imports import ACTION_TO_MOVE

CKPT = Path("checkpoints/dm_port/9M.pt")
pytestmark = pytest.mark.skipif(not CKPT.exists(), reason="Phase 0 not done")


def test_m_final_is_legal_on_real_root():
    from src.reasoning.model import ReasoningModel, ReasoningModelConfig
    from src.reasoning.sampler import rollout_batch

    torch.manual_seed(2)
    cfg = ReasoningModelConfig(dm_checkpoint=str(CKPT), max_seq_len=120)
    model = ReasoningModel(cfg).eval()
    root = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    result = rollout_batch(
        model=model,
        root_fens=[root],
        k_samples=8,
        min_think_tokens=2,
        max_think_tokens=4,
        rollout_temperature=1.0,
    )
    for idx in range(8):
        final_pos = int(result.final_move_positions[idx])
        token_id = int(result.token_sequences[idx, final_pos])
        move = chess.Move.from_uci(ACTION_TO_MOVE[token_id])
        assert move in chess.Board(root).legal_moves, (
            f"m_final {move.uci()} not legal on real root; sampler is masking "
            "off the imagined leaf instead of the real root."
        )
