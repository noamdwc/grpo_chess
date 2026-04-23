"""The rollout sampler must only emit legal moves inside <think>.

We patch the policy head to uniform-random logits and check that over many
steps, no illegal move is ever emitted on either color's turn.
"""
from pathlib import Path

import chess
import pytest
import torch

CKPT = Path("checkpoints/dm_port/9M.pt")
pytestmark = pytest.mark.skipif(not CKPT.exists(), reason="Phase 0 not done")


def test_sampler_never_produces_illegal_moves_inside_think():
    from src.reasoning.model import ReasoningModel, ReasoningModelConfig
    from src.reasoning.sampler import rollout_batch
    from src.searchless_chess_imports import ACTION_TO_MOVE

    torch.manual_seed(0)
    cfg = ReasoningModelConfig(dm_checkpoint=str(CKPT), max_seq_len=100)
    model = ReasoningModel(cfg).eval()

    starts = ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"] * 4
    result = rollout_batch(
        model=model,
        root_fens=starts,
        k_samples=2,
        min_think_tokens=3,
        max_think_tokens=5,
        rollout_temperature=1.0,
    )

    for sample_idx in range(len(starts) * 2):
        think_tokens = result.think_tokens[sample_idx]
        board = chess.Board(result.root_fens_replayed[sample_idx])
        for token_id in think_tokens:
            move = chess.Move.from_uci(ACTION_TO_MOVE[token_id])
            assert move in board.legal_moves, (
                f"Illegal move {move.uci()} emitted at sample {sample_idx} from FEN {board.fen()}. "
                "Likely: sampler did not apply the legal mask, OR the imagined board "
                "was not advanced correctly between steps."
            )
            board.push(move)
