"""Terminal imagined boards must be able to emit <end_think> immediately."""

from types import SimpleNamespace

import chess
import torch
import torch.nn as nn

from src.reasoning.model import ReasoningModelOutput
from src.reasoning.tokens import END_THINK_ID, FEN_LEN, TOTAL_VOCAB_SIZE
from src.searchless_chess_imports import MOVE_TO_ACTION


class _FixedLogitModel(nn.Module):
    def __init__(self, preferred_token: int) -> None:
        super().__init__()
        self._dummy = nn.Parameter(torch.zeros(()))
        self._preferred_token = preferred_token
        self.config = SimpleNamespace(max_seq_len=120)

    def forward(self, tokens: torch.Tensor, seq_lens: torch.Tensor) -> ReasoningModelOutput:
        batch, seq_len = tokens.shape
        del seq_lens
        logits = torch.zeros((batch, seq_len, TOTAL_VOCAB_SIZE), dtype=torch.float32, device=tokens.device)
        logits[..., self._preferred_token] = 100.0
        logits[..., END_THINK_ID] = 10.0
        hidden = torch.zeros((batch, seq_len, 1), dtype=torch.float32, device=tokens.device)
        return ReasoningModelOutput(logits=logits, hidden=hidden)


def test_terminal_imagined_board_can_end_think_before_minimum():
    from src.reasoning.sampler import rollout_batch

    root = "8/8/8/8/8/k7/8/KQ6 w - - 0 1"
    terminal_move = "b1b5"
    model = _FixedLogitModel(MOVE_TO_ACTION[terminal_move]).eval()

    result = rollout_batch(
        model=model,
        root_fens=[root],
        k_samples=1,
        min_think_tokens=2,
        max_think_tokens=2,
        rollout_temperature=1.0,
        move_sampling_temperature=1.0,
    )

    assert result.think_tokens == [[MOVE_TO_ACTION[terminal_move]]]
    assert result.end_think_positions.tolist() == [FEN_LEN + 2]
    assert result.token_sequences[0, FEN_LEN + 1].item() == MOVE_TO_ACTION[terminal_move]
    assert result.token_sequences[0, FEN_LEN + 2].item() == END_THINK_ID

    imagined_board = chess.Board(result.imagined_leaf_fens[0])
    assert imagined_board.is_game_over(claim_draw=True)
    assert imagined_board.is_stalemate()
