"""<end_think> position is recorded correctly and is never before min_think_tokens."""
from pathlib import Path

import pytest
import torch

from src.reasoning.tokens import END_THINK_ID, FEN_LEN

CKPT = Path("checkpoints/dm_port/9M.pt")
pytestmark = pytest.mark.skipif(not CKPT.exists(), reason="Phase 0 not done")


def test_end_think_position_respects_minimum():
    from src.reasoning.model import ReasoningModel, ReasoningModelConfig
    from src.reasoning.sampler import rollout_batch

    torch.manual_seed(1)
    cfg = ReasoningModelConfig(dm_checkpoint=str(CKPT), max_seq_len=120)
    model = ReasoningModel(cfg).eval()
    result = rollout_batch(
        model=model,
        root_fens=["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"],
        k_samples=4,
        min_think_tokens=3,
        max_think_tokens=6,
        rollout_temperature=1.0,
    )
    for idx in range(result.token_sequences.shape[0]):
        end_pos = int(result.end_think_positions[idx])
        assert end_pos >= FEN_LEN + 1 + 3, f"end_think at position {end_pos} before min_think_tokens"
        assert result.token_sequences[idx, end_pos].item() == END_THINK_ID
