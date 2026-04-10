"""Output tensor shapes match the contract for various (B, K, max_think)."""
from pathlib import Path

import pytest
import torch

CKPT = Path("checkpoints/dm_port/9M.pt")
pytestmark = pytest.mark.skipif(not CKPT.exists(), reason="Phase 0 not done")


@pytest.mark.parametrize("batch_size,k_samples,max_think", [(1, 1, 2), (2, 3, 5), (3, 2, 8)])
def test_shapes(batch_size, k_samples, max_think):
    from src.reasoning.model import ReasoningModel, ReasoningModelConfig
    from src.reasoning.sampler import rollout_batch

    torch.manual_seed(3)
    cfg = ReasoningModelConfig(dm_checkpoint=str(CKPT), max_seq_len=120)
    model = ReasoningModel(cfg).eval()
    fens = ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"] * batch_size
    result = rollout_batch(
        model=model,
        root_fens=fens,
        k_samples=k_samples,
        min_think_tokens=1,
        max_think_tokens=max_think,
        rollout_temperature=1.0,
    )
    total = batch_size * k_samples
    assert result.token_sequences.shape[0] == total
    assert result.seq_lens.shape == (total,)
    assert result.log_probs_old.shape == result.token_sequences.shape
    assert result.sampled_mask.shape == result.token_sequences.shape
    assert result.end_think_positions.shape == (total,)
    assert result.final_move_positions.shape == (total,)
    assert len(result.imagined_leaf_fens) == total
    assert len(result.root_fens_replayed) == total
