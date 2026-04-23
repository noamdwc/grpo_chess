"""Regression tests for masked log-prob recomputation in reasoning GRPO."""
from pathlib import Path

import pytest
import torch

from src.reasoning.model import ReasoningModelOutput

CKPT = Path("checkpoints/dm_port/9M.pt")


def test_logprob_sequence_applies_legal_token_mask():
    from src.reasoning.model import ReasoningModel

    logits = torch.tensor(
        [[[2.0, 0.0, -1.0], [0.5, 1.5, -2.0]]],
        dtype=torch.float32,
    )
    tokens = torch.tensor([[0, 1]], dtype=torch.long)
    seq_lens = torch.tensor([2], dtype=torch.long)
    legal_token_mask = torch.tensor(
        [[[True, False, False], [False, True, False]]],
        dtype=torch.bool,
    )

    class FakeModel:
        def forward(self, tokens, seq_lens):
            del tokens, seq_lens
            return ReasoningModelOutput(logits=logits, hidden=torch.zeros((1, 2, 1)))

    out = ReasoningModel.logprob_sequence(
        FakeModel(),
        tokens,
        seq_lens,
        temperature=1.0,
        legal_token_mask=legal_token_mask,
    )

    assert out.shape == tokens.shape
    assert out[0, 0].item() == pytest.approx(0.0)
    assert out[0, 1].item() == pytest.approx(0.0)


@pytest.mark.skipif(not CKPT.exists(), reason="Phase 0 not done")
def test_rollout_log_probs_match_masked_recompute_when_policy_is_unchanged():
    from src.reasoning.model import ReasoningModel, ReasoningModelConfig
    from src.reasoning.sampler import rollout_batch

    torch.manual_seed(0)
    model = ReasoningModel(ReasoningModelConfig(dm_checkpoint=str(CKPT), max_seq_len=120)).eval()

    result = rollout_batch(
        model=model,
        root_fens=["8/8/8/8/8/8/8/K6k w - - 0 1"],
        k_samples=2,
        min_think_tokens=2,
        max_think_tokens=4,
        rollout_temperature=1.0,
        move_sampling_temperature=1.0,
    )

    recomputed = model.logprob_sequence(
        result.token_sequences,
        result.seq_lens,
        temperature=1.0,
        legal_token_mask=result.legal_token_mask,
    )

    assert torch.allclose(
        recomputed[result.sampled_mask],
        result.log_probs_old[result.sampled_mask],
        atol=1e-5,
    )
