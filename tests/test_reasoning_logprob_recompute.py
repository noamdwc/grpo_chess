"""logprob_sequence: correctness of PPO recompute.

Key invariant: at position t, the log-prob must depend ONLY on tokens at
positions 0..t-1 (via shift_right + prefix-LM causal mask on the thinking tail).
If this test passes, the prefix-LM construction is producing log-probs
equivalent to a sequential per-step recompute, up to numerics.
"""
from pathlib import Path

import pytest
import torch

CKPT = Path("checkpoints/dm_port/9M.pt")
pytestmark = pytest.mark.skipif(not CKPT.exists(), reason="Phase 0 not done")


def test_logprob_at_t_does_not_depend_on_tokens_after_t():
    from src.reasoning.model import ReasoningModel, ReasoningModelConfig

    torch.manual_seed(0)
    cfg = ReasoningModelConfig(dm_checkpoint=str(CKPT), max_seq_len=90)
    model = ReasoningModel(cfg).eval()

    T = 85
    seq = torch.randint(0, 1000, (1, T), dtype=torch.long)
    seq_lens = torch.tensor([T])

    # Compute log-prob at position 80 once with the real sequence,
    # then replace tokens at positions 81..T-1 with garbage and recompute.
    with torch.no_grad():
        lp_1 = model.logprob_sequence(seq, seq_lens)[0, 80].item()
        seq_perturbed = seq.clone()
        seq_perturbed[0, 81:] = 7  # garbage
        lp_2 = model.logprob_sequence(seq_perturbed, seq_lens)[0, 80].item()

    assert abs(lp_1 - lp_2) < 1e-5, (
        f"log-prob at position 80 changed ({lp_1} vs {lp_2}) when tokens "
        "after 80 were modified. The prefix-LM causal mask is broken, "
        "or DMTransformer is using the wrong attention pattern."
    )
