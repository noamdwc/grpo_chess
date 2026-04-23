"""Group-normalized advantages produce the expected shape and values."""
import torch

from src.grpo_logic.reasoning_loss import compute_group_advantages


def test_group_normalization_known_values():
    rewards = torch.tensor([1.0, 2.0, 3.0, 0.0, 0.0, 0.0])
    advantages = compute_group_advantages(rewards, group_size=3)
    assert torch.allclose(advantages[:3], torch.tensor([-1.0, 0.0, 1.0]), atol=1e-4)
    assert torch.allclose(advantages[3:], torch.zeros(3), atol=1e-4)
