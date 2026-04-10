"""The PPO loss mask zeros out padding and non-sampled positions."""
import torch

from src.grpo_logic.reasoning_loss import ppo_surrogate


def test_padding_positions_do_not_contribute():
    log_probs_old = torch.zeros((2, 10))
    log_probs_new = torch.zeros((2, 10))
    log_probs_new[0, 5] = -0.5
    advantages = torch.tensor([[1.0] * 10, [1.0] * 10])
    sampled_mask = torch.zeros((2, 10), dtype=torch.bool)
    sampled_mask[0, 2:5] = True
    loss = ppo_surrogate(
        log_probs_old=log_probs_old,
        log_probs_new=log_probs_new,
        advantages=advantages,
        sampled_mask=sampled_mask,
        clip_ratio=0.2,
    )
    assert loss.item() == pytest_approx(-1.0, rel=1e-5)


def pytest_approx(x, rel):
    import pytest

    return pytest.approx(x, rel=rel)
